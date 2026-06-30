//! Minimal CoolProp C-API bindings, loaded at runtime via libloading so the same
//! plugin binary works on macOS / Linux / Windows by shipping that platform's
//! libCoolProp next to it (.dylib / .so / .dll) -- no build-time linking, no rpath
//! juggling, uniform across OSes.
//!
//! THREAD-SAFETY MODEL (the whole reason this can be parallel):
//!  * CoolProp's heavy math runs on a per-handle AbstractState and is pure --
//!    safe to run from many threads at once AS LONG AS each thread owns its own
//!    handle (a state caches its last flash; never share one across threads).
//!  * The only shared mutable state in the C-API is the global handle table
//!    (touched by factory/free) and one-time fluid-library init. We serialize
//!    ONLY handle create/destroy with a narrow Mutex; the hot path
//!    (update_and_1_out) takes NO lock, so it parallelizes. CoolProp 8.0 hardened
//!    internal thread-safety, so the narrow lock is belt-and-suspenders.
//!  * Every call passes its OWN errcode + message buffer (caller-owned, on the
//!    stack), so error reporting is not a shared-state hazard.
//!
//! FFI SAFETY (this is the only module with `unsafe`; `lib.rs` has none). Every
//! `unsafe` block here calls a loaded C function and upholds, by construction:
//!   - strings are `CString`s kept alive across the call (no dangling / interior NUL);
//!   - array pointers come from slices whose length is passed alongside and
//!     validated (`out.len()` == inputs);
//!   - the `AbstractState` handle is owned by `State` and freed exactly once (RAII),
//!     never used after free, never shared across threads;
//!   - errcode / message buffers are stack-local and caller-owned.
//!
//! `clippy::undocumented_unsafe_blocks` (Cargo.toml) requires a `// SAFETY:` note
//! on every block, so unsafe cannot be added without a justification.

use libloading::Library;
use std::ffi::{c_char, c_double, c_long, CString};
use std::sync::Mutex;

const BUFLEN: c_long = 1024;

type FactoryFn = unsafe extern "C" fn(*const c_char, *const c_char, *mut c_long, *mut c_char, c_long) -> c_long;
type FreeFn = unsafe extern "C" fn(c_long, *mut c_long, *mut c_char, c_long);
type Batch1Fn = unsafe extern "C" fn(
    c_long,
    c_long,
    *const c_double,
    *const c_double,
    c_long,
    c_long,
    *mut c_double,
    *mut c_long,
    *mut c_char,
    c_long,
);
type SetFracFn = unsafe extern "C" fn(c_long, *const c_double, c_long, *mut c_long, *mut c_char, c_long);
type SpecPhaseFn = unsafe extern "C" fn(c_long, *const c_char, *mut c_long, *mut c_char, c_long);
type IndexFn = unsafe extern "C" fn(*const c_char) -> c_long;
type HAPropsSIFn = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    c_double,
    *const c_char,
    c_double,
    *const c_char,
    c_double,
) -> c_double;

#[derive(Debug)]
pub struct CpError(pub String);

fn cstr(s: &str) -> Result<CString, CpError> {
    CString::new(s).map_err(|_| CpError(format!("interior NUL in {s:?}")))
}

fn read_msg(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// A loaded libCoolProp + resolved C-API entry points. Cheap to share (`&`)
/// across threads; construct once per process. (fn pointers, Library and Mutex
/// are all Send+Sync, so this is auto Send+Sync.)
pub struct CoolProp {
    _lib: Library, // kept alive for the lifetime of the fn pointers below
    factory: FactoryFn,
    free: FreeFn,
    batch1: Batch1Fn,
    set_fractions: SetFracFn,
    specify_phase: SpecPhaseFn,
    get_param_index: IndexFn,
    ha_props_si: HAPropsSIFn,
    /// NARROW lock: guards ONLY handle create/destroy (the shared handle table),
    /// never the evaluation path -- so concurrent flashing stays parallel.
    handle_lock: Mutex<()>,
}

impl CoolProp {
    pub fn load(path: &str) -> Result<Self, CpError> {
        // SAFETY: dlopen'ing an arbitrary path is unsafe -- the caller asserts
        // `path` is a real libCoolProp. A bad path errors here, not later.
        let lib = unsafe { Library::new(path) }.map_err(|e| CpError(format!("dlopen {path}: {e}")))?;

        macro_rules! sym {
            ($t:ty, $name:literal) => {
                *lib.get::<$t>($name)
                    .map_err(|e| CpError(format!("missing symbol {}: {e}", String::from_utf8_lossy($name))))?
            };
        }
        // SAFETY: resolving symbols from the freshly-loaded `lib`. Each is looked
        // up by its exact C name; a missing symbol is a hard error. The derefs
        // yield plain `extern "C"` fn pointers that do not borrow `lib`, so `lib`
        // can be moved into `_lib` afterwards and the pointers stay valid for its
        // lifetime.
        let (factory, free, batch1, set_fractions, specify_phase, get_param_index, ha_props_si) = unsafe {
            (
                sym!(FactoryFn, b"AbstractState_factory\0"),
                sym!(FreeFn, b"AbstractState_free\0"),
                sym!(Batch1Fn, b"AbstractState_update_and_1_out\0"),
                sym!(SetFracFn, b"AbstractState_set_fractions\0"),
                sym!(SpecPhaseFn, b"AbstractState_specify_phase\0"),
                sym!(IndexFn, b"get_param_index\0"),
                sym!(HAPropsSIFn, b"HAPropsSI\0"),
            )
        };
        Ok(CoolProp {
            _lib: lib,
            factory,
            free,
            batch1,
            set_fractions,
            specify_phase,
            get_param_index,
            ha_props_si,
            handle_lock: Mutex::new(()),
        })
    }

    pub fn param_index(&self, name: &str) -> Result<c_long, CpError> {
        let c = cstr(name)?;
        // SAFETY: `get_param_index` reads a NUL-terminated C string; `c` outlives the call.
        let i = unsafe { (self.get_param_index)(c.as_ptr()) };
        if i < 0 {
            return Err(CpError(format!("unknown parameter {name:?}")));
        }
        Ok(i)
    }

    /// Batched humid air: loops the scalar HAPropsSI (no AbstractState handle, no
    /// global lock; CoolProp 8.0 has per-thread HumidAir backends). CStrings
    /// resolved once; runs lock-free so independent HA expressions parallelize.
    #[allow(clippy::too_many_arguments)] // mirrors HAPropsSI: 3 name/value pairs + output + out slice
    pub fn ha_props_si_batch(
        &self,
        output: &str,
        n1: &str,
        v1: &[f64],
        n2: &str,
        v2: &[f64],
        n3: &str,
        v3: &[f64],
        out: &mut [f64],
    ) -> Result<(), CpError> {
        let n = out.len();
        if v1.len() != n || v2.len() != n || v3.len() != n {
            return Err(CpError("ha_props_si_batch: length mismatch".into()));
        }
        let (o, a, b, c) = (cstr(output)?, cstr(n1)?, cstr(n2)?, cstr(n3)?);
        for i in 0..n {
            // SAFETY: all four name pointers are CStrings alive for the whole loop;
            // the f64 values are passed by value. HAPropsSI has no shared mutable
            // state we touch (8.0 per-thread backends), so the loop is lock-free.
            let val =
                unsafe { (self.ha_props_si)(o.as_ptr(), a.as_ptr(), v1[i], b.as_ptr(), v2[i], c.as_ptr(), v3[i]) };
            out[i] = if val.is_finite() { val } else { f64::NAN }; // CoolProp returns _HUGE on error
        }
        Ok(())
    }

    /// Create a low-level state. `fluid` is a single fluid ("Water") or a mixture
    /// as one string ("CO2&O2"). Guarded by the narrow handle lock.
    pub fn state(&self, backend: &str, fluid: &str) -> Result<State<'_>, CpError> {
        let (b, f) = (cstr(backend)?, cstr(fluid)?);
        let mut err: c_long = 0;
        let mut msg = [0 as c_char; BUFLEN as usize];
        let handle = {
            let _g = self.handle_lock.lock().unwrap();
            // SAFETY: `b`/`f` are CStrings alive across the call; `err`/`msg` are
            // valid stack buffers (`msg` has BUFLEN bytes). The handle-table write
            // is serialised by `handle_lock` held here.
            unsafe { (self.factory)(b.as_ptr(), f.as_ptr(), &mut err, msg.as_mut_ptr(), BUFLEN) }
        };
        if err != 0 {
            return Err(CpError(format!("factory({backend},{fluid}): {}", read_msg(&msg))));
        }
        Ok(State { cp: self, handle })
    }
}

/// An owned AbstractState handle. RAII-frees on drop (under the narrow lock).
/// Must be used by a single thread (CoolProp caches the last flash per handle).
pub struct State<'a> {
    cp: &'a CoolProp,
    handle: c_long,
}

impl State<'_> {
    /// Set the (mole-basis) mixture composition on this state.
    pub fn set_mole_fractions(&mut self, fracs: &[f64]) -> Result<(), CpError> {
        let mut err: c_long = 0;
        let mut msg = [0 as c_char; BUFLEN as usize];
        // SAFETY: `fracs` ptr + len describe the same slice; `self.handle` is a
        // live handle we own; `err`/`msg` are valid stack buffers.
        unsafe {
            (self.cp.set_fractions)(
                self.handle,
                fracs.as_ptr(),
                fracs.len() as c_long,
                &mut err,
                msg.as_mut_ptr(),
                BUFLEN,
            );
        }
        if err != 0 {
            return Err(CpError(format!("set_fractions: {}", read_msg(&msg))));
        }
        Ok(())
    }

    /// Pin the phase (assume_phase), skipping the phase-determination flash.
    pub fn specify_phase(&mut self, phase: &str) -> Result<(), CpError> {
        let p = cstr(phase)?;
        let mut err: c_long = 0;
        let mut msg = [0 as c_char; BUFLEN as usize];
        // SAFETY: `p` is a CString alive across the call; `self.handle` is live; buffers valid.
        unsafe { (self.cp.specify_phase)(self.handle, p.as_ptr(), &mut err, msg.as_mut_ptr(), BUFLEN) };
        if err != 0 {
            return Err(CpError(format!("specify_phase({phase}): {}", read_msg(&msg))));
        }
        Ok(())
    }

    /// BATCHED: one call, `out.len()` flashes, one output. The handle-table lock
    /// was taken once at construction; THIS call takes NO lock -> parallel across
    /// states. The hot path. `out` is filled with finite values or NaN: rows whose
    /// flash fails are left as the pre-filled NaN, and any `_HUGE`/inf is collapsed
    /// to NaN (so failed rows become NaN -> null, like the Python path). The errcode
    /// is intentionally ignored -- the output array is the result; construction
    /// errors already surfaced from `state()`.
    pub fn update_and_1_out(
        &mut self,
        input_pair: i64, // cast to c_long at the FFI call: c_long is i32 on Windows (LLP64)
        v1: &[f64],
        v2: &[f64],
        out_key: c_long,
        out: &mut [f64],
    ) -> Result<(), CpError> {
        let n = out.len();
        if v1.len() != n || v2.len() != n {
            return Err(CpError("update_and_1_out: length mismatch".into()));
        }
        out.fill(f64::NAN);
        let mut err: c_long = 0;
        let mut msg = [0 as c_char; BUFLEN as usize];
        // SAFETY: v1/v2/out pointers each describe a slice of length `n` (checked
        // above) and `n` is passed to C; `self.handle` is live; buffers valid. The
        // C++ loop runs on this state only (per-thread handle), so it is lock-free.
        unsafe {
            (self.cp.batch1)(
                self.handle,
                input_pair as c_long,
                v1.as_ptr(),
                v2.as_ptr(),
                n as c_long,
                out_key,
                out.as_mut_ptr(),
                &mut err,
                msg.as_mut_ptr(),
                BUFLEN,
            );
        }
        for x in out.iter_mut() {
            if !x.is_finite() {
                *x = f64::NAN;
            }
        }
        Ok(())
    }
}

impl Drop for State<'_> {
    fn drop(&mut self) {
        let _g = self.cp.handle_lock.lock().unwrap();
        let mut err: c_long = 0;
        let mut msg = [0 as c_char; BUFLEN as usize];
        // SAFETY: frees a handle we own exactly once (Drop runs once); the
        // handle-table write is serialised by `handle_lock` held here.
        unsafe { (self.cp.free)(self.handle, &mut err, msg.as_mut_ptr(), BUFLEN) };
    }
}

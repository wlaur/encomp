//! Minimal CoolProp C-API bindings, loaded at runtime via libloading so the same
//! plugin binary works on macOS / Linux / Windows by shipping that platform's
//! libCoolProp next to it (.dylib / .so / .dll) -- no build-time linking, no rpath
//! juggling, uniform across OSes.
//!
//! THREAD-SAFETY MODEL (the whole reason this can be parallel):
//!  * CoolProp's heavy math runs on a per-handle AbstractState and is pure --
//!    safe to run from many threads at once AS LONG AS each thread owns its own
//!    handle (a state caches its last flash; never share one across threads).
//!  * The shared mutable state in the C-API is the global handle table (touched
//!    by factory/free) and CoolProp's lazy init (process-global fluid library +
//!    index maps, and per-backend/fluid init such as TTSE/BICUBIC tables). We
//!    serialize handle create/destroy with a narrow Mutex, and the per-config
//!    lazy init with a warmup registry (`ensure_warmed_*`): the first flash of
//!    each (backend, fluid) runs once single-threaded before the pool flashes it.
//!    The hot path (update_and_1_out) takes NO lock, so it parallelizes. CoolProp
//!    8.0 hardened internal thread-safety, so the locks are belt-and-suspenders.
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
use std::collections::HashSet;
use std::ffi::{CString, c_char, c_double, c_int, c_long};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

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
type UpdateFn = unsafe extern "C" fn(c_long, c_long, c_double, c_double, *mut c_long, *mut c_char, c_long);
type KeyedOutputFn = unsafe extern "C" fn(c_long, c_long, *mut c_long, *mut c_char, c_long) -> c_double;
type IndexFn = unsafe extern "C" fn(*const c_char) -> c_long;
type StringFn = unsafe extern "C" fn(*const c_char, *mut c_char, c_int) -> c_long;
type ExtractBackendFn = unsafe extern "C" fn(*const c_char, *mut c_char, c_long, *mut c_char, c_long) -> c_int;
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

fn lock_mutex<'a, T>(mutex: &'a Mutex<T>, name: &str) -> Result<MutexGuard<'a, T>, CpError> {
    mutex.lock().map_err(|_| CpError(format!("{name} lock was poisoned")))
}

fn read_lock<'a, T>(lock: &'a RwLock<T>, name: &str) -> Result<RwLockReadGuard<'a, T>, CpError> {
    lock.read()
        .map_err(|_| CpError(format!("{name} read lock was poisoned")))
}

fn write_lock<'a, T>(lock: &'a RwLock<T>, name: &str) -> Result<RwLockWriteGuard<'a, T>, CpError> {
    lock.write()
        .map_err(|_| CpError(format!("{name} write lock was poisoned")))
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
    update: UpdateFn,
    keyed_output: KeyedOutputFn,
    get_param_index: IndexFn,
    get_input_pair_index: IndexFn,
    get_global_param_string: StringFn,
    get_parameter_information_string: StringFn,
    extract_backend: ExtractBackendFn,
    ha_props_si: HAPropsSIFn,
    /// NARROW lock: guards ONLY handle create/destroy (the shared handle table),
    /// never the evaluation path -- so concurrent flashing stays parallel.
    handle_lock: Mutex<()>,
    /// keys ("F\0<backend>\0<fluid>", or "HA") already warmed up. Read-locked on the hot
    /// path (concurrent, no contention once warm); write-locked only for a first-time warmup.
    warmed: RwLock<HashSet<String>>,
    handles_created: AtomicU64,
    handles_freed: AtomicU64,
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
        let (
            factory,
            free,
            batch1,
            set_fractions,
            specify_phase,
            update,
            keyed_output,
            get_param_index,
            get_input_pair_index,
            get_global_param_string,
            get_parameter_information_string,
            extract_backend,
            ha_props_si,
        ) = unsafe {
            (
                sym!(FactoryFn, b"AbstractState_factory\0"),
                sym!(FreeFn, b"AbstractState_free\0"),
                sym!(Batch1Fn, b"AbstractState_update_and_1_out\0"),
                sym!(SetFracFn, b"AbstractState_set_fractions\0"),
                sym!(SpecPhaseFn, b"AbstractState_specify_phase\0"),
                sym!(UpdateFn, b"AbstractState_update\0"),
                sym!(KeyedOutputFn, b"AbstractState_keyed_output\0"),
                sym!(IndexFn, b"get_param_index\0"),
                sym!(IndexFn, b"get_input_pair_index\0"),
                sym!(StringFn, b"get_global_param_string\0"),
                sym!(StringFn, b"get_parameter_information_string\0"),
                sym!(ExtractBackendFn, b"C_extract_backend\0"),
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
            update,
            keyed_output,
            get_param_index,
            get_input_pair_index,
            get_global_param_string,
            get_parameter_information_string,
            extract_backend,
            ha_props_si,
            handle_lock: Mutex::new(()),
            warmed: RwLock::new(HashSet::new()),
            handles_created: AtomicU64::new(0),
            handles_freed: AtomicU64::new(0),
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

    pub fn input_pair_index(&self, name: &str) -> Result<i64, CpError> {
        let c = cstr(name)?;
        // SAFETY: `get_input_pair_index` reads a NUL-terminated C string; `c` outlives the call.
        let i = unsafe { (self.get_input_pair_index)(c.as_ptr()) };
        if i < 0 {
            return Err(CpError(format!("unknown input pair {name:?}")));
        }
        // return the logical index as i64 (matching update_and_1_out / kwargs.input_pair). The
        // c_long -> i64 conversion widens on Windows LLP64 (c_long is i32) and is a no-op on
        // LP64, where clippy flags it as useless -- hence the targeted allow.
        #[allow(clippy::useless_conversion)]
        let index = i64::from(i);
        Ok(index)
    }

    /// Read a process-global metadata string such as the bundled library version.
    /// Successful calls use only caller-owned buffers. Errors deliberately return a
    /// narrow Rust message rather than consulting CoolProp's process-global error
    /// outbox, whose text is not safe to associate with one concurrent caller.
    pub fn global_param_string(&self, name: &str) -> Result<String, CpError> {
        let name = cstr(name)?;
        let mut out = [0 as c_char; BUFLEN as usize];
        // SAFETY: `name` is NUL terminated and alive for the call; `out` is a writable
        // BUFLEN-byte caller-owned buffer, and BUFLEN fits the C `int` parameter.
        let ok = unsafe { (self.get_global_param_string)(name.as_ptr(), out.as_mut_ptr(), BUFLEN as c_int) };
        if ok != 1 {
            return Err(CpError("CoolProp global metadata lookup failed".into()));
        }
        Ok(read_msg(&out))
    }

    /// Return one parameter-information field (currently used for SI units).
    pub fn parameter_information(&self, name: &str, field: &str) -> Result<String, CpError> {
        let name = cstr(name)?;
        let field = cstr(field)?;
        let mut out = [0 as c_char; BUFLEN as usize];
        let field_bytes = field.as_bytes_with_nul();
        if field_bytes.len() > out.len() {
            return Err(CpError("CoolProp parameter-information selector is too long".into()));
        }
        for (dest, source) in out.iter_mut().zip(field_bytes) {
            *dest = *source as c_char;
        }
        // CoolProp uses this buffer as both input (the information selector) and
        // output (the resulting string). Both pointers remain valid for the call.
        // SAFETY: `name` is NUL terminated; `out` is initialized with a NUL-terminated
        // selector and provides BUFLEN writable bytes.
        let ok = unsafe { (self.get_parameter_information_string)(name.as_ptr(), out.as_mut_ptr(), BUFLEN as c_int) };
        if ok != 1 {
            return Err(CpError(format!(
                "unknown parameter {name:?} or information field {field:?}"
            )));
        }
        Ok(read_msg(&out))
    }

    /// Split an optional ``BACKEND::`` prefix using CoolProp's own parser.
    pub fn extract_backend(&self, name: &str) -> Result<(String, String), CpError> {
        let name = cstr(name)?;
        let mut backend = [0 as c_char; BUFLEN as usize];
        let mut fluid = [0 as c_char; BUFLEN as usize];
        // SAFETY: `name` is NUL terminated; both output buffers are caller-owned,
        // writable, and their exact lengths are supplied.
        let code =
            unsafe { (self.extract_backend)(name.as_ptr(), backend.as_mut_ptr(), BUFLEN, fluid.as_mut_ptr(), BUFLEN) };
        if code != 0 {
            return Err(CpError("CoolProp backend/fluid name exceeds the native buffer".into()));
        }
        Ok((read_msg(&backend), read_msg(&fluid)))
    }

    /// Warm up ONE (backend, fluid) config, lazily. The FIRST call (any config) also runs
    /// CoolProp's process-global lazy inits (fluid library, index maps, flash machinery);
    /// every distinct config additionally pays one warmup flash so its backend-specific init
    /// -- notably TTSE/BICUBIC table building and REFPROP globals -- runs ONCE, single-threaded,
    /// before the thread pool flashes that config. A backend/fluid that is never used is never
    /// warmed (so an IF97-only run never touches HEOS, and vice versa). Best-effort: the warmup
    /// flash's own error is ignored -- triggering the init is the point, and a genuinely invalid
    /// config surfaces its error from the real evaluation.
    ///
    /// The key deliberately EXCLUDES fractions and assumed phase: CoolProp's lazy global
    /// init (fluid library, binary-pair data, TTSE/BICUBIC tables) depends only on the
    /// (backend, fluid-list) pair, while set_fractions / specify_phase are per-handle
    /// state with no global side effects -- so one warmup per (backend, fluid) covers
    /// every composition and phase of that config.
    pub fn ensure_warmed_fluid(&self, backend: &str, fluid: &str) -> Result<(), CpError> {
        let key = format!("F\0{backend}\0{fluid}");
        self.ensure_warmed(&key, || {
            if let Ok(mut st) = self.state(backend, fluid)
                && let (Ok(pair), Ok(okey)) = (self.input_pair_index("PT_INPUTS"), self.param_index("DMASS"))
            {
                let mut out = [0.0f64];
                let _ = st.update_and_1_out(pair, &[101_325.0], &[300.0], okey, &mut out);
            }
        })
    }

    /// One-time warmup for the humid-air (HAPropsSI) path, so its global init runs
    /// single-threaded before the pool loops HAPropsSI. Keyed once (no per-fluid variation).
    pub fn ensure_warmed_humid_air(&self) -> Result<(), CpError> {
        self.ensure_warmed("HA", || {
            let mut out = [0.0f64];
            let _ = self.ha_props_si_batch("W", "P", &[101_325.0], "T", &[293.15], "R", &[0.5], &mut out);
        })
    }

    /// Run `warm` exactly once for `key`. Fast path: a shared read lock + membership test, so
    /// once a key is warm all threads pass concurrently. Slow path (first time per key): the
    /// write lock is held ACROSS `warm`, so the first-ever warmup serialises CoolProp's global
    /// init, and each new config's backend init completes before any parallel flash of it. No
    /// re-entrancy (warm never calls back here) and no lock-order cycle (warm takes only the
    /// separate handle_lock), so this cannot deadlock.
    fn ensure_warmed(&self, key: &str, warm: impl FnOnce()) -> Result<(), CpError> {
        if read_lock(&self.warmed, "CoolProp warmup registry")?.contains(key) {
            return Ok(());
        }
        let mut warmed = write_lock(&self.warmed, "CoolProp warmup registry")?;
        if warmed.contains(key) {
            return Ok(());
        }
        warm();
        warmed.insert(key.to_string());
        Ok(())
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
            out[i] = if valid_result(val) { val } else { f64::NAN }; // CoolProp returns _HUGE on error
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
            let _g = lock_mutex(&self.handle_lock, "CoolProp handle table")?;
            // SAFETY: `b`/`f` are CStrings alive across the call; `err`/`msg` are
            // valid stack buffers (`msg` has BUFLEN bytes). The handle-table write
            // is serialised by `handle_lock` held here.
            unsafe { (self.factory)(b.as_ptr(), f.as_ptr(), &mut err, msg.as_mut_ptr(), BUFLEN) }
        };
        if err != 0 {
            return Err(CpError(format!("factory({backend},{fluid}): {}", read_msg(&msg))));
        }
        self.handles_created.fetch_add(1, Ordering::Relaxed);
        Ok(State { cp: self, handle })
    }

    pub fn handle_counts(&self) -> (u64, u64) {
        (
            self.handles_created.load(Ordering::Relaxed),
            self.handles_freed.load(Ordering::Relaxed),
        )
    }
}

/// CoolProp's C wrapper returns `_HUGE` (currently 1e300) for several failures.
/// No physical property encomp supports approaches this sentinel.
fn valid_result(value: f64) -> bool {
    value.is_finite() && value.abs() < 1e290
}

/// An owned AbstractState handle. RAII-frees on drop (under the narrow lock).
/// Must be used by a single thread (CoolProp caches the last flash per handle).
pub struct State<'a> {
    cp: &'a CoolProp,
    handle: c_long,
}

impl State<'_> {
    /// Set the composition. Wraps the C-API `AbstractState_set_fractions`, which selects the
    /// basis per fluid -- mole for the mixture EOS (HEOS/PR/...), and for the incompressible
    /// backend the fluid's own mass or volume basis -- so one call is correct for every fluid.
    pub fn set_fractions(&mut self, fracs: &[f64]) -> Result<(), CpError> {
        let mut err: c_long = 0;
        let mut msg = [0 as c_char; BUFLEN as usize];
        // c_long is i32 on Windows (LLP64); a mixture never has that many species, but the
        // cast is checked rather than truncating
        let n = c_long::try_from(fracs.len())
            .map_err(|_| CpError("set_fractions: too many species for the C API".to_string()))?;
        // SAFETY: `fracs` ptr + len describe the same slice; `self.handle` is a
        // live handle we own; `err`/`msg` are valid stack buffers.
        unsafe {
            (self.cp.set_fractions)(self.handle, fracs.as_ptr(), n, &mut err, msg.as_mut_ptr(), BUFLEN);
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
        input_pair: i64,
        v1: &[f64],
        v2: &[f64],
        out_key: c_long,
        out: &mut [f64],
    ) -> Result<(), CpError> {
        let n = out.len();
        if v1.len() != n || v2.len() != n {
            return Err(CpError("update_and_1_out: length mismatch".into()));
        }
        // c_long is i32 on Windows (LLP64), so a chunk longer than i32::MAX would be passed
        // to C truncated (and possibly negative). Refuse rather than read out of bounds.
        let n_c = c_long::try_from(n).map_err(|_| {
            CpError(format!(
                "update_and_1_out: chunk of {n} rows exceeds the C API length limit"
            ))
        })?;
        let pair = c_long::try_from(input_pair)
            .map_err(|_| CpError(format!("input pair {input_pair} exceeds the C API integer range")))?;
        out.fill(f64::NAN);
        let mut err: c_long = 0;
        let mut msg = [0 as c_char; BUFLEN as usize];
        // SAFETY: v1/v2/out pointers each describe a slice of length `n` (checked
        // above) and `n` is passed to C; `self.handle` is live; buffers valid. The
        // C++ loop runs on this state only (per-thread handle), so it is lock-free.
        unsafe {
            (self.cp.batch1)(
                self.handle,
                pair,
                v1.as_ptr(),
                v2.as_ptr(),
                n_c,
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

    /// Fast scalar evaluation through the same batched C entry point as the array
    /// plugin. The successful path is one FFI call. If that call leaves its output
    /// untouched, retry through the individually error-reporting update/keyed-output
    /// functions so Python receives the existing narrow `ValueError` behavior without
    /// putting the global PropsSI error outbox back on the hot path.
    pub fn update_and_1_out_scalar(
        &mut self,
        input_pair: i64,
        value1: f64,
        value2: f64,
        out_key: c_long,
    ) -> Result<f64, CpError> {
        let mut out = [f64::NAN];
        self.update_and_1_out(input_pair, &[value1], &[value2], out_key, &mut out)?;
        if valid_result(out[0]) {
            return Ok(out[0]);
        }

        let pair = c_long::try_from(input_pair)
            .map_err(|_| CpError(format!("input pair {input_pair} exceeds the C API integer range")))?;
        let mut err: c_long = 0;
        let mut msg = [0 as c_char; BUFLEN as usize];
        // SAFETY: `self.handle` is live and thread-owned; scalar values are passed by
        // value; error buffers are caller-owned and valid for BUFLEN bytes.
        unsafe {
            (self.cp.update)(self.handle, pair, value1, value2, &mut err, msg.as_mut_ptr(), BUFLEN);
        }
        if err != 0 {
            return Err(CpError(read_msg(&msg)));
        }

        err = 0;
        msg.fill(0);
        // SAFETY: `self.handle` remains live after the successful update; the output
        // key came from this same library; error buffers are caller-owned.
        let value = unsafe { (self.cp.keyed_output)(self.handle, out_key, &mut err, msg.as_mut_ptr(), BUFLEN) };
        if err != 0 {
            return Err(CpError(read_msg(&msg)));
        }
        if valid_result(value) {
            Ok(value)
        } else {
            Err(CpError("CoolProp returned a non-finite result".into()))
        }
    }
}

impl Drop for State<'_> {
    fn drop(&mut self) {
        let _g = self.cp.handle_lock.lock().unwrap_or_else(|e| e.into_inner());
        let mut err: c_long = 0;
        let mut msg = [0 as c_char; BUFLEN as usize];
        // SAFETY: frees a handle we own exactly once (Drop runs once); the
        // handle-table write is serialised by `handle_lock` held here.
        unsafe { (self.cp.free)(self.handle, &mut err, msg.as_mut_ptr(), BUFLEN) };
        self.cp.handles_freed.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_msg_stops_at_nul_and_survives_non_utf8() {
        let buf: Vec<c_char> = b"error text\0garbage".iter().map(|&b| b as c_char).collect();
        assert_eq!(read_msg(&buf), "error text");
        assert_eq!(read_msg(&[0]), "");
        // a negative c_char (high byte) round-trips through the lossy decode
        let weird: Vec<c_char> = vec![b'a' as c_char, -1, 0];
        assert_eq!(read_msg(&weird), "a\u{fffd}");
    }

    #[test]
    fn cstr_rejects_interior_nul() {
        assert!(cstr("Water").is_ok());
        assert!(cstr("Wa\0ter").is_err());
    }
}

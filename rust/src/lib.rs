//! CoolProp property evaluation as a native Polars expression plugin, on the
//! CoolProp C-API bindings in `coolprop.rs`. Independent property nodes run in
//! parallel on Polars' thread pool (no GIL), via the BATCHED C-API (one
//! AbstractState_update_and_1_out per chunk) with only handle create/destroy and
//! the first-use per-config warmup locked. Per chunk the state is built once
//! (composition + assumed phase set up front), then a single batched flash runs
//! over the whole chunk.

mod coolprop;

use coolprop::{CoolProp, CpError, State};
use polars::prelude::*;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, OnceLock};

// One loaded libCoolProp per process (lib_path is constant for a session).
static CP: OnceLock<CoolProp> = OnceLock::new();
// serialises the one-time load + warmup (see `coolprop`); untouched on the hot path
static INIT: Mutex<()> = Mutex::new(());

/// Prepared scalar configurations contain only owned Rust values. Python caches
/// their integer IDs, so a hot scalar call crosses PyO3 with four integers and two
/// floats rather than repeatedly allocating backend/fluid/property strings.
#[derive(Debug, PartialEq)]
struct FluidConfig {
    backend: String,
    fluid: String,
    fractions: Option<Vec<f64>>,
    phase: Option<String>,
}

#[derive(Debug, PartialEq)]
struct HumidAirConfig {
    output: String,
    name1: String,
    name2: String,
    name3: String,
}

static FLUID_CONFIGS: Mutex<Vec<Arc<FluidConfig>>> = Mutex::new(Vec::new());
static HUMID_AIR_CONFIGS: Mutex<Vec<Arc<HumidAirConfig>>> = Mutex::new(Vec::new());

const SCALAR_STATE_CACHE_CAPACITY: usize = 16;

struct ScalarStateCache {
    /// Least recently used at the front, most recently used at the back.
    entries: VecDeque<(usize, State<'static>)>,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl ScalarStateCache {
    const fn new() -> Self {
        Self {
            entries: VecDeque::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    fn clear(&mut self) {
        // Dropping each State frees its native handle under CoolProp's narrow
        // handle-table lock. This also makes destruction directly testable.
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
        self.evictions = 0;
    }
}

thread_local! {
    /// Mutable AbstractState handles never leave their owning OS thread.
    static SCALAR_STATES: RefCell<ScalarStateCache> = const { RefCell::new(ScalarStateCache::new()) };
}

fn perr(e: CpError) -> PolarsError {
    PolarsError::ComputeError(e.0.into())
}

fn compute_error(msg: impl Into<String>) -> PolarsError {
    PolarsError::ComputeError(msg.into().into())
}

fn cp_error(e: CpError) -> PyErr {
    PyValueError::new_err(e.0)
}

fn runtime_error(message: impl Into<String>) -> PyErr {
    PyRuntimeError::new_err(message.into())
}

fn validate_input_count(kind: &str, got: usize, expected: usize) -> PolarsResult<()> {
    if got != expected {
        return Err(compute_error(format!("{kind}: expected {expected} inputs, got {got}")));
    }
    Ok(())
}

/// The extension-type name of encomp's unit-carrying dtype (encomp/polars.py
/// EXTENSION_NAME -- the two must stay in sync). Its metadata is the unit string in
/// encomp's canonical registry rendering.
const ENCOMP_UNIT_EXTENSION_NAME: &str = "encomp.unit";

/// Validate an extension-typed input (e.g. encomp's "encomp.unit" columns, whose unit
/// rides in the dtype metadata). The plugin computes on raw SI magnitudes and has no
/// units engine, so the most it can honestly do is string equality: an "encomp.unit"
/// input whose canonical unit string equals `expected_unit` (the canonical rendering of
/// the SI unit CoolProp expects, supplied by the Python wrapper) is accepted and its
/// storage values are used as-is. Anything else is refused -- silently using the
/// storage values could treat e.g. bar as Pa. The plugin's own polars registry loads
/// unknown extension types as generic (never as bare storage), so the dtype always
/// arrives intact here. Runs at schema-resolution time via the output-dtype callbacks,
/// so a bad plan fails before any flash is computed; the compute entrypoints repeat it
/// defensively.
fn validate_extension_input(kind: &str, name: &str, dtype: &DataType, expected_unit: Option<&str>) -> PolarsResult<()> {
    let DataType::Extension(typ, storage) = dtype else {
        return Ok(());
    };
    if typ.name() == ENCOMP_UNIT_EXTENSION_NAME
        && let (Some(expected), Some(carried)) = (expected_unit, typ.serialize_metadata())
    {
        if carried == expected {
            return Ok(()); // already the SI unit this input expects: storage used as-is
        }
        let display = |u: &str| {
            if u.is_empty() {
                "dimensionless".to_string()
            } else {
                format!("'{u}'")
            }
        };
        return Err(PolarsError::InvalidOperation(
            format!(
                "{kind}: input '{name}' carries unit {} but this input expects {}. Convert the \
                 column first, e.g. with encomp.polars.quantities and `.to(...)`, or pass raw \
                 SI values via `.ext.storage()`.",
                display(&carried),
                display(expected),
            )
            .into(),
        ));
    }
    Err(PolarsError::InvalidOperation(
        format!(
            "{kind}: input '{name}' has extension dtype '{typ}' (storage {storage}); the plugin \
             computes on raw SI magnitudes and does not interpret this extension type. Unwrap it \
             with `.ext.storage()` first."
        )
        .into(),
    ))
}

/// Accept only plain numeric columns (or Null, so an all-null column can flow to an
/// all-null result). Extension inputs are validated separately and compute on their
/// numeric storage. In particular, do not let Polars' non-strict f64 cast turn Boolean
/// or numeric-looking String sensor columns into physical values.
fn validate_numeric_input(kind: &str, name: &str, dtype: &DataType) -> PolarsResult<()> {
    let storage = match dtype {
        DataType::Extension(_, storage) => storage.as_ref(),
        other => other,
    };
    if storage.is_primitive_numeric() || storage == &DataType::Null {
        return Ok(());
    }
    Err(PolarsError::InvalidOperation(
        format!(
            "{kind}: input '{name}' must be a numeric column, got {dtype}. Boolean, String, nested, and temporal columns cannot represent a CoolProp state input."
        )
        .into(),
    ))
}

/// Broadcast a length-1 input to `n` (Polars passes a scalar `pl.lit` input as
/// length 1 alongside full columns; map_batches broadcasts via its struct, we
/// must do it here). No copy when the length already matches.
fn broadcast(s: &[f64], n: usize) -> PolarsResult<std::borrow::Cow<'_, [f64]>> {
    use std::borrow::Cow;
    if s.len() == n {
        Ok(Cow::Borrowed(s))
    } else if s.len() == 1 {
        Ok(Cow::Owned(vec![s[0]; n]))
    } else {
        Err(PolarsError::ComputeError(
            format!("input length {} incompatible with {n}", s.len()).into(),
        ))
    }
}

fn broadcast_len(values: &[Vec<f64>], scalar_mask: &[bool]) -> usize {
    let mut n: Option<usize> = None;
    for (i, v) in values.iter().enumerate() {
        if scalar_mask.get(i).copied().unwrap_or(false) {
            continue;
        }
        n = Some(match n {
            Some(current) => current.max(v.len()),
            None => v.len(),
        });
    }
    n.unwrap_or_else(|| values.iter().map(Vec::len).max().unwrap_or(0))
}

/// Materialize a Series as `Vec<f64>` with nulls mapped to NaN. The kernels explicitly
/// mask every row containing a non-finite input after evaluation: some backends return
/// finite state-independent constants for a NaN input instead of failing the flash.
fn to_f64(s: &Series) -> PolarsResult<Vec<f64>> {
    // to_storage: an accepted extension-typed input (validate_extension_input) computes
    // on its storage values; the identity for plain columns
    let s = s.to_storage().cast(&DataType::Float64)?;
    Ok(s.f64()?.iter().map(|o| o.unwrap_or(f64::NAN)).collect())
}

fn load_coolprop(lib_path: &str) -> Result<&'static CoolProp, CpError> {
    if let Some(c) = CP.get() {
        return Ok(c); // fast path: already initialised, no lock
    }
    // double-checked locking: the first caller dlopens the lib while the rest block, so the
    // load happens once (a plain get/set would let several threads load at once). CoolProp's
    // process-global init is NOT done here -- it is deferred to the first per-config warmup
    // (CoolProp::ensure_warmed_*), so an unused backend is never warmed.
    let _guard = INIT
        .lock()
        .map_err(|_| CpError("CoolProp init lock was poisoned".into()))?;
    if let Some(c) = CP.get() {
        return Ok(c);
    }
    let c = CoolProp::load(lib_path)?;
    CP.set(c)
        .map_err(|_| CpError("CoolProp was initialized concurrently".into()))?;
    CP.get().ok_or_else(|| CpError("CoolProp failed to initialize".into()))
}

fn coolprop(lib_path: &str) -> PolarsResult<&'static CoolProp> {
    load_coolprop(lib_path).map_err(perr)
}

fn initialized_coolprop() -> Result<&'static CoolProp, CpError> {
    CP.get()
        .ok_or_else(|| CpError("native CoolProp is not initialized; call initialize(lib_path) first".into()))
}

fn get_fluid_config(config_id: usize) -> Result<Arc<FluidConfig>, CpError> {
    FLUID_CONFIGS
        .lock()
        .map_err(|_| CpError("fluid configuration registry lock was poisoned".into()))?
        .get(config_id)
        .cloned()
        .ok_or_else(|| CpError(format!("unknown fluid configuration id {config_id}")))
}

fn get_humid_air_config(config_id: usize) -> Result<Arc<HumidAirConfig>, CpError> {
    HUMID_AIR_CONFIGS
        .lock()
        .map_err(|_| CpError("humid-air configuration registry lock was poisoned".into()))?
        .get(config_id)
        .cloned()
        .ok_or_else(|| CpError(format!("unknown humid-air configuration id {config_id}")))
}

fn configured_state(cp: &'static CoolProp, config: &FluidConfig) -> Result<State<'static>, CpError> {
    cp.ensure_warmed_fluid(&config.backend, &config.fluid)?;
    let mut state = cp.state(&config.backend, &config.fluid)?;
    if let Some(fractions) = &config.fractions {
        state.set_fractions(fractions)?;
    }
    if let Some(phase) = &config.phase {
        state.specify_phase(phase)?;
    }
    Ok(state)
}

fn evaluate_scalar_fluid(
    cp: &'static CoolProp,
    config_id: usize,
    input_pair: i64,
    value1: f64,
    value2: f64,
    output: i64,
) -> Result<f64, CpError> {
    SCALAR_STATES.with(|cache_cell| {
        let mut cache = cache_cell.borrow_mut();
        let position = cache.entries.iter().position(|(id, _)| *id == config_id);
        let mut entry = if let Some(position) = position {
            cache.hits += 1;
            cache
                .entries
                .remove(position)
                .ok_or_else(|| CpError("scalar state cache entry disappeared".into()))?
        } else {
            cache.misses += 1;
            let config = get_fluid_config(config_id)?;
            let state = configured_state(cp, &config)?;
            if cache.entries.len() == SCALAR_STATE_CACHE_CAPACITY {
                cache.entries.pop_front();
                cache.evictions += 1;
            }
            (config_id, state)
        };
        let output = std::ffi::c_long::try_from(output)
            .map_err(|_| CpError(format!("output parameter {output} exceeds the C API integer range")))?;
        let value = entry.1.update_and_1_out_scalar(input_pair, value1, value2, output);
        // A failed flash does not invalidate the AbstractState. Return it to the LRU
        // before propagating the error so intermittent bad sensor rows do not turn
        // every following scalar evaluation into a cold state construction.
        cache.entries.push_back(entry);
        value
    })
}

#[derive(Clone, Copy)]
struct ResolvedInputPair {
    first: std::ffi::c_long,
    second: std::ffi::c_long,
    pair: i64,
}

/// Mirrors CoolProp's reviewed `generate_update_pair` table, but asks the loaded
/// library for every parameter and pair number at runtime. Enum integers therefore
/// never cross a version boundary or live in encomp source.
const INPUT_PAIR_NAMES: &[(&str, &str, &str)] = &[
    ("Q", "T", "QT_INPUTS"),
    ("Qmass", "T", "QmassT_INPUTS"),
    ("P", "Q", "PQ_INPUTS"),
    ("P", "Qmass", "PQmass_INPUTS"),
    ("P", "T", "PT_INPUTS"),
    ("Dmolar", "T", "DmolarT_INPUTS"),
    ("Dmass", "T", "DmassT_INPUTS"),
    ("Hmolar", "T", "HmolarT_INPUTS"),
    ("Hmass", "T", "HmassT_INPUTS"),
    ("Smolar", "T", "SmolarT_INPUTS"),
    ("Smass", "T", "SmassT_INPUTS"),
    ("T", "Umolar", "TUmolar_INPUTS"),
    ("T", "Umass", "TUmass_INPUTS"),
    ("Dmass", "Hmass", "DmassHmass_INPUTS"),
    ("Dmolar", "Hmolar", "DmolarHmolar_INPUTS"),
    ("Dmass", "Smass", "DmassSmass_INPUTS"),
    ("Dmolar", "Smolar", "DmolarSmolar_INPUTS"),
    ("Dmass", "Umass", "DmassUmass_INPUTS"),
    ("Dmolar", "Umolar", "DmolarUmolar_INPUTS"),
    ("Dmass", "P", "DmassP_INPUTS"),
    ("Dmolar", "P", "DmolarP_INPUTS"),
    ("Dmass", "Q", "DmassQ_INPUTS"),
    ("Dmass", "Qmass", "DmassQmass_INPUTS"),
    ("Dmolar", "Q", "DmolarQ_INPUTS"),
    ("Dmolar", "Qmass", "DmolarQmass_INPUTS"),
    ("Hmass", "P", "HmassP_INPUTS"),
    ("Hmolar", "P", "HmolarP_INPUTS"),
    ("P", "Smass", "PSmass_INPUTS"),
    ("P", "Smolar", "PSmolar_INPUTS"),
    ("P", "Umass", "PUmass_INPUTS"),
    ("P", "Umolar", "PUmolar_INPUTS"),
    ("Hmass", "Smass", "HmassSmass_INPUTS"),
    ("Hmolar", "Smolar", "HmolarSmolar_INPUTS"),
    ("Smass", "Umass", "SmassUmass_INPUTS"),
    ("Smolar", "Umolar", "SmolarUmolar_INPUTS"),
];

static RESOLVED_INPUT_PAIRS: OnceLock<Result<Vec<ResolvedInputPair>, String>> = OnceLock::new();

fn resolved_input_pairs(cp: &CoolProp) -> Result<&[ResolvedInputPair], CpError> {
    RESOLVED_INPUT_PAIRS
        .get_or_init(|| {
            INPUT_PAIR_NAMES
                .iter()
                .map(|(first, second, pair)| {
                    Ok(ResolvedInputPair {
                        first: cp.param_index(first).map_err(|e| e.0)?,
                        second: cp.param_index(second).map_err(|e| e.0)?,
                        pair: cp.input_pair_index(pair).map_err(|e| e.0)?,
                    })
                })
                .collect()
        })
        .as_deref()
        .map_err(|message| CpError(message.clone()))
}

fn resolve_pair_native(cp: &CoolProp, name1: &str, name2: &str) -> Result<(i64, bool), CpError> {
    let key1 = cp.param_index(name1)?;
    let key2 = cp.param_index(name2)?;
    for resolved in resolved_input_pairs(cp)? {
        if key1 == resolved.first && key2 == resolved.second {
            return Ok((resolved.pair, false));
        }
        if key1 == resolved.second && key2 == resolved.first {
            return Ok((resolved.pair, true));
        }
    }
    Err(CpError(format!(
        "unsupported CoolProp input pair: {name1:?}, {name2:?}"
    )))
}

/// Fallback implementation of CoolProp's documented fraction grammar. Upstream
/// exposes backend splitting through C but not `extract_fractions`, so this parser
/// is parity-tested against the Python binding in the oracle CI job.
fn extract_fractions(fluid: &str) -> Result<(String, Option<Vec<f64>>), CpError> {
    if fluid.contains('[') && fluid.contains(']') {
        let parts: Vec<&str> = fluid.split('&').collect();
        let mut names = Vec::with_capacity(parts.len());
        let mut fractions = Vec::with_capacity(parts.len());
        for part in &parts {
            let without_close = part
                .strip_suffix(']')
                .ok_or_else(|| CpError(format!("Fluid entry [{part}] must end with ']' character")))?;
            let mut pieces = without_close.split('[');
            let name = pieces.next().unwrap_or_default();
            let fraction_text = pieces
                .next()
                .ok_or_else(|| CpError(format!("Could not break [{without_close}] into name/fraction")))?;
            if name.is_empty() || pieces.next().is_some() {
                return Err(CpError(format!("Could not break [{without_close}] into name/fraction")));
            }
            let fraction = fraction_text
                .parse::<f64>()
                .map_err(|_| CpError(format!("fraction [{fraction_text}] was not converted fully")))?;
            if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
                return Err(CpError(format!(
                    "fraction [{fraction_text}] was not converted to a value between 0 and 1 inclusive"
                )));
            }
            // CoolProp removes zero-fraction components from multi-fluid names, but
            // retains a zero single-fluid INCOMP concentration.
            if fraction > 10.0 * f64::EPSILON || parts.len() == 1 {
                names.push(name);
                fractions.push(fraction);
            }
        }
        return Ok((names.join("&"), Some(fractions)));
    }

    if fluid.contains('-') && fluid.contains('%') {
        let parts: Vec<&str> = fluid.split('-').collect();
        if parts.len() != 2 || parts[0].is_empty() {
            return Err(CpError(format!(
                "format of incompressible solution {fluid:?} is invalid; expected EG-20%"
            )));
        }
        let fraction_text = parts[1]
            .strip_suffix('%')
            .ok_or_else(|| CpError(format!("invalid incompressible concentration {fluid:?}")))?;
        let fraction = fraction_text
            .parse::<f64>()
            .map_err(|_| CpError(format!("invalid incompressible concentration {fluid:?}")))?
            * 0.01;
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(CpError(format!(
                "incompressible concentration {fluid:?} must be between 0% and 100% inclusive"
            )));
        }
        return Ok((parts[0].to_string(), Some(vec![fraction])));
    }

    Ok((fluid.to_string(), None))
}

#[pyfunction(name = "initialize")]
fn py_initialize(lib_path: &str) -> PyResult<()> {
    load_coolprop(lib_path).map(|_| ()).map_err(cp_error)
}

#[pyfunction(name = "parameter_index")]
fn py_parameter_index(name: &str) -> PyResult<i64> {
    let index = initialized_coolprop()
        .and_then(|cp| cp.param_index(name))
        .map_err(cp_error)?;
    #[allow(clippy::useless_conversion)]
    Ok(i64::from(index))
}

#[pyfunction(name = "parameter_information")]
fn py_parameter_information(name: &str, field: &str) -> PyResult<String> {
    initialized_coolprop()
        .and_then(|cp| cp.parameter_information(name, field))
        .map_err(cp_error)
}

#[pyfunction(name = "resolve_input_pair")]
fn py_resolve_input_pair(name1: &str, name2: &str) -> PyResult<(i64, bool)> {
    resolve_pair_native(initialized_coolprop().map_err(cp_error)?, name1, name2).map_err(cp_error)
}

#[pyfunction(name = "resolve_fluid_name")]
fn py_resolve_fluid_name(name: &str) -> PyResult<(String, String, Option<Vec<f64>>)> {
    let cp = initialized_coolprop().map_err(cp_error)?;
    let (backend, fluid) = cp.extract_backend(name).map_err(cp_error)?;
    let (fluid, fractions) = extract_fractions(&fluid).map_err(cp_error)?;
    let backend = if backend == "?" { "HEOS".to_string() } else { backend };
    Ok((backend, fluid, fractions))
}

#[pyfunction(name = "validate_fluid", signature = (backend, fluid, fractions=None))]
fn py_validate_fluid(py: Python<'_>, backend: String, fluid: String, fractions: Option<Vec<f64>>) -> PyResult<()> {
    let cp = initialized_coolprop().map_err(cp_error)?;
    py.detach(move || {
        cp.ensure_warmed_fluid(&backend, &fluid)?;
        let mut state = cp.state(&backend, &fluid)?;
        if let Some(fractions) = fractions {
            state.set_fractions(&fractions)?;
        }
        Ok(())
    })
    .map_err(cp_error)
}

#[pyfunction(name = "prepare_fluid", signature = (backend, fluid, fractions=None, phase=None))]
fn py_prepare_fluid(
    backend: String,
    fluid: String,
    fractions: Option<Vec<f64>>,
    phase: Option<String>,
) -> PyResult<usize> {
    let candidate = FluidConfig {
        backend,
        fluid,
        fractions,
        phase,
    };
    let mut configs = FLUID_CONFIGS
        .lock()
        .map_err(|_| runtime_error("fluid configuration registry lock was poisoned"))?;
    if let Some(id) = configs.iter().position(|config| config.as_ref() == &candidate) {
        return Ok(id);
    }
    let id = configs.len();
    configs.push(Arc::new(candidate));
    Ok(id)
}

#[pyfunction(name = "prepare_humid_air")]
fn py_prepare_humid_air(output: String, name1: String, name2: String, name3: String) -> PyResult<usize> {
    let candidate = HumidAirConfig {
        output,
        name1,
        name2,
        name3,
    };
    let mut configs = HUMID_AIR_CONFIGS
        .lock()
        .map_err(|_| runtime_error("humid-air configuration registry lock was poisoned"))?;
    if let Some(id) = configs.iter().position(|config| config.as_ref() == &candidate) {
        return Ok(id);
    }
    let id = configs.len();
    configs.push(Arc::new(candidate));
    Ok(id)
}

#[pyfunction(name = "fluid_scalar")]
fn py_fluid_scalar(
    py: Python<'_>,
    config_id: usize,
    input_pair: i64,
    value1: f64,
    value2: f64,
    output: i64,
) -> PyResult<f64> {
    let cp = initialized_coolprop().map_err(cp_error)?;
    py.detach(move || evaluate_scalar_fluid(cp, config_id, input_pair, value1, value2, output))
        .map_err(cp_error)
}

#[pyfunction(name = "humid_air_scalar")]
fn py_humid_air_scalar(py: Python<'_>, config_id: usize, value1: f64, value2: f64, value3: f64) -> PyResult<f64> {
    let cp = initialized_coolprop().map_err(cp_error)?;
    let config = get_humid_air_config(config_id).map_err(cp_error)?;
    py.detach(move || {
        cp.ensure_warmed_humid_air()?;
        let mut out = [f64::NAN];
        cp.ha_props_si_batch(
            &config.output,
            &config.name1,
            &[value1],
            &config.name2,
            &[value2],
            &config.name3,
            &[value3],
            &mut out,
        )?;
        Ok(out[0])
    })
    .map_err(cp_error)
}

#[pyfunction(name = "lib_version")]
fn py_lib_version() -> PyResult<String> {
    initialized_coolprop()
        .and_then(|cp| cp.global_param_string("version"))
        .map_err(cp_error)
}

#[pyfunction(name = "clear_scalar_cache")]
fn py_clear_scalar_cache() {
    SCALAR_STATES.with(|cache| cache.borrow_mut().clear());
}

#[pyfunction(name = "scalar_cache_info")]
fn py_scalar_cache_info() -> (usize, usize, u64, u64, u64) {
    SCALAR_STATES.with(|cache| {
        let cache = cache.borrow();
        (
            cache.entries.len(),
            SCALAR_STATE_CACHE_CAPACITY,
            cache.hits,
            cache.misses,
            cache.evictions,
        )
    })
}

#[pyfunction(name = "handle_counts")]
fn py_handle_counts() -> PyResult<(u64, u64)> {
    initialized_coolprop().map(CoolProp::handle_counts).map_err(cp_error)
}

/// The PyO3 API and Polars plugin C ABI intentionally coexist in this cdylib.
/// No Python object is stored in native global state; scalar handles are bounded
/// and thread-local, so the module is safe to import without enabling the GIL.
#[pymodule]
#[pyo3(gil_used = false)]
fn _internal(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_initialize, module)?)?;
    module.add_function(wrap_pyfunction!(py_parameter_index, module)?)?;
    module.add_function(wrap_pyfunction!(py_parameter_information, module)?)?;
    module.add_function(wrap_pyfunction!(py_resolve_input_pair, module)?)?;
    module.add_function(wrap_pyfunction!(py_resolve_fluid_name, module)?)?;
    module.add_function(wrap_pyfunction!(py_validate_fluid, module)?)?;
    module.add_function(wrap_pyfunction!(py_prepare_fluid, module)?)?;
    module.add_function(wrap_pyfunction!(py_prepare_humid_air, module)?)?;
    module.add_function(wrap_pyfunction!(py_fluid_scalar, module)?)?;
    module.add_function(wrap_pyfunction!(py_humid_air_scalar, module)?)?;
    module.add_function(wrap_pyfunction!(py_lib_version, module)?)?;
    module.add_function(wrap_pyfunction!(py_clear_scalar_cache, module)?)?;
    module.add_function(wrap_pyfunction!(py_scalar_cache_info, module)?)?;
    module.add_function(wrap_pyfunction!(py_handle_counts, module)?)?;
    Ok(())
}

/// Output dtype preserves the input precision: Float32 only when every non-scalar
/// input column is Float32; otherwise Float64. `scalar_mask[i] == true` marks input
/// i as a length-1 literal (the Python wrapper detects these via empty root names),
/// so a Float64 scalar does not force the whole result up to Float64. If every input
/// is a scalar, promote over all of them. CoolProp always computes in f64; this only
/// sizes the result so a Float32 pipeline keeps Float32. An accepted extension-typed
/// input counts as its storage dtype (Float32 storage keeps a Float32 pipeline).
fn output_dtype(dtypes: &[DataType], scalar_mask: &[bool]) -> DataType {
    let is_f32 = |dt: &DataType| matches!(dt.to_storage(), DataType::Float32);
    let mut saw_column = false;
    let mut all_f32 = true;
    for (i, dt) in dtypes.iter().enumerate() {
        if scalar_mask.get(i).copied().unwrap_or(false) {
            continue;
        }
        saw_column = true;
        if !is_f32(dt) {
            all_f32 = false;
        }
    }
    if !saw_column {
        all_f32 = dtypes.iter().all(is_f32);
    }
    if all_f32 { DataType::Float32 } else { DataType::Float64 }
}

fn expected_unit(expected_units: &[String], i: usize) -> Option<&str> {
    expected_units.get(i).map(String::as_str)
}

fn cp_output(input_fields: &[Field], kwargs: EvalKwargs) -> PolarsResult<Field> {
    validate_input_count("cp_evaluate", input_fields.len(), 2)?;
    for (i, f) in input_fields.iter().enumerate() {
        validate_extension_input(
            "cp_evaluate",
            f.name(),
            f.dtype(),
            expected_unit(&kwargs.expected_units, i),
        )?;
        validate_numeric_input("cp_evaluate", f.name(), f.dtype())?;
    }
    let dtypes: Vec<DataType> = input_fields.iter().map(|f| f.dtype().clone()).collect();
    Ok(Field::new(
        input_fields[0].name().clone(),
        output_dtype(&dtypes, &kwargs.scalar_mask),
    ))
}

fn ha_output(input_fields: &[Field], kwargs: HaKwargs) -> PolarsResult<Field> {
    validate_input_count("ha_evaluate", input_fields.len(), 3)?;
    for (i, f) in input_fields.iter().enumerate() {
        validate_extension_input(
            "ha_evaluate",
            f.name(),
            f.dtype(),
            expected_unit(&kwargs.expected_units, i),
        )?;
        validate_numeric_input("ha_evaluate", f.name(), f.dtype())?;
    }
    let dtypes: Vec<DataType> = input_fields.iter().map(|f| f.dtype().clone()).collect();
    Ok(Field::new(
        input_fields[0].name().clone(),
        output_dtype(&dtypes, &kwargs.scalar_mask),
    ))
}

#[derive(Deserialize)]
struct EvalKwargs {
    lib_path: String,            // absolute path to libCoolProp (.dylib/.so/.dll)
    backend: String,             // "IF97", "HEOS", ...
    fluid: String,               // "Water", or a mixture as one string "CO2&O2"
    input_pair: i64,             // CoolProp input_pairs index (resolved caller-side)
    output: String,              // CoolProp parameter name, e.g. "DMASS"
    phase: Option<String>,       // CoolProp phase string (assume_phase)
    fractions: Option<Vec<f64>>, // composition/concentration; set_fractions picks the basis per fluid
    scalar_mask: Vec<bool>,      // per-input: true = length-1 literal (neutral for output dtype)
    // per-input canonical SI unit strings (encomp registry rendering), for accepting
    // "encomp.unit" extension-typed inputs already in the expected unit; empty (the
    // serde default) refuses every extension-typed input
    #[serde(default)]
    expected_units: Vec<String>,
}

/// inputs[0], inputs[1] are the two state values already in the canonical order
/// for `input_pair` (the Python caller orders them with generate_update_pair, so
/// ARBITRARY input pairs are supported -- PT, PH, PQ, PS, ...). One batched flash
/// over the whole chunk; Polars runs independent properties concurrently.
#[polars_expr(output_type_func_with_kwargs = cp_output)]
fn cp_evaluate(inputs: &[Series], kwargs: EvalKwargs) -> PolarsResult<Series> {
    validate_input_count("cp_evaluate", inputs.len(), 2)?;
    validate_input_count("cp_evaluate scalar_mask", kwargs.scalar_mask.len(), 2)?;
    for (i, s) in inputs.iter().enumerate() {
        validate_extension_input(
            "cp_evaluate",
            s.name(),
            s.dtype(),
            expected_unit(&kwargs.expected_units, i),
        )?;
        validate_numeric_input("cp_evaluate", s.name(), s.dtype())?;
    }
    let cp = coolprop(&kwargs.lib_path)?;
    // warm THIS (backend, fluid) once, single-threaded, before the parallel flash below
    cp.ensure_warmed_fluid(&kwargs.backend, &kwargs.fluid).map_err(perr)?;
    let pair = kwargs.input_pair;
    let okey = cp.param_index(&kwargs.output).map_err(perr)?;
    // decide the output precision from the original input dtypes (before the f64 cast)
    let out_dtype = output_dtype(
        &inputs.iter().map(|s| s.dtype().clone()).collect::<Vec<_>>(),
        &kwargs.scalar_mask,
    );

    let v1v = to_f64(&inputs[0])?;
    let v2v = to_f64(&inputs[1])?;
    let values = vec![v1v, v2v];
    let n = broadcast_len(&values, &kwargs.scalar_mask);
    let v1v = &values[0];
    let v2v = &values[1];
    let v1 = broadcast(v1v, n)?;
    let v2 = broadcast(v2v, n)?;

    // state built once per chunk (handle lock held only here), then a lock-free
    // batched flash -> parallel across the independent property expressions.
    let mut st = cp.state(&kwargs.backend, &kwargs.fluid).map_err(perr)?;
    if let Some(fr) = &kwargs.fractions {
        st.set_fractions(fr).map_err(perr)?;
    }
    if let Some(ph) = &kwargs.phase {
        st.specify_phase(ph).map_err(perr)?;
    }

    let mut out = vec![0.0f64; n]; // update_and_1_out fills finite-or-NaN (failed rows -> NaN)
    st.update_and_1_out(pair, &v1, &v2, okey, &mut out).map_err(perr)?;
    for ((result, a), b) in out.iter_mut().zip(v1.iter()).zip(v2.iter()) {
        if !a.is_finite() || !b.is_finite() {
            *result = f64::NAN;
        }
    }
    nan_to_null(out).cast(&out_dtype)
}

/// Build a Float64 Series whose non-finite entries (failed / out-of-range rows, left
/// as NaN by the batched flash) are NULL, not NaN. encomp uses null as its single
/// missing-value sentinel, so emitting it here lets the Python wrapper skip a
/// `fill_nan(None)` -- that wrapper lowers to `when(is_not_nan).then(x).otherwise(null)`,
/// which references the plugin subtree more than once and, with no common-subexpression
/// elimination, re-runs the whole CoolProp flash 2-3x. One linear validity pass here is
/// negligible next to the flash it saves.
fn nan_to_null(out: Vec<f64>) -> Series {
    let ca: Float64Chunked = out.into_iter().map(|x| x.is_finite().then_some(x)).collect();
    ca.into_series().with_name(PlSmallStr::EMPTY)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_passthrough_and_scalar() {
        let full = [1.0, 2.0, 3.0];
        assert_eq!(broadcast(&full, 3).unwrap().as_ref(), &full);
        assert_eq!(broadcast(&[7.0], 3).unwrap().as_ref(), &[7.0, 7.0, 7.0]);
        assert_eq!(broadcast(&[7.0], 0).unwrap().as_ref(), &[] as &[f64]);
        assert!(broadcast(&[1.0, 2.0], 3).is_err());
    }

    #[test]
    fn broadcast_len_ignores_scalar_literals() {
        assert_eq!(broadcast_len(&[vec![1.0], vec![]], &[true, false]), 0);
        assert_eq!(broadcast_len(&[vec![1.0], vec![2.0, 3.0]], &[true, false]), 2);
        assert_eq!(broadcast_len(&[vec![1.0], vec![2.0]], &[true, true]), 1);
    }

    #[test]
    fn output_dtype_column_precision_wins() {
        let f32d = || DataType::Float32;
        let f64d = || DataType::Float64;
        // all-Float32 columns stay Float32; any Float64 column promotes
        assert_eq!(output_dtype(&[f32d(), f32d()], &[false, false]), DataType::Float32);
        assert_eq!(output_dtype(&[f32d(), f64d()], &[false, false]), DataType::Float64);
        // a Float64 scalar literal is neutral next to a Float32 column
        assert_eq!(output_dtype(&[f32d(), f64d()], &[false, true]), DataType::Float32);
        // all-scalar inputs promote over the scalars themselves
        assert_eq!(output_dtype(&[f32d(), f32d()], &[true, true]), DataType::Float32);
        assert_eq!(output_dtype(&[f32d(), f64d()], &[true, true]), DataType::Float64);
        // integer columns promote to Float64
        assert_eq!(
            output_dtype(&[DataType::Int64, f32d()], &[false, false]),
            DataType::Float64
        );
    }

    #[test]
    fn nan_to_null_maps_non_finite() {
        let s = nan_to_null(vec![1.0, f64::NAN, f64::INFINITY, -2.5]);
        assert_eq!(s.len(), 4);
        assert_eq!(s.null_count(), 2);
        let ca = s.f64().unwrap();
        assert_eq!(ca.get(0), Some(1.0));
        assert_eq!(ca.get(1), None);
        assert_eq!(ca.get(2), None);
        assert_eq!(ca.get(3), Some(-2.5));
    }

    #[test]
    fn fraction_parser_covers_mixtures_and_incompressibles() {
        assert_eq!(extract_fractions("Water").unwrap(), ("Water".into(), None));
        assert_eq!(
            extract_fractions("CO2[0.5]&O2[0.5]").unwrap(),
            ("CO2&O2".into(), Some(vec![0.5, 0.5]))
        );
        assert_eq!(
            extract_fractions("CO2[0]&O2[1]").unwrap(),
            ("O2".into(), Some(vec![1.0]))
        );
        assert_eq!(extract_fractions("MEG[0.5]").unwrap(), ("MEG".into(), Some(vec![0.5])));
        assert_eq!(extract_fractions("EG-20%").unwrap(), ("EG".into(), Some(vec![0.2])));
        assert!(extract_fractions("CO2[1.2]&O2[-0.2]").is_err());
        assert!(extract_fractions("CO2[abc]&O2[1]").is_err());
    }
}

#[derive(Deserialize)]
struct HaKwargs {
    lib_path: String,
    output: String,         // HAPropsSI output name, e.g. "W", "Twb", "H"
    name1: String,          // e.g. "P"
    name2: String,          // e.g. "T"
    name3: String,          // e.g. "R"
    scalar_mask: Vec<bool>, // per-input: true = length-1 literal (neutral for output dtype)
    // see EvalKwargs.expected_units
    #[serde(default)]
    expected_units: Vec<String>,
}

/// Humid air: inputs[0..3] are the three HAPropsSI values for name1/name2/name3.
/// Loops the scalar HAPropsSI in Rust (lock-free) so independent humid-air
/// expressions parallelize too -- humid air is NOT left to the Python path.
#[polars_expr(output_type_func_with_kwargs = ha_output)]
fn ha_evaluate(inputs: &[Series], kwargs: HaKwargs) -> PolarsResult<Series> {
    validate_input_count("ha_evaluate", inputs.len(), 3)?;
    validate_input_count("ha_evaluate scalar_mask", kwargs.scalar_mask.len(), 3)?;
    for (i, s) in inputs.iter().enumerate() {
        validate_extension_input(
            "ha_evaluate",
            s.name(),
            s.dtype(),
            expected_unit(&kwargs.expected_units, i),
        )?;
        validate_numeric_input("ha_evaluate", s.name(), s.dtype())?;
    }
    let cp = coolprop(&kwargs.lib_path)?;
    cp.ensure_warmed_humid_air().map_err(perr)?; // one-time HAPropsSI global init, single-threaded
    let out_dtype = output_dtype(
        &inputs.iter().map(|s| s.dtype().clone()).collect::<Vec<_>>(),
        &kwargs.scalar_mask,
    );
    let v: Vec<Vec<f64>> = inputs.iter().take(3).map(to_f64).collect::<PolarsResult<_>>()?;
    let n = broadcast_len(&v, &kwargs.scalar_mask);
    let (b0, b1, b2) = (broadcast(&v[0], n)?, broadcast(&v[1], n)?, broadcast(&v[2], n)?);
    let mut out = vec![0.0f64; n];
    cp.ha_props_si_batch(
        &kwargs.output,
        &kwargs.name1,
        &b0,
        &kwargs.name2,
        &b1,
        &kwargs.name3,
        &b2,
        &mut out,
    )
    .map_err(perr)?;
    for (((result, a), b), c) in out.iter_mut().zip(b0.iter()).zip(b1.iter()).zip(b2.iter()) {
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            *result = f64::NAN;
        }
    }
    nan_to_null(out).cast(&out_dtype)
}

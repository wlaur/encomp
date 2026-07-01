//! CoolProp property evaluation as a native Polars expression plugin, on the
//! CoolProp C-API bindings in `coolprop.rs`. Independent property nodes run in
//! parallel on Polars' thread pool (no GIL), via the BATCHED C-API (one
//! AbstractState_update_and_1_out per chunk) with only handle create/destroy
//! locked. Per chunk the state is built once (composition + assumed phase set up
//! front), then a single batched flash runs over the whole chunk.

mod coolprop;

use coolprop::{CoolProp, CpError};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::sync::{Mutex, OnceLock};

// One loaded libCoolProp per process (lib_path is constant for a session).
static CP: OnceLock<CoolProp> = OnceLock::new();
// serialises the one-time load + warmup (see `coolprop`); untouched on the hot path
static INIT: Mutex<()> = Mutex::new(());

fn perr(e: CpError) -> PolarsError {
    PolarsError::ComputeError(e.0.into())
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

/// Materialize a Series as `Vec<f64>` with nulls mapped to NaN. CoolProp turns a NaN
/// input into a failed flash (NaN output -> null via `nan_to_null`), so a null input
/// row becomes a null output row -- matching the eager numpy path, rather than the old
/// `cont_slice()` hard error on the first null (nulls are ubiquitous in real frames).
fn to_f64(s: &Series) -> PolarsResult<Vec<f64>> {
    let s = s.cast(&DataType::Float64)?;
    Ok(s.f64()?.iter().map(|o| o.unwrap_or(f64::NAN)).collect())
}

fn coolprop(lib_path: &str) -> PolarsResult<&'static CoolProp> {
    if let Some(c) = CP.get() {
        return Ok(c); // fast path: already initialised, no lock
    }
    // double-checked locking: the first caller loads + warms up while the rest block, so
    // CoolProp's global init runs single-threaded (a plain get/set would let several threads
    // load-and-warm at once -- the race we avoid).
    let _guard = INIT.lock().unwrap();
    if let Some(c) = CP.get() {
        return Ok(c);
    }
    let c = CoolProp::load(lib_path).map_err(perr)?;
    c.warmup();
    let _ = CP.set(c);
    Ok(CP.get().unwrap())
}

/// Output dtype preserves the input precision: Float32 only when every non-scalar
/// input column is Float32; otherwise Float64. `scalar_mask[i] == true` marks input
/// i as a length-1 literal (the Python wrapper detects these via empty root names),
/// so a Float64 scalar does not force the whole result up to Float64. If every input
/// is a scalar, promote over all of them. CoolProp always computes in f64; this only
/// sizes the result so a Float32 pipeline keeps Float32.
fn output_dtype(dtypes: &[&DataType], scalar_mask: &[bool]) -> DataType {
    let mut saw_column = false;
    let mut all_f32 = true;
    for (i, dt) in dtypes.iter().enumerate() {
        if scalar_mask.get(i).copied().unwrap_or(false) {
            continue;
        }
        saw_column = true;
        if !matches!(dt, DataType::Float32) {
            all_f32 = false;
        }
    }
    if !saw_column {
        all_f32 = dtypes.iter().all(|dt| matches!(dt, DataType::Float32));
    }
    if all_f32 {
        DataType::Float32
    } else {
        DataType::Float64
    }
}

fn cp_output(input_fields: &[Field], kwargs: EvalKwargs) -> PolarsResult<Field> {
    let dtypes: Vec<&DataType> = input_fields.iter().map(|f| f.dtype()).collect();
    Ok(Field::new(
        input_fields[0].name().clone(),
        output_dtype(&dtypes, &kwargs.scalar_mask),
    ))
}

fn ha_output(input_fields: &[Field], kwargs: HaKwargs) -> PolarsResult<Field> {
    let dtypes: Vec<&DataType> = input_fields.iter().map(|f| f.dtype()).collect();
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
}

/// inputs[0], inputs[1] are the two state values already in the canonical order
/// for `input_pair` (the Python caller orders them with generate_update_pair, so
/// ARBITRARY input pairs are supported -- PT, PH, PQ, PS, ...). One batched flash
/// over the whole chunk; Polars runs independent properties concurrently.
#[polars_expr(output_type_func_with_kwargs = cp_output)]
fn cp_evaluate(inputs: &[Series], kwargs: EvalKwargs) -> PolarsResult<Series> {
    let cp = coolprop(&kwargs.lib_path)?;
    let pair = kwargs.input_pair;
    let okey = cp.param_index(&kwargs.output).map_err(perr)?;
    // decide the output precision from the original input dtypes (before the f64 cast)
    let out_dtype = output_dtype(
        &inputs.iter().map(|s| s.dtype()).collect::<Vec<_>>(),
        &kwargs.scalar_mask,
    );

    let v1v = to_f64(&inputs[0])?;
    let v2v = to_f64(&inputs[1])?;
    let n = v1v.len().max(v2v.len());
    let v1 = broadcast(&v1v, n)?;
    let v2 = broadcast(&v2v, n)?;

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

#[derive(Deserialize)]
struct HaKwargs {
    lib_path: String,
    output: String,         // HAPropsSI output name, e.g. "W", "Twb", "H"
    name1: String,          // e.g. "P"
    name2: String,          // e.g. "T"
    name3: String,          // e.g. "R"
    scalar_mask: Vec<bool>, // per-input: true = length-1 literal (neutral for output dtype)
}

/// Humid air: inputs[0..3] are the three HAPropsSI values for name1/name2/name3.
/// Loops the scalar HAPropsSI in Rust (lock-free) so independent humid-air
/// expressions parallelize too -- humid air is NOT left to the Python path.
#[polars_expr(output_type_func_with_kwargs = ha_output)]
fn ha_evaluate(inputs: &[Series], kwargs: HaKwargs) -> PolarsResult<Series> {
    let cp = coolprop(&kwargs.lib_path)?;
    let out_dtype = output_dtype(
        &inputs.iter().map(|s| s.dtype()).collect::<Vec<_>>(),
        &kwargs.scalar_mask,
    );
    let v: Vec<Vec<f64>> = (0..3).map(|i| to_f64(&inputs[i])).collect::<PolarsResult<_>>()?;
    let n = v[0].len().max(v[1].len()).max(v[2].len());
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
    nan_to_null(out).cast(&out_dtype)
}

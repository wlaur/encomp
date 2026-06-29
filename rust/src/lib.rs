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
use std::sync::OnceLock;

// One loaded libCoolProp per process (lib_path is constant for a session).
static CP: OnceLock<CoolProp> = OnceLock::new();

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

fn coolprop(lib_path: &str) -> PolarsResult<&'static CoolProp> {
    if let Some(c) = CP.get() {
        return Ok(c);
    }
    let c = CoolProp::load(lib_path).map_err(perr)?;
    let _ = CP.set(c); // benign race: first writer wins, others reload-and-drop
    Ok(CP.get().unwrap())
}

#[derive(Deserialize)]
struct EvalKwargs {
    lib_path: String,                 // absolute path to libCoolProp (.dylib/.so/.dll)
    backend: String,                  // "IF97", "HEOS", ...
    fluid: String,                    // "Water", or a mixture as one string "CO2&O2"
    input_pair: i64,                  // CoolProp input_pairs index (resolved caller-side)
    output: String,                   // CoolProp parameter name, e.g. "DMASS"
    phase: Option<String>,            // CoolProp phase string (assume_phase)
    mole_fractions: Option<Vec<f64>>, // constant composition (mole fractions)
}

/// inputs[0], inputs[1] are the two state values already in the canonical order
/// for `input_pair` (the Python caller orders them with generate_update_pair, so
/// ARBITRARY input pairs are supported -- PT, PH, PQ, PS, ...). One batched flash
/// over the whole chunk; Polars runs independent properties concurrently.
#[polars_expr(output_type = Float64)]
fn cp_evaluate(inputs: &[Series], kwargs: EvalKwargs) -> PolarsResult<Series> {
    let cp = coolprop(&kwargs.lib_path)?;
    let pair = kwargs.input_pair;
    let okey = cp.param_index(&kwargs.output).map_err(perr)?;

    let p1s = inputs[0].cast(&DataType::Float64)?.rechunk();
    let p2s = inputs[1].cast(&DataType::Float64)?.rechunk();
    let v1 = p1s
        .f64()?
        .cont_slice()
        .map_err(|_| PolarsError::ComputeError("null inputs not supported".into()))?;
    let v2 = p2s
        .f64()?
        .cont_slice()
        .map_err(|_| PolarsError::ComputeError("null inputs not supported".into()))?;
    let n = v1.len().max(v2.len());
    let v1 = broadcast(v1, n)?;
    let v2 = broadcast(v2, n)?;

    // state built once per chunk (handle lock held only here), then a lock-free
    // batched flash -> parallel across the independent property expressions.
    let mut st = cp.state(&kwargs.backend, &kwargs.fluid).map_err(perr)?;
    if let Some(fr) = &kwargs.mole_fractions {
        st.set_mole_fractions(fr).map_err(perr)?;
    }
    if let Some(ph) = &kwargs.phase {
        st.specify_phase(ph).map_err(perr)?;
    }

    let mut out = vec![0.0f64; n]; // update_and_1_out fills finite-or-NaN (failed rows -> NaN -> null)
    st.update_and_1_out(pair, &v1, &v2, okey, &mut out).map_err(perr)?;
    Ok(Series::new(PlSmallStr::EMPTY, out))
}

#[derive(Deserialize)]
struct HaKwargs {
    lib_path: String,
    output: String, // HAPropsSI output name, e.g. "W", "Twb", "H"
    name1: String,  // e.g. "P"
    name2: String,  // e.g. "T"
    name3: String,  // e.g. "R"
}

/// Humid air: inputs[0..3] are the three HAPropsSI values for name1/name2/name3.
/// Loops the scalar HAPropsSI in Rust (lock-free) so independent humid-air
/// expressions parallelize too -- humid air is NOT left to the Python path.
#[polars_expr(output_type = Float64)]
fn ha_evaluate(inputs: &[Series], kwargs: HaKwargs) -> PolarsResult<Series> {
    let cp = coolprop(&kwargs.lib_path)?;
    let s: Vec<_> = (0..3)
        .map(|i| inputs[i].cast(&DataType::Float64).map(|s| s.rechunk()))
        .collect::<PolarsResult<_>>()?;
    let v: Vec<&[f64]> = s
        .iter()
        .map(|x| {
            x.f64()?
                .cont_slice()
                .map_err(|_| PolarsError::ComputeError("null inputs not supported".into()))
        })
        .collect::<PolarsResult<_>>()?;
    let n = v[0].len().max(v[1].len()).max(v[2].len());
    let (b0, b1, b2) = (broadcast(v[0], n)?, broadcast(v[1], n)?, broadcast(v[2], n)?);
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
    Ok(Series::new(PlSmallStr::EMPTY, out))
}

fn main() {
    // pyo3 `extension-module` deliberately does not link libpython: the Python
    // symbols (_Py_DecRef, ...) are resolved at runtime from the host process
    // that loads this plugin. macOS rejects undefined symbols at link time, so
    // allow dynamic lookup (this is exactly what maturin does under the hood).
    // Using rustc-link-arg-cdylib keeps this scoped to our final link step, so
    // it does NOT invalidate the (expensive) polars dependency build. (libCoolProp
    // is dlopen'd at runtime, so no link/rpath to it is needed.) Linux allows
    // undefined symbols in .so by default; Windows pyo3 links python -- macOS-only.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg-cdylib=-undefined");
        println!("cargo:rustc-link-arg-cdylib=dynamic_lookup");
    }
}

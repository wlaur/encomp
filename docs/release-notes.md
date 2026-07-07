# Release Notes

## 1.6.1

This patch release fixes issues found after 1.6.0.

- Hardened `Quantity` runtime validation so subscripted constructors and the `.m` setter reject string, `None`, and non-1D magnitudes, and normalize numpy array magnitudes to `float64`.
- Fixed in-place `*=`, `/=`, `//=`, and `**=` so ndarray-backed quantities rebuild the correct dimensionality subclass instead of mutating into an inconsistent runtime type.
- Restored pickling for quantities with dynamically derived, unregistered dimensionalities.
- Fixed CoolProp expression handling for invalid input pairs, duplicate/synonym state inputs, scalar literals on empty frames, and invalid plugin inputs so they raise Python/Polars exceptions instead of panicking or returning silent nulls.
- Interpreted basis-percent unit spellings such as `mol%`, `mol-%`, `kg%`, `m3%`, and `vol%` as dimensionless percent values.
- Made gas condition literal support consistent across gas conversion helpers.
- Updated docs and CI so documentation snippets type-check under `pyright` from a source checkout.

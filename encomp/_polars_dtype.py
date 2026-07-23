"""Pint-light registration core for encomp's Polars unit extension dtype.

This module may be imported by :mod:`encomp.coolprop` before any unit computation.
Keep Pint and :mod:`encomp.units` imports inside the methods that actually need them:
registering the dtype itself must be cheap and must happen before Polars reads a file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from .units import Unit

EXTENSION_NAME = "encomp.unit"
"""Public, stable Arrow extension name for a unit-carrying column."""

_CANONICAL_UNIT_FORMAT = "~P"
"""Fixed Pint format spec for the unit string stored in dtype metadata."""


def canonical_unit_string(unit: str | Unit[Any]) -> str:
    """Return the fixed canonical rendering stored in extension metadata.

    The rendering is independent of encomp's process-wide display format because it
    is an on-disk, cross-process contract. It normalizes spelling (``"m^3"`` and
    ``"m³"``), not physical equivalence (``"Pa"`` and ``"N/m²"`` remain distinct).
    """
    from tokenize import TokenError

    from pint.errors import UndefinedUnitError

    from .units import Quantity, Unit

    if isinstance(unit, str):
        try:
            parsed = Unit(Quantity.correct_unit(unit))
        except (AssertionError, TokenError) as e:
            # Pint's tokenizer uses internal assertions for a few malformed strings.
            # Do not leak that implementation detail from a public dtype constructor.
            raise UndefinedUnitError(unit) from e
    else:
        parsed = unit
    return format(parsed, _CANONICAL_UNIT_FORMAT)


def _validate_storage(storage: pl.DataType) -> None:
    if not (storage.is_float() or storage.is_integer()):
        raise TypeError(
            f"unit dtype storage must be a float or integer dtype, got {storage!r}. "
            "Boolean and non-numeric columns cannot carry a unit, and a column that "
            "already has a unit dtype cannot be re-wrapped."
        )


class UnitDType(pl.BaseExtension):
    """Polars extension dtype carrying a physical unit as column metadata.

    A dtype instance consists of the stable extension name ``"encomp.unit"``, a
    numeric storage dtype, and a canonical unit string. Polars refuses value-producing
    arithmetic on extension columns; validated :class:`encomp.polars.QuantityFrame`
    descriptors enter Quantity unit algebra and ``QuantityFrame.derive`` returns
    results to a frame.
    """

    def __init__(self, unit: str | Unit[Any], storage: pl.DataType | None = None) -> None:
        if storage is None:
            storage = pl.Float64()
        _validate_storage(storage)
        super().__init__(EXTENSION_NAME, storage, canonical_unit_string(unit))

    @property
    def unit(self) -> Unit[Any]:
        """The column unit parsed from the extension metadata."""
        from .units import Unit

        metadata = self.ext_metadata()
        if metadata is None:
            raise ValueError(f"{EXTENSION_NAME} dtype without a unit string in its metadata: {self!r}")
        return Unit(metadata)

    def _string_repr(self) -> str:
        return f"unit[{self.ext_metadata()}]"


# This import side effect is the purpose of the module. Both encomp.polars and every
# encomp entry point that can feed extension-typed data to the native plugin import it
# before reading data, so Polars never strips the unit metadata first.
pl.register_extension_type(EXTENSION_NAME, UnitDType)

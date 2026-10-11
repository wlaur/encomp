"""Canonical dimensional algebra and deterministic overload rendering, without Pint."""

PRODUCTS: tuple[tuple[str, str, str], ...] = (
    ("MassFlow", "Time", "Mass"),
    ("VolumeFlow", "Time", "Volume"),
    ("Power", "Time", "Energy"),
    ("Velocity", "Time", "Length"),
    ("Density", "Volume", "Mass"),
    ("Length", "Length", "Area"),
    ("Length", "Area", "Volume"),
    ("MassFlow", "EnergyPerMass", "Power"),
    ("Mass", "EnergyPerMass", "Energy"),
    ("Density", "VolumeFlow", "MassFlow"),
    ("Area", "Velocity", "VolumeFlow"),
    ("MassFlow", "SpecificVolume", "VolumeFlow"),
    ("Pressure", "Volume", "Energy"),
    ("Pressure", "VolumeFlow", "Power"),
    ("Mass", "SpecificVolume", "Volume"),
    ("SpecificHeatCapacity", "TemperatureDifference", "EnergyPerMass"),
    ("Pressure", "Area", "Force"),
    ("Force", "Length", "Energy"),
    ("Force", "Velocity", "Power"),
    ("Substance", "MolarMass", "Mass"),
    ("MolarDensity", "Volume", "Substance"),
    ("HeatTransferCoefficient", "TemperatureDifference", "PowerPerArea"),
    ("HeatTransferCoefficient", "Length", "ThermalConductivity"),
)

# Temperature inverses resolve to absolute Temperature at runtime. Only these
# divisions by TemperatureDifference are part of the static contract.
DIVISIONS_ONLY: tuple[tuple[str, str, str], ...] = (
    ("EnergyPerMass", "TemperatureDifference", "SpecificHeatCapacity"),
    ("PowerPerArea", "TemperatureDifference", "HeatTransferCoefficient"),
)

RECIPROCALS: tuple[tuple[str, str], ...] = (("Density", "SpecificVolume"), ("Time", "Frequency"))

# These generic signatures and their ordering are part of Quantity's fixed contract.
_MUL_PREFIX = """\
    @overload
    def __mul__(self: Quantity[Dimensionless, float], other: Quantity[DT_, MT_]) -> Quantity[DT_, MT_]: ...
    @overload
    def __mul__(
        self: Quantity[Dimensionless, MT], other: Quantity[Dimensionless, MT]
    ) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __mul__(self: Quantity[Dimensionless, MT], other: Quantity[DT_, float]) -> Quantity[DT_, MT]: ...
    @overload
    def __mul__(self: Quantity[Dimensionless, MT], other: Quantity[DT_, MT]) -> Quantity[DT_, MT]: ...
    @overload
    def __mul__(self: Quantity[DT, MT], other: Quantity[Dimensionless, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self: Quantity[DT, MT], other: Quantity[Dimensionless, float]) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self: Quantity[DT, float], other: Quantity[Dimensionless, MT_]) -> Quantity[DT, MT_]: ...
"""

_MUL_SUFFIX = """\

    # Unknown * Unknown = Unknown
    @overload
    def __mul__(
        self: Quantity[UnknownDimensionality, MT], other: Quantity[UnknownDimensionality, MT]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __mul__(
        self: Quantity[UnknownDimensionality, MT], other: Quantity[UnknownDimensionality, float]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __mul__(
        self: Quantity[UnknownDimensionality, float], other: Quantity[UnknownDimensionality, MT_]
    ) -> Quantity[UnknownDimensionality, MT_]: ...

    @overload
    def __mul__(
        self: Quantity[DT, float], other: Quantity[Any, Numpy1DArray]
    ) -> Quantity[UnknownDimensionality, Numpy1DArray]: ...
    @overload
    def __mul__(
        self: Quantity[DT, float], other: Quantity[Any, pl.Series]
    ) -> Quantity[UnknownDimensionality, pl.Series]: ...
    @overload
    def __mul__(
        self: Quantity[DT, float], other: Quantity[Any, pl.Expr]
    ) -> Quantity[UnknownDimensionality, pl.Expr]: ...
    @overload
    def __mul__(self: Quantity[DT, float], other: Quantity[Any, float]) -> Quantity[UnknownDimensionality, float]: ...
    @overload
    def __mul__(self, other: float) -> Quantity[DT, MT]: ...
    @overload
    def __mul__(self, other: Quantity[DT_, float]) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __mul__(self, other: Quantity[DT_, MT]) -> Quantity[UnknownDimensionality, MT]: ...
"""

_DIV_PREFIX = """\
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, float], other: Quantity[Dimensionless, Numpy1DArray]
    ) -> Quantity[Dimensionless, Numpy1DArray]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, float], other: Quantity[Dimensionless, pl.Series]
    ) -> Quantity[Dimensionless, pl.Series]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, float], other: Quantity[Dimensionless, pl.Expr]
    ) -> Quantity[Dimensionless, pl.Expr]: ...

    # Unknown / Unknown = Unknown
    @overload
    def __truediv__(
        self: Quantity[UnknownDimensionality, MT], other: Quantity[UnknownDimensionality, MT]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[UnknownDimensionality, MT], other: Quantity[UnknownDimensionality, float]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[UnknownDimensionality, float], other: Quantity[UnknownDimensionality, MT_]
    ) -> Quantity[UnknownDimensionality, MT_]: ...

    @overload
    def __truediv__(self: Quantity[DT, MT], other: Quantity[Dimensionless, float]) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(self: Quantity[DT, MT], other: Quantity[Dimensionless, MT]) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(self: Quantity[DT, float], other: Quantity[Dimensionless, MT_]) -> Quantity[DT, MT_]: ...
    @overload
    def __truediv__(self: Quantity[DT, MT], other: Quantity[DT, MT]) -> Quantity[Dimensionless, MT]: ...
    @overload
    def __truediv__(self: Quantity[DT, MT], other: Quantity[DT, float]) -> Quantity[Dimensionless, MT]: ...
"""

_DIV_SUFFIX = """\

    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[DT_, float]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[Dimensionless, MT], other: Quantity[DT_, MT]
    ) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __truediv__(
        self: Quantity[DT, float], other: Quantity[Any, Numpy1DArray]
    ) -> Quantity[UnknownDimensionality, Numpy1DArray]: ...
    @overload
    def __truediv__(
        self: Quantity[DT, float], other: Quantity[Any, pl.Series]
    ) -> Quantity[UnknownDimensionality, pl.Series]: ...
    @overload
    def __truediv__(
        self: Quantity[DT, float], other: Quantity[Any, pl.Expr]
    ) -> Quantity[UnknownDimensionality, pl.Expr]: ...
    @overload
    def __truediv__(
        self: Quantity[DT, float], other: Quantity[Any, float]
    ) -> Quantity[UnknownDimensionality, float]: ...
    @overload
    def __truediv__(self, other: float) -> Quantity[DT, MT]: ...
    @overload
    def __truediv__(self, other: Quantity[DT_, float]) -> Quantity[UnknownDimensionality, MT]: ...
    @overload
    def __truediv__(self, other: Quantity[DT_, MT]) -> Quantity[UnknownDimensionality, MT]: ...
"""

_RDIV_PREFIX = """\
    @overload
    def __rtruediv__(self: Quantity[Dimensionless, MT], other: float) -> Quantity[Dimensionless, MT]: ...
"""

_RDIV_SUFFIX = """\

    @overload
    def __rtruediv__(self, other: float) -> Quantity[UnknownDimensionality, MT]: ...
"""

_COMMENT_HEADERS: dict[tuple[str, str, str], str] = {
    (
        "__truediv__",
        "EnergyPerMass",
        "TemperatureDifference",
    ): (
        "    # EnergyPerMass / TemperatureDifference = SpecificHeatCapacity\n"
        "    # (the reverse, EnergyPerMass / SpecificHeatCapacity, is intentionally omitted: it\n"
        "    # resolves to Temperature rather than TemperatureDifference since the two share the\n"
        "    # [temperature] dimension, so it is left to fall through to UnknownDimensionality)"
    ),
    (
        "__truediv__",
        "PowerPerArea",
        "TemperatureDifference",
    ): (
        "    # PowerPerArea / TemperatureDifference = HeatTransferCoefficient\n"
        "    # (the reverse, PowerPerArea / HeatTransferCoefficient, is intentionally omitted for\n"
        "    # the same reason as EnergyPerMass / SpecificHeatCapacity above: it resolves to\n"
        "    # Temperature rather than TemperatureDifference)"
    ),
}


def generated_markers(operator: str) -> tuple[str, str]:
    return (
        f"    # BEGIN GENERATED: dimensional overloads ({operator})",
        f"    # END GENERATED: dimensional overloads ({operator})",
    )


def _signature(operator: str, self_arg: str, other_arg: str, result: str) -> str:
    line = f"    def {operator}({self_arg}, {other_arg}) -> {result}: ..."
    if len(line) <= 120:
        return f"    @overload\n{line}\n"
    return f"    @overload\n    def {operator}(\n        {self_arg}, {other_arg}\n    ) -> {result}: ...\n"


def _triplet(operator: str, left: str, right: str, result: str) -> str:
    symbol = "*" if operator == "__mul__" else "/"
    header = _COMMENT_HEADERS.get((operator, left, right), f"    # {left} {symbol} {right} = {result}")
    variants = (("MT", "MT", "MT"), ("MT", "float", "MT"), ("float", "MT_", "MT_"))
    return (
        header
        + "\n"
        + "".join(
            _signature(
                operator,
                f"self: Quantity[{left}, {self_mt}]",
                f"other: Quantity[{right}, {other_mt}]",
                f"Quantity[{result}, {result_mt}]",
            )
            for self_mt, other_mt, result_mt in variants
        )
    )


def render_overload_blocks() -> dict[str, str]:
    products: list[tuple[str, str, str]] = []
    divisions: list[tuple[str, str, str]] = []
    for left, right, result in PRODUCTS:
        for pair in ((left, right, result), (right, left, result)):
            if pair not in products:
                products.append(pair)
        if "TemperatureDifference" in (left, right):
            continue
        for inverse in ((result, right, left), (result, left, right)):
            if inverse not in divisions:
                divisions.append(inverse)
    divisions.extend(DIVISIONS_ONLY)

    multiplication = _MUL_PREFIX.rstrip() + "\n\n"
    multiplication += "\n".join(_triplet("__mul__", *pair) for pair in products)
    multiplication += _MUL_SUFFIX

    division = _DIV_PREFIX.rstrip() + "\n\n"
    division += "\n".join(_triplet("__truediv__", *pair) for pair in divisions)
    division += _DIV_SUFFIX

    reflected = _RDIV_PREFIX.rstrip() + "\n\n"
    reflected += "    # reciprocals: a scalar divided by a quantity (e.g. 1 / density = specific volume)\n"
    for left, right in RECIPROCALS:
        for numerator, denominator in ((left, right), (right, left)):
            reflected += _signature(
                "__rtruediv__",
                f"self: Quantity[{numerator}, MT]",
                "other: float",
                f"Quantity[{denominator}, MT]",
            )
    reflected += _RDIV_SUFFIX + "\n"
    return {"__mul__": multiplication, "__truediv__": division, "__rtruediv__": reflected}


def replace_generated_blocks(source: str) -> str:
    for operator, block in render_overload_blocks().items():
        begin, end = generated_markers(operator)
        before, separator, tail = source.partition(begin + "\n")
        if not separator:
            raise ValueError(f"Missing generated marker: {begin.strip()}")
        _, separator, after = tail.partition(end)
        if not separator:
            raise ValueError(f"Missing generated marker: {end.strip()}")
        source = before + begin + "\n" + block + end + after
    return source

import itertools
from pathlib import Path
from textwrap import dedent, indent

import black

from encomp.utypes import (
    Dimensionality,
    Dimensionless,
    Energy,
    EnergyPerMass,
    HeatingValue,
    HigherHeatingValue,
    LowerHeatingValue,
    Mass,
    SpecificHeatCapacity,
    Temperature,
    TemperatureDifference,
)


def get_registry() -> dict:
    return {a: b for a, b in Dimensionality._registry_reversed.items() if not b.__name__.startswith("Dimensionality[")}


def get_dim(dim: type[Dimensionality]) -> type[Dimensionality]:
    if dim is Temperature:
        return TemperatureDifference

    return dim


def get_signature(
    self: type[Dimensionality] | str,
    other: type[Dimensionality] | str,
    output: type[Dimensionality] | str,
    method: str,
) -> str:
    t_self = self if isinstance(self, str) else f"Quantity[{get_dim(self).__name__}, MT]"

    t_other = other if isinstance(other, str) else f"Quantity[{get_dim(other).__name__}, Any]"

    t_output = output if isinstance(output, str) else f"Quantity[{get_dim(output).__name__}, MT]"

    return dedent(
        f"""

        @overload
        def {method}(self: {t_self}, other: {t_other}) -> {t_output}:
            ...
    """
    ).strip()


def generate_overloaded_signatures(
    dimensionalities: list[type[Dimensionality]], verbose: bool = True
) -> tuple[str, str, str]:
    _product_override: dict[tuple[type[Dimensionality], type[Dimensionality]], type[Dimensionality]] = {
        (EnergyPerMass, Mass): Energy,
        (HeatingValue, Mass): Energy,
        (LowerHeatingValue, Mass): Energy,
        (HigherHeatingValue, Mass): Energy,
    }

    _quotient_override: dict[tuple[type[Dimensionality], type[Dimensionality]], type[Dimensionality]] = {
        (Energy, Mass): EnergyPerMass,
        (EnergyPerMass, Temperature): SpecificHeatCapacity,
    }

    registry = get_registry()

    product_signatures: list[str] = []
    quotient_signatures: list[str] = []
    rquotient_signatures: list[str] = []

    if verbose:
        print(f"Generating signatures for combinations of {len(dimensionalities)} dimensionalities")

    # overrides must be added before the generated signatures
    for (self, other), output in _product_override.items():
        product_signatures.append(get_signature(other, self, output, "__mul__"))

        product_signatures.append(get_signature(self, other, output, "__mul__"))

    for (self, other), output in _quotient_override.items():
        quotient_signatures.append(get_signature(self, other, output, "__truediv__"))

    # add inversions of dimensionalities (1 / Time -> Frequency etc...)

    for self in dimensionalities:
        if self.dimensions is None or self is Dimensionless:
            continue

        inverted_dimensionality = registry.get(1 / self.dimensions)

        if inverted_dimensionality is None:
            continue

        rquotient_signatures.append(get_signature(self, "float", inverted_dimensionality, "__rtruediv__"))

    # loop over all binary combinations
    # this includes both (i, j) and (j, i)
    for self, other in itertools.product(dimensionalities, repeat=2):
        if self.dimensions is None or other.dimensions is None:
            continue

        # products of dimensionless quantities is always the
        # same dimensionality as other, this is handled with a type variable
        if self is not Dimensionless and other is not Dimensionless:
            product_dimensionality = _product_override.get(
                (self, other), registry.get(self.dimensions * other.dimensions)
            )

        else:
            product_dimensionality = None

        # division of two identical dimensionalities is always dimensionless,
        # this case can be skipped
        # also, division by dimensionless can be handled with a type variable
        if self is not other and other is not Dimensionless:
            quotient_dimensionality = _quotient_override.get(
                (self, other), registry.get(self.dimensions / other.dimensions)
            )

        else:
            quotient_dimensionality = None

        if product_dimensionality is not None:
            product_signatures.append(get_signature(self, other, product_dimensionality, "__mul__"))

        if quotient_dimensionality is not None:
            quotient_signatures.append(get_signature(self, other, quotient_dimensionality, "__truediv__"))

    product_signatures_src = black.format_file_contents(
        "\n\n".join(product_signatures), fast=False, mode=black.FileMode()
    )

    quotient_signatures_src = black.format_file_contents(
        "\n\n".join(quotient_signatures), fast=False, mode=black.FileMode()
    )

    rquotient_signatures_src = black.format_file_contents(
        "\n\n".join(rquotient_signatures), fast=False, mode=black.FileMode()
    )

    product_signatures_src = indent(product_signatures_src, prefix=" " * 4)
    quotient_signatures_src = indent(quotient_signatures_src, prefix=" " * 4)
    rquotient_signatures_src = indent(rquotient_signatures_src, prefix=" " * 4)

    if verbose:
        print(f"Generated {len(product_signatures)} product signatures")
        print(f"Generated {len(quotient_signatures)} quotient signatures")
        print(f"Generated {len(rquotient_signatures)} rquotient signatures")

    return product_signatures_src, quotient_signatures_src, rquotient_signatures_src


def get_overload_signatures() -> tuple[str, str, str]:
    dimensionalities = list(get_registry().values())
    return generate_overloaded_signatures(dimensionalities)


def write_overload_signatures() -> None:
    mul, div, rdiv = get_overload_signatures()

    with Path("generated/__mul__.py").open("w", encoding="utf-8") as f:
        f.write(mul)

    with Path("generated/__truediv__.py").open("w", encoding="utf-8") as f:
        f.write(div)

    with Path("generated/__rdiv__.py").open("w", encoding="utf-8") as f:
        f.write(rdiv)


if __name__ == "__main__":
    write_overload_signatures()

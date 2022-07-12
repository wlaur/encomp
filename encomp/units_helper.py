import itertools
from textwrap import dedent

import autopep8

from encomp.utypes import (Dimensionality,
                           Dimensionless,
                           Energy,
                           Mass,
                           Temperature,
                           HeatingValue,
                           SpecificHeatCapacity,
                           LowerHeatingValue,
                           HigherHeatingValue)


def get_signature(self: type[Dimensionality],
                  other: type[Dimensionality],
                  output: type[Dimensionality],
                  method: str) -> str:

    return dedent(
        f"""

        @overload
        def {method}(self: Quantity[{self.__name__}], other: Quantity[{other.__name__}]  # type: ignore
            ) -> Quantity[{output.__name__}]:
            ...
    """
    ).strip()


def generate_overloaded_signatures(
    dimensionalities: list[type[Dimensionality]],
    verbose: bool = True
) -> tuple[str, str]:

    _product_override: dict[
        tuple[type[Dimensionality], type[Dimensionality]],
        type[Dimensionality]
    ] = {

        (HeatingValue, Mass): Energy,
        (LowerHeatingValue, Mass): Energy,
        (HigherHeatingValue, Mass): Energy
    }

    _quotient_override: dict[
        tuple[type[Dimensionality], type[Dimensionality]],
        type[Dimensionality]
    ] = {

        (Energy, Mass): HeatingValue,
        (HeatingValue, Temperature): SpecificHeatCapacity,
    }

    registry = Dimensionality._registry_reversed

    product_signatures: list[str] = []
    quotient_signatures: list[str] = []

    if verbose:
        print(
            'Generating signatures for combinations of '
            f'{len(dimensionalities)} dimensionalities'
        )

    # overrides must be added before the generated signatures
    for (self, other), output in _product_override.items():

        product_signatures.append(
            get_signature(other, self, output, '__mul__')
        )

        product_signatures.append(
            get_signature(self, other, output, '__mul__')
        )

    for (self, other), output in _quotient_override.items():
        quotient_signatures.append(
            get_signature(self, other, output, '__truediv__')
        )

    # loop over all binary combinations
    # this includes both (i, j) and (j, i)
    for a, b in itertools.product(dimensionalities, repeat=2):

        if a.dimensions is None or b.dimensions is None:
            continue

        # products of dimensionless quantities is always the
        # same dimensionality as other, this is handled with a type variable
        if a is not Dimensionless and b is not Dimensionless:

            product_dimensionality = _product_override.get(
                (a, b),
                registry.get(a.dimensions * b.dimensions)
            )

        else:
            product_dimensionality = None

        # division of two identical dimensionalities is always dimensionless,
        # this case can be skipped
        # also, division by dimensionless can be handled with a type variable
        if a is not b and b is not Dimensionless:

            quotient_dimensionality = _quotient_override.get(
                (a, b),
                registry.get(a.dimensions / b.dimensions)
            )

        else:
            quotient_dimensionality = None

        if product_dimensionality is not None:
            product_signatures.append(
                get_signature(a, b, product_dimensionality, '__mul__')
            )

        if quotient_dimensionality is not None:
            quotient_signatures.append(
                get_signature(a, b, quotient_dimensionality, '__truediv__')
            )

    product_signatures_src = autopep8.fix_code(
        '\n\n'.join(product_signatures)
    )

    quotient_signatures_src = autopep8.fix_code(
        '\n\n'.join(quotient_signatures)
    )

    if verbose:
        print(f'Generated {len(product_signatures)} product signatures')
        print(f'Generated {len(quotient_signatures)} quotient signatures')

    return product_signatures_src, quotient_signatures_src


def get_overload_signatures() -> tuple[str, str]:
    dimensionalities = list(Dimensionality._registry_reversed.values())
    return generate_overloaded_signatures(dimensionalities)

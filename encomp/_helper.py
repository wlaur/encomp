import itertools
from typing import Union
from textwrap import dedent, indent

import autopep8

from encomp.utypes import (Dimensionality,
                           Dimensionless,
                           Energy,
                           Mass,
                           Temperature,
                           EnergyPerMass,
                           HeatingValue,
                           SpecificHeatCapacity,
                           LowerHeatingValue,
                           HigherHeatingValue)


def get_registry():

    return {
        a: b for a, b in Dimensionality._registry_reversed.items()
        if not b.__name__.startswith('Dimensionality[')
    }


def get_signature(self: Union[type[Dimensionality], str],
                  other: Union[type[Dimensionality], str],
                  output: Union[type[Dimensionality], str],
                  method: str) -> str:

    if isinstance(self, str):
        t_self = self
    else:
        t_self = f'Quantity[{self.__name__}]'

    if isinstance(other, str):
        t_other = other
    else:
        t_other = f'Quantity[{other.__name__}]'

    if isinstance(output, str):
        t_output = output
    else:
        t_output = f'Quantity[{output.__name__}]'

    return dedent(
        f"""

        @overload
        def {method}(self: {t_self}, other: {t_other}  # type: ignore
            ) -> {t_output}:
            ...
    """
    ).strip()


def generate_overloaded_signatures(
    dimensionalities: list[type[Dimensionality]],
    verbose: bool = True
) -> tuple[str, str, str]:

    _product_override: dict[
        tuple[type[Dimensionality], type[Dimensionality]],
        type[Dimensionality]
    ] = {
        (EnergyPerMass, Mass): Energy,
        (HeatingValue, Mass): Energy,
        (LowerHeatingValue, Mass): Energy,
        (HigherHeatingValue, Mass): Energy
    }

    _quotient_override: dict[
        tuple[type[Dimensionality], type[Dimensionality]],
        type[Dimensionality]
    ] = {

        (Energy, Mass): EnergyPerMass,
        (EnergyPerMass, Temperature): SpecificHeatCapacity,
    }

    registry = get_registry()

    product_signatures: list[str] = []
    quotient_signatures: list[str] = []
    rquotient_signatures: list[str] = []

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

    # add inversions of dimensionalities (1 / Time -> Frequency etc...)

    for self in dimensionalities:

        if self.dimensions is None or self is Dimensionless:
            continue

        inverted_dimensionality = registry.get(1 / self.dimensions)

        if inverted_dimensionality is None:
            continue

        rquotient_signatures.append(
            get_signature(
                self, 'MagnitudeScalar', inverted_dimensionality, '__rtruediv__'
            )
        )

    # loop over all binary combinations
    # this includes both (i, j) and (j, i)
    for self, other in itertools.product(dimensionalities, repeat=2):

        if self.dimensions is None or other.dimensions is None:
            continue

        # products of dimensionless quantities is always the
        # same dimensionality as other, this is handled with a type variable
        if self is not Dimensionless and other is not Dimensionless:

            product_dimensionality = _product_override.get(
                (self, other),
                registry.get(self.dimensions * other.dimensions)
            )

        else:
            product_dimensionality = None

        # division of two identical dimensionalities is always dimensionless,
        # this case can be skipped
        # also, division by dimensionless can be handled with a type variable
        if self is not other and other is not Dimensionless:

            quotient_dimensionality = _quotient_override.get(
                (self, other),
                registry.get(self.dimensions / other.dimensions)
            )

        else:
            quotient_dimensionality = None

        if product_dimensionality is not None:
            product_signatures.append(
                get_signature(
                    self, other, product_dimensionality, '__mul__'
                )
            )

        if quotient_dimensionality is not None:

            quotient_signatures.append(
                get_signature(
                    self, other, quotient_dimensionality, '__truediv__'
                )
            )

    product_signatures_src = autopep8.fix_code(
        '\n\n'.join(product_signatures)
    )

    quotient_signatures_src = autopep8.fix_code(
        '\n\n'.join(quotient_signatures)
    )

    rquotient_signatures_src = autopep8.fix_code(
        '\n\n'.join(rquotient_signatures)
    )

    product_signatures_src = indent(product_signatures_src, prefix=' ' * 4)
    quotient_signatures_src = indent(quotient_signatures_src, prefix=' ' * 4)
    rquotient_signatures_src = indent(rquotient_signatures_src, prefix=' ' * 4)

    if verbose:
        print(f'Generated {len(product_signatures)} product signatures')
        print(f'Generated {len(quotient_signatures)} quotient signatures')
        print(f'Generated {len(rquotient_signatures)} rquotient signatures')

    return product_signatures_src, quotient_signatures_src, rquotient_signatures_src


def get_overload_signatures() -> tuple[str, str, str]:
    dimensionalities = list(get_registry().values())
    return generate_overloaded_signatures(dimensionalities)

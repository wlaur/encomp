from encomp.units import Quantity as Q


def test_overloads() -> None:
    v1 = Q(1, "kg")

    v2 = v1 * 2  # noqa: F841

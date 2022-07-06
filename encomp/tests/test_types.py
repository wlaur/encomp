import pytest


@pytest.mark.mypy_testing
def mypy_test_invalid_assignment() -> None:

    # autopep8: off

    foo = "abc"

    foo = 123  # E: Incompatible types in assignment (expression has type "int", variable has type "str")


    # autopep8: on

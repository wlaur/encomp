import os
from pathlib import Path

from encomp.units import Quantity
from encomp.context import silence_stdout, quantity_format, temp_dir, working_dir


def test_context():

    with silence_stdout():
        print('silenced')

    with quantity_format('~Lx'):

        s = str(Quantity(1, 'kPa'))
        assert s == '\\SI[]{1}{\\kilo\\pascal}'

    s = str(Quantity(1, 'kPa'))
    assert s == '1 kPa'


def test_temp_dir():

    orig_cwd = Path.cwd()

    with temp_dir():

        temp = Path.cwd()

        assert temp != orig_cwd

    assert not temp.is_dir()


def test_working_dir():

    with temp_dir():

        parent = Path.cwd()
        sub = parent / 'sub'

        sub.mkdir(exist_ok=False)

        with working_dir(sub):
            assert Path.cwd().is_relative_to(parent)

    assert not sub.is_dir()
    assert not parent.is_dir()

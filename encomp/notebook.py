"""
Module that imports commonly used functions and
sets up the ``encomp`` Jupyter Notebook environment.
This module also works in a non-Jupyter REPL or script.
"""

import sys

# check if this is imported into a Notebook
__INTERACTIVE__ = 'ipykernel_launcher.py' in sys.argv[0]

import os
import json
import inspect
from pathlib import Path
from pprint import pprint
from typing import Union

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

try:
    import seaborn as sns
    SNS_PALETTE = sns.color_palette()
    SNS_BLUE = SNS_PALETTE[0]

except ImportError:
    pass


try:
    from IPython.core.pylabtools import print_figure
    from IPython.display import (display,
                                Markdown,
                                HTML,
                                SVG,
                                Javascript,
                                Math,
                                IFrame)
except ImportError:
    pass

from encomp.settings import SETTINGS
from encomp.misc import grid_dimensions
from encomp.sympy import sp
from encomp.units import Quantity, ureg
from encomp.units import ureg as u
from encomp.units import Quantity as Q
from encomp.utypes import *
from encomp.fluids import Fluid, Water, HumidAir


GraphicInput = Union[Path, str, plt.Figure]

if __INTERACTIVE__:

    # loads Jupyter Notebook magics: %%markdown, %%output, %%write and %read
    import encomp.magics

    plt.style.use('seaborn-notebook')  # type: ignore
    matplotlib.rcParams['font.sans-serif'] = 'Arial'  # type: ignore
    matplotlib.rcParams['font.family'] = 'sans-serif'  # type: ignore

    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats(SETTINGS.matplotlib_notebook_format)

    # this is required to get table output in PDF
    # set to False as default
    pd.options.display.latex.repr = False


def mprint(x):
    """
    Prints the input as markdown and displays it.
    """
    return display(Markdown(str(x)))


class Graphic:

    IMAGE_EXTENSIONS = ('.png', '.jpg', 'jpeg', '.svg')

    def __init__(self,
                 inp:  Union[GraphicInput, list[GraphicInput], tuple[GraphicInput, ...]],
                 width: int = 600,
                 height: int = 400,
                 nrows: int = -1,
                 ncols: int = -1):
        """
        Displays one or more graphics as notebook output.
        The graphics can be:

        * Matplotlib figures
        * Images (SVG, PNG, JPEG, JPG)
        * PDF documents or images

        Since the Notebook browser interface cannot access files outside
        of the current directory, PDF files must be placed in the same directory
        as the ``.ipynb``-file (or a subdirectory).

        Parameters
        ----------
        inp : Union[GraphicInput, list[GraphicInput], tuple[GraphicInput, ...]]
            File path to and image file (PDF, SVG, PNG, JPEG, JPG) as string or ``pathlib.Path``,
            a Matplotlib figure, raw SVG markup, or a list or tuple of these.
        width : int, optional
            Width of one graphic, by default 600
        height : int, optional
            Height of one graphic, by default 400
        nrows : int
            Number of rows in image grid in case there are
            multiple inputs, by default -1 (automatic)
        ncols : int
            Number of columns in image grid in case there are
            multiple inputs, by default -1 (automatic)
        """

        self.width = width
        self.height = height

        if isinstance(inp, tuple):
            inp = list(inp)

        if not isinstance(inp, list):
            inp = [inp]

        self.data = []

        for n in inp:

            if isinstance(n, plt.Figure):

                # TODO: convert to PDF instead of SVG to include selectable text
                n = print_figure(n, fmt='svg')
                self.data.append(['svg', n])
                continue

            if isinstance(n, Path):
                n = Graphic.get_path(n)
                self.data.append(n)  # type: ignore
                continue

            if isinstance(n, str):

                # TODO: is this the best way to check if a string is SVG data?
                # cannot use startswith since there might be metadata headers before
                if '<svg' in n:
                    self.data.append(['svg', n])
                    continue

                # file paths as strings
                try:
                    n = Graphic.get_path(n)
                    self.data.append(n)  # type: ignore
                    continue

                except Exception:
                    pass

        for i, n in enumerate(self.data):  # type: ignore

            if isinstance(n, str) and n.lower().endswith('.svg'):

                with open(n, 'r', encoding='utf-8') as f:
                    svg_data = f.read()

                self.data[i] = ['svg', svg_data]

        self.N = len(self.data)

        self.nrows, self.ncols = grid_dimensions(self.N, nrows, ncols)

    def _repr_html_(self):
        """
        Displayed by the ``text/html`` key in the notebook cell output dict.
        This will be rendered as HTML by the Notebook interface.
        """

        html_src = ['<table>']
        idx = 0

        td_width = f'{100 / self.ncols:.0f}%'

        for _ in range(self.nrows):

            # don't want alternating row colors
            html_src.append('<tr style="background: transparent">')

            for _ in range(self.ncols):

                if idx > len(self.data) - 1:

                    html_src.extend(['</tr>',
                                     '</table>'])

                    return '\n'.join(html_src)

                p = self.data[idx]
                idx += 1

                if isinstance(p, (Path, str)):

                    p = str(p)

                    if p.lower().endswith(Graphic.IMAGE_EXTENSIONS):
                        tag = 'img'

                    elif p.lower().endswith('.pdf'):
                        tag = 'iframe'

                    cell_src = (f'<td width="{td_width}"><{tag} src="{p}" '
                                f'width="{self.width}" '
                                f'height="{self.height}"></{tag}></td>')

                    html_src.append(cell_src)

                elif isinstance(p, list) and p[0] == 'svg':

                    svg_src = '<svg ' + p[1].strip().split('<svg', 1)[1]
                    cell_src = f'<td width="{td_width}">{svg_src}</td>'

                    html_src.append(cell_src)

            html_src.append('</tr>')

        html_src.append('</table>')

        return '\n'.join(html_src)

    def __repr__(self):
        """
        Displayed by the ``text/plain`` key in the notebook cell output dict.
        """

        raise NotImplementedError('TODO: implement this for PDF output')

    @staticmethod
    def get_path(p: Union[str, Path]) -> str:

        # use file paths relative to the current directory
        cwd = Path('.').absolute()

        # expand environment variables and relative file paths
        p = Path(os.path.realpath(os.path.expandvars(p))).absolute()

        # as_posix() uses / as separator (not \ as for WindowsPath)
        if cwd not in p.parents:
            return p.as_posix()

        return p.relative_to(cwd).as_posix()

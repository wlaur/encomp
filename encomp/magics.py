"""
Contains Jupyter Notebook cell and line magics.

.. note::
    This module requires ``IPython``
"""

import sys
import os
import json
from typing import Union
from pathlib import Path

from IPython.core.getipython import get_ipython
from IPython.display import display, Javascript, Markdown
from IPython.core.magic import (Magics,
                                magics_class,
                                cell_magic,
                                line_magic,
                                needs_local_scope)
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring


from encomp.misc import name_assignments
from encomp.serialize import save, load

# check if this is imported into a Notebook
__INTERACTIVE__ = 'ipykernel_launcher.py' in sys.argv[0]


@magics_class
class NotebookMagics(Magics):

    @staticmethod
    def get_js_code(metadata: dict):

        metadata_str = json.dumps(metadata)

        js_src = f"""

// find cell element and index
var cell_element = this.element.parents('.cell');
var cell_idx = Jupyter.notebook.get_cell_elements().index(cell_element);

// get the cell object
var cell = Jupyter.notebook.get_cell(cell_idx);

// define a new variable that holds the metadata
var added_metadata = {metadata_str};

// update metadata keys, create new if they do not exist
cell._metadata = {{...cell._metadata, ...added_metadata}};
"""

        return js_src

    @cell_magic
    @needs_local_scope
    def output(self, line, cell, local_ns):
        """
        Cell magic for displaying cell output (figure or table) and
        sets metadata keys that are defined inside the cell itself.

        The following cell metadata keys are supported:

        * ``caption``
        * ``caption_location``
        * ``label``
        * ``width``
        * ``height``
        * ``fontsize``
        * ``prefix``
        * ``suffix``

        The metadata key(s) are defined inside this cell
        with the names listed above

        Example of a code cell that uses this magic:

        .. code-block:: python

            %%output

            # function that returns a plt.Figure (without calling plt.show())
            fig = get_fig(...)

            caption = f'Use previously defined variable {some_var} and math mode: $x = {x:.2f}$'
            label = 'fig:example'

            # only one of width, height are supported
            width = 0.7

            # set the figure as the cell output to display it
            fig

        This metadata can be viewed and modified via the
        *View → Cell Toolbar → Edit Metadata* dialog from the notebook editor.
        """

        # these names will be evaluted from inside this cell
        metadata_keys = [
            'caption',
            'caption_location',
            'label',
            'width',
            'height',
            'fontsize'
        ]

        # remove the %%output magic (first part of the source), don't want to re-run this
        src = cell.split('%%output')[-1]

        metadata = {}

        assignments = name_assignments(src)

        for name, statement in assignments:

            if name in metadata_keys:

                exec(statement, globals(), local_ns)

                # store the key/value in a dict
                metadata[name] = local_ns[name]

        # need to call display() on JavaScript source
        # to actually execute it
        # this will not remove metadata keys, need to overwrite them
        # or remove from the Edit Metadata dialog to get rid of them completely
        display(Javascript(self.get_js_code(metadata)))

        # run the cell normally
        self.shell.run_cell(src)

        return None

    @cell_magic
    @needs_local_scope
    def markdown(self, line, cell, local_ns):
        """
        Cell magic for displaying markdown with variables.
        A simple wrapper around f-strings and the IPython Markdown class.
        """

        # update the current variable scope with the local namespace from the notebook
        locals().update(local_ns)

        # escape backslashes, don't allow \n or \r, etc inside the cell
        cell = cell.replace('\\', '\\\\')

        # construct a multiline f-string from the cell input
        f_string_source = f'f"""{cell}"""'

        # evaluate the f-string, with locals() updated
        md_source = eval(f_string_source)

        # render markdown from the evaluated f-string
        return Markdown(md_source)

    def _get_json_path(self, fpath: Union[str, Path]) -> Path:

        fpath = Path(os.path.realpath(os.path.expandvars(fpath))).absolute()

        if fpath.suffix.lower() != '.json':
            raise ValueError(f'Expected a JSON file, passed {fpath.name}')

        return fpath

    @line_magic
    @needs_local_scope
    @magic_arguments()
    @argument('input', type=str, help='Input JSON')
    def read(self, line, local_ns):
        """
        Line magic that reads variables stored with
        :py:meth:`encomp.magics.NotebookMagics.write` back into
        the local namespace
        """

        args = parse_argstring(self.read, line)
        fpath = self._get_json_path(args.input)

        if not Path(fpath).is_file():
            raise FileNotFoundError(
                f'Input JSON {fpath.name} does not exist')

        variables = load(path=fpath)
        local_ns.update(variables)

    @cell_magic
    @needs_local_scope
    @magic_arguments()
    @argument('output', type=str, help='Output JSON')
    def write(self, line, cell, local_ns):
        """
        Cell magic that writes variables defined inside the cell
        as a JSON file.
        """

        args = parse_argstring(self.write, line)
        fpath = self._get_json_path(args.output)

        exec(cell, globals(), local_ns)

        assigned_names = [n[0] for n in name_assignments(cell)]

        # only write variables assigned inside this cell
        variables = {n: local_ns[n] for n in assigned_names}
        save(variables, path=fpath)


if __INTERACTIVE__:

    # update the syntax highlighting for the %%markdown cell
    display(Javascript("""
require(['notebook/js/codecell'], function(codecell) {
codecell.CodeCell.options_default.highlight_modes['text/x-markdown'] = {'reg':[/^%%markdown/]} ;
Jupyter.notebook.events.one('kernel_ready.Kernel', function(){
Jupyter.notebook.get_cells().map(function(cell){
    if (cell.cell_type == 'code'){ cell.auto_highlight(); } }) ;
});
});"""))

    # register cell magics: %%markdown and %%output
    get_ipython().register_magics(NotebookMagics)

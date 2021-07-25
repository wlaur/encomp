import os
import fire


def build_docs():

    cmds = [

        # exclude the encomp.tests subpackage
        'sphinx-apidoc -f -o docs/source encomp encomp/tests',
        'call docs\make clean',
        'call docs\make html'
    ]

    for n in cmds:
        os.system(n)


def pip_install():

    os.system('pip install .')


def upload_pip():

    cmds = [
        'rmdir /s/q build',
        'rmdir /s/q dist',
        'python setup.py bdist_wheel',
        'twine upload dist/*'
    ]

    for n in cmds:
        os.system(n)


def main(task=None):

    if task is None:
        build_docs()
        pip_install()

    elif task == 'pip':
        upload_pip()

    elif task == 'docs':
        build_docs()

    elif task == 'install':
        pip_install()

    else:
        raise ValueError(f'Unknown task: {task}')


if __name__ == '__main__':
    fire.Fire(main)

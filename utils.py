import os
import fire


def build_docs():

    cmds = [
        'sphinx-apidoc -f -o docs\source encomp',
        'call docs\make clean',
        'call docs\make html'
    ]

    for n in cmds:
        os.system(n)


def pip_install():

    os.system('pip install .')


def main(task=None):

    if task is None:
        build_docs()
        pip_install()

    elif task == 'docs':
        build_docs()

    else:
        raise ValueError(f'Unknown task: {task}')


if __name__ == '__main__':

    fire.Fire(main)

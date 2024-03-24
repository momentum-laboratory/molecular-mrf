import sys
import subprocess
import os
import shutil

import os

from setuptools import setup
import shutil
import subprocess
import sys



def install_last_pypulseq():
    print('START installing of pypulseq')
    cur_dir = os.path.dirname(__file__)
    os.chdir('cest_mrf')
    # print(os.getcwd())
    # print(os.path.dirname(__file__))
    # print(os.path.join(os.getcwd(),'pypulseq'))
    # if shutil.which('git') is None:
    #     raise Exception(f'cest_mrf: Git is not installed on your system. Please install Git.')
    # cloned = subprocess.call(['git', 'clone','--branch', 'master' , 'https://github.com/imr-framework/pypulseq'])
    # installed = subprocess.call([sys.executable, '-m', 'pip',  'install', '-e','.'], cwd=os.path.join(os.path.dirname(__file__),'pypulseq'))
    # reseted = subprocess.call(['git', 'reset','--hard', 'cc9ccfb'], cwd=os.path.join(os.path.dirname(__file__),'pypulseq') )
    # os.remove(os.path.join(os.path.dirname(__file__), os.path.join('pypulseq','pypulseq','Sequence','write_seq.py')))
    # shutil.copy(os.path.join(os.path.dirname(__file__),'write_seq.py'), os.path.join(os.path.dirname(__file__),os.path.join('pypulseq','pypulseq','Sequence','write_seq.py')))
    # #os.remove('write_seq.py')

    if shutil.which('git') is None:
        raise Exception(f'cest_mrf: Git is not installed on your system. Please install Git.')
    cloned = subprocess.call(['git', 'clone','--branch', 'master' , 'https://github.com/imr-framework/pypulseq'])
    installed = subprocess.call([sys.executable, '-m', 'pip',  'install', '-e','.'], cwd=os.path.join(os.getcwd(),'pypulseq'))
    reseted = subprocess.call(['git', 'reset','--hard', 'cc9ccfb'], cwd=os.path.join(os.getcwd(),'pypulseq') )
    os.remove(os.path.join(os.getcwd(), os.path.join('pypulseq','pypulseq','Sequence','write_seq.py')))
    shutil.copy(os.path.join(os.getcwd(),'write_seq.py'), os.path.join(os.getcwd(),os.path.join('pypulseq','pypulseq','Sequence','write_seq.py')))
    #os.remove('write_seq.py')

    os.chdir(cur_dir)
    print('pypulseq successfully installed')


from setuptools import setup, find_packages


def cest_mrf_install(cest_mrf_src: str = os.path.join(__file__, 'cest_mrf'), options = None) -> bool:
    print(f'cest_mrf: start installation')
    # print(f"{os.path.dirname(os.getcwd())}")
    # check_install = subprocess.call([sys.executable, '-m', 'pip', 'install', *options, f'--prefix={os.path.dirname(os.getcwd())}', 'cest_mrf/'])

    install_last_pypulseq()

    setup(
        name='cest_mrf',
        author='Nikita Vladimirov',
        author_email='nikitav@mail.tau.ac.il',
        version='0.4',
        description='Python code to use C++ pulseq-CEST to simulate MRI signal and MRF dictionary generation.',
        install_requires=['bmctool==0.5.0', 'numpy==1.19.5', 'scipy==1.10.0', 'PyYAML==6.0', 'sigpy==0.1.22', 'ipykernel==6.29.0', 'tqdm==4.66.2', 'h5py==3.10.0'],
        keywords='MRI, Bloch, CEST, simulations',
        packages=find_packages(),
        # packages=['cest_mrf'],
        # package={"": "."},
        # package_dir={'cest_mrf': '.'},
        include_package_data=True,
        python_requires='>=3.9'
    )

    return True
    # return True if check_install == 0 else False

def check_sim_package_exists() -> bool:
    reqs = subprocess.call([sys.executable, '-m', 'pip', 'show', 'BMCSimulator'])
    return True if reqs is not None else False

def sim_setup(sim_path: str):
    print(f'BMCSimulator: start installation')
    if not check_sim_package_exists():
        print(f'BMCSimulator: package already installed. Proceeding to next step.')
        return True
    else:
        dist_path = os.path.join(sim_path, 'dist')
        check_dist = subprocess.call(['pip', 'install', '--find-links='+dist_path, 'BMCSimulator'])
        if check_dist == 0:
            print('BMCSimulator: package successfully installed using a pre-compiled distribution.')
            return True
        else:
            print('BMCSimulator: searching a matching pre-compiled distribution.')

        print('BMCSimulator: no matching pre-compiled distribution found. Trying to install using SWIG.')

        if not shutil.which('swig'):
            raise Exception(f'BMCSimulator: SWIG is not installed on your system. Please install SWIG.')

        print(f'BMCSimulator: compiling BMCSimulator package using SWIG...')
        check_build = subprocess.call([sys.executable, 'setup.py', 'build_ext', '--inplace'], cwd=sim_path)
        if check_build != 0:
            print(f'BMCSimulator: Could not build BMCSimulator package')
            return False

        print(f'BMCSimulator: installing BMCSimulator package...')
        check_install = subprocess.call([sys.executable, 'setup.py', 'install'], cwd=sim_path)

        if check_install != 0:
            print(f'BMCSimulator: Could not install BMCSimulator package build using SWIG.')
            return False
        else:
            return True


if __name__ == '__main__':
    options = ''
    if len(sys.argv) > 1:
        options = sys.argv[1:]
    root_path = os.path.dirname(os.path.abspath(__file__))
    cest_mrf_path = os.path.join(root_path, 'cest_mrf')
    sim_lib_path = os.path.join(root_path, 'cest_mrf', 'sim_lib')

    cest_mrf_installed = cest_mrf_install(cest_mrf_src=cest_mrf_path, options = options)
    sim_lib_installed = sim_setup(sim_path = sim_lib_path)

    if cest_mrf_installed:
        print(f'\ncest_mrf installation: SUCCESSFUL \n')
    else:
        print(f'\ncest_mrf installation: FAILED \n')

    if sim_lib_installed:
        print(f'\nBMCSimulator installation: SUCCESSFUL \n')
    else:
        print(f'\nBMCSimulator installation: FAILED \n')

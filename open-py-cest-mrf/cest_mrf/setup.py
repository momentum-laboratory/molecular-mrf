# import os

# from setuptools import setup
# import shutil
# import subprocess
# import sys
# from setuptools import setup, find_packages


# def install_last_pypulseq():
#     print('START installing of pypulseq')
#     # if shutil.which('git') is None:
#     #     raise Exception(f'cest_mrf: Git is not installed on your system. Please install Git.')
#     # cloned = subprocess.call(['git', 'clone','--branch', 'master' , 'https://github.com/imr-framework/pypulseq'])
#     # installed = subprocess.call([sys.executable, '-m', 'pip',  'install', '-e','.'], cwd=os.path.join(os.path.dirname(__file__),'pypulseq'))
#     # reseted = subprocess.call(['git', 'reset','--hard', 'cc9ccfb'], cwd=os.path.join(os.path.dirname(__file__),'pypulseq') )
#     # os.remove(os.path.join(os.path.dirname(__file__), os.path.join('pypulseq','pypulseq','Sequence','write_seq.py')))
#     # shutil.copy(os.path.join(os.path.dirname(__file__),'write_seq.py'), os.path.join(os.path.dirname(__file__),os.path.join('pypulseq','pypulseq','Sequence','write_seq.py')))
#     # #os.remove('write_seq.py')
#     print(os.getcwd())
#     if shutil.which('git') is None:
#         raise Exception(f'cest_mrf: Git is not installed on your system. Please install Git.')
#     cloned = subprocess.call(['git', 'clone','--branch', 'master' , 'https://github.com/imr-framework/pypulseq'])
#     installed = subprocess.call([sys.executable, '-m', 'pip',  'install', '-e','.'], cwd=os.path.join(os.getcwd(),'pypulseq'))
#     reseted = subprocess.call(['git', 'reset','--hard', 'cc9ccfb'], cwd=os.path.join(os.getcwd(),'pypulseq') )
#     os.remove(os.path.join(os.getcwd(), os.path.join('pypulseq','pypulseq','Sequence','write_seq.py')))
#     shutil.copy(os.path.join(os.getcwd(),'write_seq.py'), os.path.join(os.getcwd(),os.path.join('pypulseq','pypulseq','Sequence','write_seq.py')))
#     #os.remove('write_seq.py')

#     print('pypulseq successfully installed')


# install_last_pypulseq()
# setup(
#     name='cest_mrf',
#     author='Nikita Vladimirov',
#     author_email='nikitav@mail.tau.ac.il',
#     version='0.2',
#     description='Python code to use C++ pulseq-CEST to simulate MRI signal and MRF dictionary generation.',
#     install_requires=['bmctool==0.5.0', 'numpy==1.19.5', 'scipy==1.10.0', 'PyYAML==6.0', 'sigpy==0.1.22'],
#     keywords='MRI, Bloch, CEST, simulations',
#     packages=find_packages(),
#     # packages=['cest_mrf'],
#     # package_dir={'cest_mrf': '.'},
#     package_dir={'': '.'},
#     include_package_data=True,
#     python_requires='>=3.9'
# )
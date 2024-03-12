# Simulator library rebuild guide

### Prerequisites
   - [SWIG](http://www.swig.org/exec.html)
   - a working C++ compiler
  
### Rebuilding
First you should delete existing installation: `pip uninstall BMCSimulator` after that delete all build-related files:
```
build/* 
BMCSimulator.py
BMCSimulator_wrap.cpp
```

Rebuild and reinstall the *BMCSimulator* package by running the following commands in the terminal
```
    python setup.py build_ext --inplace
    python setup.py install
```

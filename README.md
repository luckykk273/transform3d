# transform3d
C library for 3D transformations.

# Preface
All conversion functions and testing cases in this repository are the C version of [pytransform3d](https://github.com/dfki-ric/pytransform3d).

# Prerequisites
I have tested the library in Windows 11, but it should be easy to compile in other platforms.

## C Compiler
I use MSYS2 to install gcc. Feel free to use your own compiler.  
**NOTE: remember to update the path of your own compiler in CMakeLists.txt:**
```cmake
set(CMAKE_C_COMPILER "path/to/your/gcc")
```

## Cmake
I write a simple CMakeLists.txt to compile and build all the static/dynamic/executable files. I also use MSYS2 to install make and one can [make a link](https://stackoverflow.com/questions/51755089/where-is-make-on-msys2-mingw-w64) between `make` and `mingw32-make.exe` for convenience.

## Scripts
I write a simple `build.bat` to run compiling and building. Feel free to write your own scripts.

## Python
If one wants to run all tests in this repositories, you should also install Python. I use [poetry](https://github.com/python-poetry/poetry) to manage the dependencies. Please follow the scripts to install necessary dependencies with `poetry`:
```cmd
path/of/transform3d>poetry init
path/of/transform3d>poetry env use python
path/of/transform3d>poetry shell
path/of/transform3d>poetry install
```

# Tests
There are lots of testing cases in [pytransform3d]() to carefully test the correctness of conversions and the numerical precision. If one wants to test it, follow the [instructions](#python) first to ensure all dependencies are installed and then follow the scripts below:
```cmd
path/of/transform3d>cd tests
path/of/transform3d/tests>pytest
```
# TODO
1. Detailed documentation
    1. Badges
    2. Comments for functions
2. Support and test on both Windows and Linux platforms
    1. Rewrite CMakeLists.txt
    2. Add Scripts folder and scripts for different platforms
3. Write Python-C-API instead of adding API to library wrapper manually. 

# Contributions
1. Rewrite in C.
2. Compile and build the C library to dll and write a C library wrapper in Python to test functions more convenient.

# License
This library is redistributed from [pytransform3d](https://github.com/dfki-ric/pytransform3d) under [BSD 3-Clause License](https://github.com/luckykk273/transform3d/blob/main/LICENSE).

# Reference
[pytransform3d](https://github.com/dfki-ric/pytransform3d)

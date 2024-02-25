rmdir /s build
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug -DTRANS_BUILD_DYNAMIC=1 -DTRANS_BUILD_STATIC=1 -DTRANS_BUILD_EXE=1 ..
make
cd ..
rmdir /s build
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Debug ..
make
move main.exe ../main.exe
cd ..
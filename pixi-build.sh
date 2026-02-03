cmake -S mycpp -B mycpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build mycpp/build -j

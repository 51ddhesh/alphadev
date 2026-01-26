## AlphaDev

From-scratch implementation of Google Deepmind's AlphaDev paper with an assembly playground.


### Usage:
1. Create a virtual environment
```sh
uv venv
# or python3 -m venv .venv
```

2. Activate the environment
```sh
source .venv/bin/activate
```

3. Install `pybind11`
```sh
uv pip install -r requirements.txt
# or pip intall -r requirements.txt
```

4. Create build directory
```sh
mkdir -p build
rm -rf build/*
```

5. Create the build files
```sh
cmake -S . -B build -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
```

6. Create the `.so` file
```sh
cmake --build build
```

7. Run the test
```sh
python3 tests/test_env.py
```


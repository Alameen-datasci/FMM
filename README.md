# FMM

A work-in-progress CUDA-accelerated Fast Multipole Method (FMM) library written in C++ and CUDA. The current implementation builds a particle octree and computes multipole expansions; **The full FMM solver is not yet complete.**

## Requirements

- CMake 3.24 or newer
- A CUDA-capable NVIDIA GPU and CUDA Toolkit
- Ninja (used by the provided CMake presets)

## Build

From the project directory:

```sh
cmake --preset default
cmake --build --preset default
```

The `debug` preset creates a debug build. To build the GoogleTest-based tests, install GoogleTest and run:

```sh
cmake --preset testing
cmake --build --preset testing
ctest --preset testing
```

## Current API

Include `FMM.hpp`, provide particle masses and x/y/z positions as arrays, then call `build`:

```cpp
#include "FMM.hpp"

FMM fmm(count, masses, x, y, z);
fmm.build(expansion_order, particles_per_leaf);
```

`expansion_order` currently supports values from 0 to 10. `particles_per_leaf` must be positive. The current API exposes particle and node counts; acceleration output is not implemented yet.

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

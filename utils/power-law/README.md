# Project Overview

This project requires **Boost** to be correctly installed on your system and is built using **CMake**. Follow the instructions below to set up, compile, and run the code.


## Getting Started

To set up and build the project, follow these steps:

1. **Create a build directory and navigate to it:**
   ```bash
   mkdir build
   cd build
   ```

2. **Choose a build mode and run the following commands:**

    - **For Debug mode:**
        ```bash
        cmake ..
        make
        ```

    - **For Release mode:**
        ```bash
        cmake -D CMAKE_BUILD_TYPE=Release ..
        make
        ```

## Running the Program

To execute the program, use the following command:

```bash
./bin/pl --ub <upper_bound> --size <sample_size> --beta <beta>
```

### Parameters

- `<upper_bound>`: Specifies the upper bound of your support (the lower bound is always 1).

- `<sample_size>`: The number of samples to generate.

- `beta`: Beta of the Power Law distribution. Default: 1.5.

The program will print the generated samples to the console.

## Example Usage in Python

An example of how to use the program within Python is provided in the `example.ipynb` notebook.

> **Note:** The C++ code must be compiled in Release mode for the Python example to work correctly.

## Notes

This code is inspired by similar projects and examples, particularly:

- 
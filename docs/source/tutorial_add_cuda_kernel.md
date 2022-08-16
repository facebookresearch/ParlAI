# Add Custom CUDA kernel to ParlAI

This tutorial gives step-by-step instructions on how to add a mixed c++/CUDA extension to pytorch and use it in ParlAI code. Check out the reference code files inside this folder while reading through the tutorial.

## Setup:
This tutorial assumes that you have parlai env setup. On top of that, run `pip install ninja`. 

## Step 1: Define functions and bind to Python in C++ file
Create a C++ file and in it, (1) define the C++ interface (2) declare the CUDA function which we will define later in the .cu file. We then bind the C++ function to Python with the following format 
```
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("func_name_1", &func_1);
  m.def("func_name_2", &func_2);
}
```
Example file: `add_cuda.cpp`
```
#include <torch/extension.h>
#include <vector>

// CUDA forward declarations
std::vector<int> add_cuda(
    std::vector<int> &a,
    std::vector<int> &b
);

// C++ interface
std::vector<int> add(
    std::vector<int> &a,
    std::vector<int> &b
) {
    return add_cuda(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add);
}
```

## Step 2: Define CUDA kernels
Define the function that was declared at the cpp file, which includes a call to a CUDA kernel. We will put parts of the code that we want to speed up inside the kernel.

Example file: `add_cuda_kernel.cu`. Note that CUDA kernel by default does not work with C++ vectors, thus we need to convert them into pointers before passing to the kernel.

```
#include <torch/extension.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

__global__
void add_cuda_kernel(
    int N,
    int *a, 
    int *b
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < N; i += stride)
    {
        b[i] = a[i] + b[i];
    }
}


std::vector<int> add_cuda(
    std::vector<int> &a,
    std::vector<int> &b
) {
    int blockSize = 256;
    int N = a.size();
    int numBlocks = (N + blockSize - 1) / blockSize;
    #ifdef DEBUG
        printf("# of blocks: %d, block size: %d\n", numBlocks, blockSize);
    #endif

    // initialize pointer array on the gpu
    int* dev_a = &a[0];
    int* dev_b = &b[0];
    // allocate memory on the gpu 
    cudaMallocManaged(&dev_a, N * sizeof(int));
    cudaMallocManaged(&dev_b, N * sizeof(int));
    // copy data from cpu to gpu
    cudaMemcpy(dev_a, &a[0], N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, &b[0], N * sizeof(int), cudaMemcpyHostToDevice);

    add_cuda_kernel<<<numBlocks, blockSize>>>(N, dev_a, dev_b);
    // make CPU wait until the kernel is done
    cudaDeviceSynchronize();

    // initialize and allocate memory for array on cpu
    int *b_ptr;
    b_ptr = (int*)malloc( N * sizeof(int) );
    // copy result from gpu to cpu
    cudaMemcpy(b_ptr, dev_b, N * sizeof(int),cudaMemcpyDeviceToHost); 
    // convert back to vector
    memcpy(&b[0], &b_ptr[0], N*sizeof(int)); 

    //free memory
    free(b_ptr);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return b;
}
```

## Step 3: Integrate with Pytorch 
### Method 1: "Ahead of time" 
We can use `setup.py` script that uses setuptools to compile our C++ code. It looks like this:
```
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='...',
    ext_modules=[
        CUDAExtension('name_cuda', [
            'name.cpp',
            'name.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```
Run `python setup.py install` on your terminal once. Then inside the python script, we can do: 
```
import torch
import name_of_your_extension
```

Example file: `setup.py`
```
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='add',
    ext_modules=[
        CUDAExtension(
            'add_cuda',
            [
                'add_cuda.cpp',
                'add_cuda_kernel.cu',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)
```

### Method 2: "Just in time"
Add the following inside your python script
```
from torch.utils.cpp_extension import load
extension_name = load(name='...', sources=['name.cpp', 'name.cu'])
```

### Method 3: Use it!

```
#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torch.utils.cpp_extension import load
import unittest

add_cuda = load(name='add_cuda', sources=['add_cuda.cpp', 'add_cuda_kernel.cu'])

class TestUtils(unittest.TestCase):
    def test_add_array(self):
        a = [1, 2, 3, 4, 5]
        b = [2, 3, 4, 5, 6]
        c = add_cuda.add(a, b)
        assert c == [3, 5, 7, 9, 11], 'arrays did not add correctly'
```

## Resources
1. [Pytorch toutorial on C++/CUDA extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension)
2. [Nvidia tutorial on CUDA programming](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
3. [CUDA code for Vector Add](http://selkie.macalester.edu/csinparallel/modules/ConceptDataDecomposition/build/html/Decomposition/CUDA_VecAdd.html)

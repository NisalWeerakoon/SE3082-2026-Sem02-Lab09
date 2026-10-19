# SE3082 Parallel Computing - Lab 09

This repository contains the six CUDA exercises from the 2026 Lab 9 sheet and a [Colab notebook](SE3082_Lab09_IT24103467.ipynb) that writes, compiles and runs each `.cu` program in order.

Open the notebook in Google Colab, select a GPU runtime, and run all cells. The compile cells use `-arch=sm_75` for the T4 GPU named in the lab sheet; change that flag if Colab assigns a GPU with another compute capability. The notebook can be rebuilt from these source files with `python build_notebook.py`.

| Exercise | Program | Verification |
| --- | --- | --- |
| 1 | `hello.cu` | Four GPU greetings |
| 2 | `ex2_add.cu` | Result is 9 |
| 3 | `ex3_blocks.cu` | 512 blocks, one thread each, zero mismatches |
| 4 | `ex4_threads.cu` | One block, 512 threads, zero mismatches |
| 5 | `ex5_vecmul.cu` | 10,000,000 products; prints last 1,000; zero mismatches |
| 6 | `ex6_matrix.cu` | 10,000 × 10,000 element-wise products; prints last 1,000; zero mismatches |

The programs check CUDA API errors, kernel launches, and CPU/GPU result agreement. The bounds checks in Exercises 5 and 6 stop extra threads from accessing elements beyond the arrays. Exercise 6 needs approximately 1.2 GB of GPU memory and 1.2 GB of host memory for the three arrays.

The course PDF is not redistributed here; obtain it from the SE3082 CourseWeb module. Runtime outputs are intentionally left empty in the generated notebook until it has been run on a real GPU.

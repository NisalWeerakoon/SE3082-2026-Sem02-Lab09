"""Build the Colab notebook from the version-controlled CUDA sources."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FILES = [
    ("Exercise 1 - GPU setup", "hello.cu"),
    ("Exercise 2 - Add two numbers", "ex2_add.cu"),
    ("Exercise 3 - 512 blocks", "ex3_blocks.cu"),
    ("Exercise 4 - 512 threads", "ex4_threads.cu"),
    ("Exercise 5 - Ten-million-element vector product", "ex5_vecmul.cu"),
    ("Exercise 6 - 10,000 by 10,000 element-wise matrix product", "ex6_matrix.cu"),
]


def markdown(value):
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(True)}


def code(value):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": value.splitlines(True)}


cells = [
    markdown("# SE3082 Parallel Computing - Lab 09\nStudent: IT24103467\n\nRun each cell in order on a Colab GPU runtime. The source files are also in the GitHub repository.\n"),
    code("!nvidia-smi\n!nvcc --version\n!nvidia-smi --query-gpu=name,compute_cap --format=csv\n"),
    markdown("The commands below target a T4 (`sm_75`). If Colab provides another GPU, change `-arch=sm_75` to match its compute capability.\n"),
    code("%%writefile cuda_utils.cuh\n" + (ROOT / "cuda_utils.cuh").read_text(encoding="utf-8")),
]

for title, filename in FILES:
    stem = filename.removesuffix(".cu")
    cells.append(markdown(f"## {title}\n"))
    cells.append(code("%%writefile " + filename + "\n" + (ROOT / filename).read_text(encoding="utf-8")))
    cells.append(code(f"!nvcc -arch=sm_75 {filename} -o {stem} && ./{stem}\n"))

notebook = {
    "cells": cells,
    "metadata": {"colab": {"provenance": []},
                 "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
(ROOT / "SE3082_Lab09_IT24103467.ipynb").write_text(
    json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
)

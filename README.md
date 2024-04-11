# Impact histories code

This repository reproduces Figures 2 through 4 in Wogan et al. (2024), which is a paper on Earth's impact history. There are two steps to run the code, as outlined below.

## Installation and setup

If you do not have Anaconda on your system, install it here or in any way you perfer: https://www.anaconda.com/download . Next, run the following code to cerate a conda environment `impactstats`.

```bash
conda create -n impactstats python=3.11 numpy numba scipy matplotlib
```

## Run the code

To do all calculations, and reproduce Figure 2 to 4 in the paper, run all the code with

```bash
conda activate impactstats
python main.py
```

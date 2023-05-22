# Data Assimilation Summer School 2023

Material for course on **Deep Learning in Scientific Inverse Problems**, to be taught
at [Summer school on Data Assimilation](https://data-assimilation.com).

## Teaching Material
The main teaching material is available in the form of Jupiter slides. Simply type ``jupyter nbconvert Lecture.ipynb --to slides --post serve`` to access the slides.

## Notebooks
Several tutorials are presented during the course in the form of Jupyter notebooks.

| Session   | Exercise (Github) | Exercise (Colab) |
|-----------|------------------|------------------|
| EX0: Prepare brain dataset | [Link](notebooks/Prepare_brain_dataset.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/DA_summerschool_2023/blob/main/notebooks/Prepare_brain_dataset.ipynb)  |
| EX1: Prepare brain-fbp dataset | [Link](notebooks/Prepare_brainfbp_dataset.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/DA_summerschool_2023/blob/main/notebooks/Prepare_brainfbp_dataset.ipynb)  |
| EX2: Variational CTscan imaging | [Link](notebooks/Variational_ctscanimaging.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/DA_summerschool_2023/blob/main/notebooks/Variational_ctscanimaging.ipynb)  |
| EX2: DIP CTscan imaging | [Link](notebooks/DIP_ctscanimaging.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/DA_summerschool_2023/blob/main/notebooks/DIP_ctscanimaging.ipynb)  |



## Getting started
To run the different jupyter notebooks, participants can either use:

- local Python installation (simply run ``conda env create -f environment. yml``)
- a Cloud-hosted environment such as [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mrava87/DA_summerschool_2023/main)
  or  Google Colab (use links provided above to open the notebook directly in Colab).
# Data Assimilation Summer School 2023

Material for course on **Deep Learning in Scientific Inverse Problems**, to be taught
at [Summer school on Data Assimilation](https://data-assimilation.com).

## Teaching Material
The main teaching material is available in the form of Jupiter slides. Simply type ``jupyter nbconvert Lecture.ipynb --to slides --post serve`` to access the slides.

## Notebooks
Several tutorials are presented during the course in the form of Jupyter notebooks.

| Session   | Exercise (Github) | Exercise (Colab) |
|-----------|------------------|------------------|
| EX1: Visual optimization | [Link](Visual_optimization.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/pylops_notebooks/blob/master/official/timisoara_summerschool_2019/Visual_optimization.ipynb)  |
| EX2: Linear Operators | [Link](Linear_Operators.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/pylops_notebooks/blob/master/official/timisoara_summerschool_2019/Linear_Operators.ipynb)  |
| EX3: Solvers | [Link](Solvers.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/pylops_notebooks/blob/master/official/timisoara_summerschool_2019/Solvers.ipynb)  |
| EX4: Seismic Redatuming | [Link](SeismicRedatuming.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/pylops_notebooks/blob/master/official/timisoara_summerschool_2019/SeismicRedatuming.ipynb)  |
| EX5: Seismic Inversion | [Link](../../developement/SeismicInversion-Volve.ipynb) | - |
| EX6: Seismic Inversion with GPUs | [Link](../../developement-cuda/SeismicInversion.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/pylops_notebooks/blob/master/developement-cuda/SeismicInversion.ipynb)  |
| EX7: Seismic Redatuming with Dask | [Link](../../developement-distributed/WaveEquationProcessing-distributed.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/pylops_notebooks/blob/master/developement-distributed/WaveEquationProcessing-distributed.ipynb) |


## Getting started
To run the different jupyter notebooks, participants can either use:

- local Python installation (simply run ``conda env create -f environment. yml``)
- a Cloud-hosted environment such as [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pylops/pylops_transform2022/main)
  or  Google Colab (use links provided above to open the notebook directly in Colab).
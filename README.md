# Data Assimilation Summer School 2023

Material for course on **Deep Learning in Scientific Inverse Problems**, to be taught
at [Summer school on Data Assimilation](https://data-assimilation.com).

## Teaching Material
The main teaching material is available in the form of Jupiter slides. Simply type ``jupyter nbconvert Lecture.ipynb --to slides --post serve`` to access the slides.

## Notebooks
Several tutorials are presented during the course in the form of Jupyter notebooks.

| Session   | Exercise (Github) | Exercise (Colab) |
|-----------|------------------|------------------|
| EX0: Prepare brain dataset | [Link](notebooks/Prepare_brain_dataset.ipynb) |  |
| EX1: Prepare brain-fbp dataset | [Link](notebooks/Prepare_brainfbp_dataset.ipynb) |  |
| EX2: Variational CTscan imaging | [Link](notebooks/Variational_ctscanimaging.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/DA_summerschool_2023/blob/main/notebooks/Variational_ctscanimaging.ipynb)  |
| EX3: Supervised Learning for CTscan imaging | [Link](notebooks/Supervised_ctscanimaging.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/DA_summerschool_2023/blob/main/notebooks/Supervised_ctscanimaging.ipynb)  |
| EX4: DIP CTscan imaging | [Link](notebooks/DIP_ctscanimaging.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/DA_summerschool_2023/blob/main/notebooks/DIP_ctscanimaging.ipynb)  |
| EX5: PnP for CTscan imaging | [Link](notebooks/PnP_ctscanimaging.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/DA_summerschool_2023/blob/main/notebooks/PnP_ctscanimaging.ipynb)  |
| EX6: Learned iterative solver for CTscan imaging | [Link](notebooks/LearnedIt_ctscanimaging.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/mrava87/DA_summerschool_2023/blob/main/notebooks/LearnedIt_ctscanimaging.ipynb)  |


## Getting started

### Data

If you are attending the course, we will provide you with a GDrive link with a minimal dataset used in the examples (this is produced by the `Prepare_brain_dataset.ipynb` and `Prepare_brainfbp_dataset.ipynb` notebooks.

If you would like to run the entire pipeline (including the `Prepare_brain_dataset.ipynb` and `Prepare_brainfbp_dataset.ipynb`), you will need access to the original [FASTMRI](https://fastmri.med.nyu.edu) dataset. We will be working with a small subset of it that you can retrieve following these two simple steps:
- Register at the bottom of the website to obtain a list of links to be used to retrieve the dataset of interest. You will receive an email with instructions on how to retrieve the dataset;
- Run the curl command for the `brain_multicoil_val_batch_0.tar.xz` dataset (be prepared to wait long time, and ensure you have 94GB of space in your disk).

### Codes
To run the different Jupyter notebooks, participants can either use:

- local Python installation (simply run ``./install_env.sh``). Note, this requires access to a GPU. For CPU-only workstation, modify the ``environment.yml`` file accordingly.
- a Cloud-hosted environment such as Google Colab (use links provided above to open the notebook directly in Colab). Before getting started, make sure to manually upload all .py files from the `notebooks` directory and the entire `model` directory into your Colab local storage. Moreover, place the folder with the data that you have previously downloaded in your personal GDrive.


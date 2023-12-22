<h1>Info</h1>

This is the official repository of **RS_Py**, which is part of the **Readersourcing 2.0** ecosystem. This repository is a [Git Submodule](https://git-scm.com/book/it/v2/Git-Tools-Submodules) of the main project, which can be found by taking advantage of the links below.

<h1>Useful Links</h1>

- <a href="https://readersourcing.com">Readersourcing 2.0 (Web Interface)</a>
- <a href="https://github.com/Miccighel/Readersourcing-2.0">Readersourcing 2.0 (GitHub)</a>
- <a href="https://zenodo.org/record/1446468">Original Article</a>
- <a href="https://zenodo.org/record/1452397">Technical Documentation (Zenodo)</a>
- <a href="https://github.com/Miccighel/Readersourcing-2.0-TechnicalDocumentation"> Technical Documentation (GitHub)</a>
- <a href="https://zenodo.org/record/3245209">Zenodo Record</a>

<h1>Description</h1>

**RS_Py** is an additional component of the Readersourcing 2.0 ecosystem, providing a fully working implementation 
of the RSM and TRM models presented in the original paper. These models are encapsulated by the server-side 
application of Readersourcing 2.0, that is, RS_Server.

Developers with a background in the Python programming language can leverage RS_Py to generate and test new 
simulations of ratings given by readers to a set of publications. They are allowed to alter the internal logic 
of the models to test new approaches without the need to fork and edit the full implementation of Readersourcing 2.0.

<h1>Installation</h1>

To use RS_Py notebooks, it is sufficient to clone the repository and place it somewhere on the filesystem. 
Ensure that the required Python packages are installed by leveraging a distribution such as [Anaconda](https://www.anaconda.com/distribution/). 
If a lightweight installation is preferred, an instance of Python 3.7.3 or higher is needed to install 
the required packages, such as Jupyter, Pandas, and others.

<h1>Usage</h1>

**RS_Py** is organized into five main folders on the filesystem:

- The `data` folder is used to store the dataset exploited to test the models presented in the original paper.
- The `models` folder is used to store the output of these models.
- The `notebooks` folder contains Jupyter notebooks used to generate new datasets and implement the models presented by Soprano et al. (2019).
- The `scripts` folder contains implementations of the Jupyter notebooks as pure Python scripts.
- The `src` folder contains a Python script which converts Jupyter notebooks into pure Python scripts.

Within the `notebooks` folder, three Jupyter notebooks are available:

- `Readersourcing.ipynb` provides an implementation of the RSM model presented by Soprano et al. (2019).
- `TrueReview.ipynb` offers an implementation of the TRM model.
- The `Seeder.ipynb` notebook allows the generation of new datasets, which will be stored inside the `data` folder.

Inside the `src` folder, the `Convert.py` script enables the conversion of notebooks into Python scripts, 
and these are then stored inside the `scripts` folder.

The behavior of `Seeder.ipynb` and `Readersourcing.ipynb` notebooks can be customized by modifying the parameter 
settings found in the initial rows of both notebooks. Table 1, shown below, outlines the parameters available for the former, 
while Table 2 presents the parameters for the latter.

To run and use the Jupyter notebooks, navigate to the main directory of **RS_Py** using a command-line prompt 
(you should see folders such as `data`, `models`, `notebooks`, etc.) and type `jupyter notebook`. 
This command will start the Jupyter server, and you can access the *Notebook Dashboard* in your browser
at the web application's URL (typically, [http://localhost:8888](https://jupyter.readthedocs.io/en/latest/running.html#running)).

| **Parameter**         | **Description**                            | **Values**                  |
|-----------------------|--------------------------------------------|-----------------------------|
| `dataset_name`        | Name of the dataset to simulate             | String                      |
| `papers_number`       | Number of papers to simulate               | Positive integer            |
| `readers_number`      | Number of readers to simulate              | Positive integer            |
| `authors_number`      | Number of authors to simulate              | Positive integer            |
| `months_number`       | Number of months of activity to simulate   | Positive integer            |
| `paper_frequencies`   | Amount of papers rated by each reader group| Array of positive integers  |
| `readers_percent`     | Percentage of readers to assign to a single group | Positive integer    |

**Table 1:** Parameters available for the Seeder Jupyter notebook.

| **Parameter**                  | **Description**                                           | **Values**                  |
|---------------------------------|-----------------------------------------------------------|-----------------------------|
| `dataset_name`                 | Name of the dataset to simulate                            | String                      |
| `day_serialization`            | Activate serialization of data on a per-day basis         | True, False                  |
| `day_serialization_threshold`  | Serialize data every X days                               | Positive integer            |
| `days_number`                  | Amount of days simulated in the input dataset              | Positive integer            |

**Table 2:** Parameters available for the Readersourcing Jupyter notebook.


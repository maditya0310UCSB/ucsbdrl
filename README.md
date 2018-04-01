# UCSB Deep Reinforcement Learning Seminar

## Development Environment Setup

We support Anaconda (https://www.anaconda.com/download) using Python 3.6. If you want to isolate our installed packages from the rest of your Python system, make sure to install Anaconda for the local user only and do not add conda to the path (this is a check-box option during installation). A conda environment will be used to further isolate things and to make setup easier.

### Create Conda Environmen
After installing Anaconda (and checking out or downloading this repo), create environment with:<br>
`conda env create -f environment.yml`<br>
This will install the Python packages needed to run our examples.
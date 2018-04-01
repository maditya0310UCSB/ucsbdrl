# UCSB Deep Reinforcement Learning Seminar

## Development Environment Setup

We support Anaconda (https://www.anaconda.com/download) using Python 3.6. If you want to isolate our installed packages from the rest of your Python system, make sure to install Anaconda for the local user only and do not add conda to the path (this is a check-box option during installation). A conda environment will be used to further isolate things and to make setup easier.

### Create Conda Environment
After installing Anaconda (and checking out or downloading this repo), create environment with:<br>
`conda env create -f environment.yml`<br>
This will install the Python packages needed to run our examples. (It may take a bit.)

### Activate Conda Environment
You will need to `activate` our environment each time you open a new terminal. Use:<br>
`source activate ucsbdrl`<br>
This isolates all `pip` or `conda` installs from your other environments or from your system-level Python installation.

### Update Conda Environment
As the quarter progresses we may update the `environment.yml`. To update your local environment, first `pull` the repo, then:<br>
`conda env update -f=environment.yml`
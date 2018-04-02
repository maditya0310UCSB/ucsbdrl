# UCSB Deep Reinforcement Learning Seminar

## Development Environment Setup

We support Anaconda (https://www.anaconda.com/download) using Python 3.6. If you want to isolate installed packages from the rest of your Python system, make sure to install Anaconda for the local user only and do not add conda to the path (this is a check-box option during installation). A conda environment will be used to further isolate things and to make setup easier.

### Create Conda Environment
The following creates a conda environment called `drl`:<br>
`conda create --name drl`

### Activate Conda Environment
You will need to activate your environment each time you open a new terminal. Use:<br>
`source activate drl`<br>
This isolates all `pip` or `conda` installs from your other environments or from your system-level Python installation.

### Install packages
Ensure the `drl` environment is activated then:<br>
`pip install tensorflow`<br>
`pip install keras`<br>
`pip install gym`<br>
<br>
On Ubuntu<br>
`conda install pytorch-cpu torchvision -c pytorch`<br>
<br>
On MacOS<br>
`conda install pytorch torchvision -c pytorch`<br> 
# Foundation of Data Science 2020 homework 2
Homework 2: large-scale ML. Train `dask` and regular `sklearn` models 
on the New York flights dataset. 
## Installation
Easiest way is to install the whole directory in editable mode, 
since you are checking out of the repository anyway.

`pip install -r requirements.txt`

`pip install -e .`

An isolated python environment required. Either `conda` or `virtualenv` recommended.
Packages are installed and managed using `pip` in either case, 
because latest versions are not available in conda at the time of writing.

## Usage
To run the whole pipeline use the supplied shell script:

`bash run.sh`

Performance results will appear in the console and in separate `json` files 
as specified in the configuration files located at `configs/`

## Comments

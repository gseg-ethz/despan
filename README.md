# Deformation Spotter and Analysis Worflow for Elongated Point Clouds (DeSpAn)
Developed by GSEG, ETH Zurich [Group Page](https://gseg.igp.ethz.ch/), maintained by [Nicholas Meyer](mailto:meyernic@ethz.ch).

## Installation
DeSpAn was tested on `Windows 10 and 11` environments and `conda` environments using `python 3.11`. In addition, DeSpAn 
requires [CloudCompare](https://www.cloudcompare.org/) to be installed on the system.

```shell
# Clone the github repo
cd path\to\project_parent_folder
git clone https://github.com/gseg-ethz/despan.git .\DeSpAn
cd DeSpAn

conda create -n NAME_OF_CONDA_ENV python=3.11
conda activate NAME_OF_CONDA_ENV

pip install .
```

## Basic Usage
DeSpAn takes (tiled) laser scan data from two epochs of overlapping elongated regions, and performs M3C2 analysis on 
these regions. It can handle both the `.ply` and the `.laz/.las` file formats. In case of the `.laz/.las` file format, 
the user can specify if prior to performing M3C2 the data shall be filtered to only retain ground points (based on the 
`classification` label according to the [LAS 1.4 specifications](https://www.asprs.org/wp-content/uploads/2010/12/LAS_1_4_r13.pdf)).

DeSpAn has a three-tiered configuration structure (listed based on priority):
1. Command line arguments
2. Specific configuration file
3. Default configuration file

Settings of lower tiered configurations are overridden if they are contradicted in higher tiered configurations. The 
configuration files can be found under `path\to\project_parent_folder\DeSpAn\DeSpAn\conf`. Prior to the first usage, 
one should check if the `CC_exe` argument in the Default configuration file points to the correct CloudCompare 
binary.

### Calling DeSpAn
A basic call to DeSpAn can be performed by:
```shell
DeSpAn -e1 'path_to_epoch1_file_or_folder' -e2 'path_to_epoch2_file_or_folder' -r 'path_to_results'
```

In case a specific configuration is being used, it can be called with:
```shell
DeSpAn -cf 'path_to_specific_configuration_file'
```
### More settings
The full command line call can be displayed with `DeSpAn --help`.
```shell
usage: DeSpAn.exe [-h] [-cf CONFIG_FILE] [-e1 EPOCH1] [-e2 EPOCH2] [-r RESULTS_DIR] [-gd {0,1}] [-fg {0,1}]

options:
  -h, --help            show this help message and exit
  -cf CONFIG_FILE, --config_file CONFIG_FILE
                        Path of a configuration file to use
  -e1 EPOCH1, --epoch1 EPOCH1
                        Path of the first epoch's point cloud file (or folder)
  -e2 EPOCH2, --epoch2 EPOCH2
                        Path of the second epoch's point cloud file (or folder)
  -r RESULTS_DIR, --results_dir RESULTS_DIR
                        Path to results
  -gd {0,1}, --greedy_dir_search {0,1}
                        Should subdirectories be included in search (0: false, 1: true)
  -fg {0,1}, --filter_ground_points {0,1}
                        Should point cloud be filtered to only contain ground points (0: false, 1: true)
```

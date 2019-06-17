# Hyperparameter tuning for adaptive CMA-ES

This repository contains the code for my master thesis regarding hyperparameter tuning for adaptive CMA-ES. 
The main code is based of my previous work, for which the data is available [here](https://github.com/Dvermetten/Online_CMA-ES_Selection).
The code for the adaptive CMA-ES is built upon the [modEA](https://github.com/sjvrijn/ModEA) package by Sander van Rijn, while the hyperparameter optimzer used is [MIP-EGO]() by Hao Wang.
This project uses the [BBOB](https://github.com/numbbo/coco)-functions as a testbed for hyperparameter optimization. 

# Usage
To use this code, python3 is required. You need to manually install the [cocoex](https://github.com/numbbo/coco) package. All other required packages can be installed trough pip.
To run a small experiment of hyperparameter optimization on static configuration, you can use the following command:
```Shell
> python main.py fid dim conf_nr iids reps c1 cc cmu budget
```
So for example:
```Shell
> python main.py 12 5 '1,2,3,4,5' '0,1,2,3,4' 0.1 0.4 0.1 250000
```


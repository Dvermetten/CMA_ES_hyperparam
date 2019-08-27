#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
local.py: This file is to be used to store variables such as paths that are specific to the local machine.

NOTE: This is an example file. Please create a copy named 'local.py'
'''


### Pool thread count ###
Pool_max_threads = 20


### MPI Core-Counts ###
use_MPI = False
MPI_num_host_threads = 16  # Number of available threads per host
MPI_num_hosts = 12         # Number of available hosts

### Data Generation ###
datapath = "/home/diederick/Documents/PythonProjects/Adaptive_CMA-ES/Data/"  # Where to store results
datapath_npy = "/home/diederick/Documents/PythonProjects/Adaptive_CMA-ES/Data/"

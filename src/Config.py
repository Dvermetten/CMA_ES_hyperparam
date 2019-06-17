#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Config file, containing all hardcoded values such as parallelization settings
"""

from multiprocessing import cpu_count
from src.local import MPI_num_host_threads, MPI_num_hosts, Pool_max_threads

num_threads = 1  # Default case, always true
try:
    num_threads = min(cpu_count(), Pool_max_threads)
    if num_threads > 1:
        allow_parallel = True
    else:
        allow_parallel = False
except NotImplementedError:
    allow_parallel = False

### Parallelization Settings ###
use_MPI = False
MPI_num_total_threads = MPI_num_host_threads * MPI_num_hosts
evaluate_parallel = True

### Experiment Settings ###
budget_factor = 1e4  # budget = ndim * ES_budget_factor
num_runs = 25

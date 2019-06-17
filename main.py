#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
from Experiments import runSingleSplitExample, runStaticExample, run_hyperparameter_optimization_example, \
    optimize_hyperparameters_exmperiment2, verify_c2, get_defaults, budget_influence_experiment, run_bipop_test, \
    single_param_experiment, run_ranking_test, run_ranking_test_lambda, run_exp_larger, runStaticWithHyperparameter
from src.Algorithms import single_split_with_hyperparams_parallel
import warnings


def runDefault():
    # run_hyperparameter_optimization_example()
    # runSingleSplitExample()
    # runStaticExample()
    # optimize_hyperparameters_exmperiment2(12)
    # get_defaults(2)
    single_split_with_hyperparams_parallel(12,5,1251,1251,20,False,51,[1,2],2,None,25000,None,None,None,True)
    # verify_random_seed()
    # verify_c2()
    # budget_influence_experiment(12, 0)
    # run_bipop_test()

def run_experiment(nr):
    # run_exp_larger(nr)
    if nr < 4:
        run_ranking_test_lambda(21, nr)
    else:
        run_exp_larger(nr - 4)
#    run_ranking_test(12, [i+(nr*4) for i in range(4)])
#    if nr < 5:
#        budget_influence_experiment(2, nr)
#    else:
#        single_param_experiment(nr-5)

def main():
    np.set_printoptions(linewidth=1000, precision=3)
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # ignores overflow and division by 0 warnings from modea
    if len(sys.argv) == 2:
        print("running index: {0}".format(sys.argv[1]))
        run_experiment(int(sys.argv[1]))
    elif len(sys.argv) == 10:
        fid = int(sys.argv[1])
        dim = int(sys.argv[2])
        conf_nr = int(sys.argv[3])
        iids = sys.argv[4].split(',')
        reps = sys.argv[5].split(',')
        c1 = double(sys.argv[6])
        cc = double(sys.argv[7])
        cmu = double(sys.argv[8])
        budget = int(sys.argv[9])
        return runStaticWithHyperparameter(fid, dim, conf_nr, iids, reps, c1, cc, cmu, budget)
    else:
        runDefault()


if __name__ == '__main__':
    main()

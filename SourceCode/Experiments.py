import cocoex
import numpy as np
from modea import Utils

from GaussianProcess import GaussianProcess
from GaussianProcess.trend import constant_trend
from BayesOpt import BO
from BayesOpt.SearchSpace import ContinuousSpace

from src.Adaptive_CMAES import StaticCMA, SingleSplitCMAES
from src.fitnessFunction import fitnessFunc
from src.local import datapath_npy, datapath
from src.Utils import create_observer, get_target, runParallelFunction, get_default_hyperparameter_values
from src.hyperparameterOptimizer import hyperparameterOptimizer, OneSearchSpaceOptimizer
from src.Algorithms import single_split_with_hyperparams_parallel, single_split_hyperparam_single
'''
Example functions for running small experiments
'''


def runStaticExample():
    """
        Example run of a static CMA-ES.
        Runs on F1 and 2; instance 1; 5 repetitions each
        Stores results in exdata/Experiments/Static/
        :return:
    """
    dim = 5
    reps = [0, 3]
    budget = 50000
    suite = cocoex.Suite("bbob", "", "dimensions:5 instance_indices:1,2,3,4,5")
    for fid in range(2):
        rep = Utils.intToRepr(reps[fid])
        obs = create_observer(reps[fid])
        for i in range(5):
            fitness_function = suite.get_problem_by_function_dimension_instance(fid + 1, dim, 1)
            fitness_function.observe_with(obs)
            f = fitnessFunc(fitness_function)
            cma = StaticCMA(
                dim, f, budget,
                representation=rep, seed=i
            )
            cma.run_optimizer()
            fitness_function.free()
            print(cma.used_budget)


def runSingleSplitExample():
    """
        Example run of a single-split CMA-ES.
        Runs on F1, 2 and 3 instance 1; 5 repetitions each
        Stores results in exdata/Experiments/SingleSplit/
        :return:
    """
    dim = 5
    rep1s = [2703, 399, 3567]
    rep2s = [2163, 2163, 2163]
    budget = 50000
    suite = cocoex.Suite("bbob", "", "dimensions:5 instance_indices:1,2,3,4,5")
    for fid_idx in range(3):
        fid = 5
        split_idx = 11
        split_target = get_target(fid, 1, split_idx)
        rep1 = rep1s[fid_idx]
        rep2 = rep2s[fid_idx]
        obs = create_observer(rep1, rep2, split_idx)
        for i in range(5):
            fitness_function = suite.get_problem_by_function_dimension_instance(fid, dim, 1)
            fitness_function.observe_with(obs)
            f = fitnessFunc(fitness_function)
            cma = SingleSplitCMAES(
                dim, f, budget, representation=Utils.intToRepr(rep1),
                representation2=Utils.intToRepr(rep2), seed=i,
                split=split_target
            )
            cma.run_optimizer()
            fitness_function.free()
            print(cma.best_individual)
            print(cma.used_budget)


def run_hyperparameter_optimization_example():
    """
        Example run of hyperparameter optimization on a single-split CMA-ES.
        Optimizes c_1, c_c and c_mu; on F2 instances 1 and 2, with rep1 = 0 and rep2 = 3, splitpoint_index 12

        :return: The result of the hyper-parameter optimzation (element [0] contains the final hyperparameters)
    """
    fid = 2
    dim = 5
    rep1 = 0
    rep2 = 3
    split_idx = 12
    iids = [1, 2]
    num_reps = 10
    budget = 3000
    n_step = 20
    n_init_sample = 10

    optimizer = hyperparameterOptimizer(max_iter=n_step, n_init_sample=n_init_sample)
    return optimizer(fid, dim, rep1, rep2, split_idx, iids, num_reps, budget=budget)


def optimize_hyperparameters_exmperiment(fid):
    statics = np.load(f"{datapath_npy}F{fid}_worst_C1.npy")
    results = []
    params = ['c_1', 'c_c', 'c_mu']
    optimizer = hyperparameterOptimizer(params=params, n_init_sample=10, max_iter=40)
    for (split_idx, rep1, rep2, _) in statics:
        print(split_idx, rep1)
        res1 = optimizer(fid, 5, rep1, rep2, split_idx, target_idx=split_idx, budget=5000)
        res2 = optimizer(fid, 5, rep1, rep2, split_idx, target_idx=51, budget=5000, sol_points=res1[0])
        results.append((split_idx, rep1, rep2, res1, res2))
    np.save(f"Data/statics_hyperparams_F{fid}_worst.npy", results)


def optimize_hyperparameters_exmperiment2(fid):
    statics = np.load(f"{datapath_npy}F{fid}_Best_C1.npy")
    results = []
    params_full = ['c_1', 'c_c', 'c_mu']
    p0 = ['c_sigma', 'chiN', 'c_1', 'c_c', 'c_mu', 'damps']
    p1 = ['c_1']
    p2 = ['c_c']
    p3 = ['c_mu']
    optimizer = hyperparameterOptimizer(params=params_full, n_init_sample=20, max_iter=80, part_to_optimize=-1)
    optimizer0 = hyperparameterOptimizer(params=p0, n_init_sample=20, max_iter=80, part_to_optimize=-1)
    optimizer1 = hyperparameterOptimizer(params=p1, n_init_sample=20, max_iter=80, part_to_optimize=-1)
    optimizer2 = hyperparameterOptimizer(params=p2, n_init_sample=20, max_iter=80, part_to_optimize=-1)
    optimizer3 = hyperparameterOptimizer(params=p3, n_init_sample=20, max_iter=80, part_to_optimize=-1)
    for (split_idx, rep1, rep2, exp) in statics:
        print(split_idx, rep1, rep2)
        res0 = optimizer0(fid, 5, rep1, rep2, split_idx, target_idx=51, budget=5000, num_reps=5)
        print("_______________________________________________________")
        res = optimizer(fid, 5, rep1, rep2, split_idx, target_idx=51, budget=5000, num_reps=5)
        print("_______________________________________________________")
        res1 = optimizer1(fid, 5, rep1, rep2, split_idx, target_idx=51, budget=5000, sol_points=[res[0][0]], num_reps=5)
        print("_______________________________________________________")
        res2 = optimizer2(fid, 5, rep1, rep2, split_idx, target_idx=51, budget=5000, sol_points=[res[0][1]], num_reps=5)
        print("_______________________________________________________")
        res3 = optimizer3(fid, 5, rep1, rep2, split_idx, target_idx=51, budget=5000, sol_points=[res[0][2]], num_reps=5)
        print("_______________________________________________________")
        print(res)
        print(res1)
        print(res2)
        print(res3)
        print(res0)
        print("_______________________________________________________")
        results.append(((split_idx, rep1, rep2, exp), res, res1, res2, res3, res0))
        np.save(f"Data/statics_hyperparams_F{fid}_param_inf_at_{int(rep1)}.npy", results)


def budget_influence_experiment(fid, nr):
    statics = np.load(f"{datapath_npy}F{fid}_Best_C1.npy")
    results = []
    params_full = ['c_1', 'c_c', 'c_mu']
    # optimizer = hyperparameterOptimizer(params=params_full, n_init_sample=20, max_iter=80, part_to_optimize=-1)
    optimizer0 = hyperparameterOptimizer(params=params_full, n_init_sample=60, max_iter=540, part_to_optimize=-1)
    (split_idx, rep1, rep2, exp) = statics[nr]
    print(split_idx, rep1, rep2, exp)
    res0 = optimizer0(fid, 5, rep1, rep2, split_idx, target_idx=51, budget=10000, num_reps=5,
                      data_file=f"{datapath}Data/F{fid}budget_600_nr{nr}")
    # res = optimizer(fid, 5, rep1, rep2, split_idx, target_idx=51, budget=10000, num_reps=5)
    results.append(((split_idx, rep1, rep2, exp),  res0))
    np.save(f"{datapath}statics_hyperparams_F{fid}_budget_inf_nr{nr}_try2.npy", results)


def single_param_experiment(nr, fid=2):
    params_full = ['c_sigma', 'chiN', 'c_1', 'c_c', 'c_mu', 'damps']
    optimizer = hyperparameterOptimizer(params=[params_full[nr]], n_init_sample=20, max_iter=80, part_to_optimize=-1)
    statics = np.load(f"{datapath_npy}F{fid}_Best_C1.npy")
    results = []
    (split_idx, rep1, rep2, exp) = statics[1]
    print(split_idx, rep1, rep2, exp)
    print([params_full[nr]])
    res = optimizer(fid, 5, rep1, rep2, split_idx, target_idx=51, budget=10000, num_reps=5, data_file=f"{datapath}Data/F{fid}_param_nr{nr}")
    results.append(((split_idx, rep1, rep2, exp), res))
    np.save(f"{datapath}statics_hyperparams_F{fid}_param_inf_nr{nr}_config2.npy", results)



def get_defaults(fid):
    statics = np.load(f"{datapath_npy}F{fid}_Best_C1.npy")
    results = []
    p0 = ['c_sigma', 'chiN', 'c_1', 'c_c', 'c_mu', 'damps']

    for (split_idx, rep1, rep2, exp) in statics:
        hyper = get_default_hyperparameter_values(p0,5,rep1,10000)
        # x = single_split_with_hyperparams_parallel(fid, 5,rep1,rep2,split_idx,False,51,budget=5000)
        results.append(hyper)
    np.save(f"{datapath}statics_default_F{fid}_params.npy", results)


def verify_c2():
    param_vals = [0.09776335967474224, 2.200454084772841, 0.0, 0.44637014242526507, 0.0, 1.5099599760481226]
    p0 = ['c_sigma', 'chiN', 'c_1', 'c_c', 'c_mu', 'damps']
    c1 = (p0, param_vals)
    df2 = get_default_hyperparameter_values(p0, 5, 963, 50000)
    df = get_default_hyperparameter_values(p0, 5, 975, 50000)

    def obj_func(x):
        if x is None:
            c2 = None
        else:
            c2 = (p0, x)
        return single_split_with_hyperparams_parallel(17, 5, 975, 963, 13, False, 33, [1], 1, c1, 50000, c2)

    obj_func(df2)
    obj_func(None)


def run_bipop_test():
    # s = cocoex.Suite("bbob","","")
    # f = s.get_problem_by_function_dimension_instance(21,5,1)
    # # p0 = ['c_c','lambda_']
    # c = SingleSplitCMAES(5,f,50000,representation=Utils.intToRepr(11),representation2=Utils.intToRepr(9), seed=10,split=42, lambda_=12, lambda2_=18)
    # c.run_optimizer()
    h = hyperparameterOptimizer(['c_c', 'c_1', 'c_mu', 'c_sigma'], 80, 20, None, None, 5000, None, -1)
    h(20, 5, 2, 2, 51, [1, 2, 3, 4], 4, 50000, 51, None, 0, True)


def run_ranking_test(fid = 6, confs = [0,1]):
    # results = []
    # configs = np.load(f"{datapath_npy}Statics_F{fid}_mid_new.npy")
    params_full = ['c_1', 'c_c', 'c_mu']
    optimizer0 = hyperparameterOptimizer(params=params_full, n_init_sample=20, max_iter=180, part_to_optimize=-1)
    for config in [2166]:
        print(config)
        open(f"{datapath}Data_Full/F{fid}_rep{config}.csv", 'w').close()
        res0 = optimizer0(fid, 5, config, config, 20, target_idx=51, budget=25000, num_reps=5,
                          data_file=f"{datapath}Data/F{fid}_st_{config}", opt_split=False)
        #results.append((config,  res0))
    #np.save(f"{datapath}F{fid}_static_{confs[0]}_to_{confs[-1]}.npy", results)


def run_ranking_test_lambda(fid = 21, confs_nr = 0):
    # results = []
    configs = np.load(f"{datapath_npy}Statics_F{fid}_lambda.npy")
    params_full = ['c_1', 'c_c', 'c_mu', 'lambda_']
    optimizer0 = hyperparameterOptimizer(params=params_full, n_init_sample=50, max_iter=650, part_to_optimize=-1)
    config = configs[confs_nr]
    x = Utils.intToRepr(config)
    x[-1] = 0
    config_new = Utils.reprToInt(x)
    print(config_new)
    open(f"{datapath}Data_Full/F{fid}_rep{config_new}_lambda.csv", 'w').close()
    optimizer0(fid, 5, config_new, config_new, 51, target_idx=51, budget=25000,
                      num_reps=10, data_file=f"{datapath}Data/F{fid}_st_{config_new}_lambda")
        # results.append((config_new,  res0))
    # np.save(f"{datapath}F{fid}_static_{confs[0]}_to_{confs[-1]}.npy", results)


def run_exp_larger(nr):
    # for i in range((4 * nr), 4 * (nr + 1)):
    # vals = np.load(f"{datapath_npy}F12_new.npy")[nr]
    params = {}
    params['c_1'] = 0.0669
    params['c_c'] = 1.0000
    params['c_mu'] = 0.0000
    opt_par = single_split_with_hyperparams_parallel(6, 5, 2166, 2166, 51, False, 51,
                                                     [1,2,3,4,5], 50, params, 25000, params, None, None, False)
    # single_split_with_hyperparams_parallel(12, 5, int(vals), int(vals), 51, False, 51,
    #                                                      [1, 2, 3, 4, 5], 50, None, 50000, None, None, None, True)

"""
    Interface function to the parallell execution of CMA-ES

    :param fid: The function id (from bbob)
    :param dim: The dimension to run the bbob-problem in
    :param conf_nr: The number of the configuration to run
    :param iids: The instances of the bbob-function to run
    :param reps: The amount of repetitions to run. Can also be a list of the repetition numbers (seeds).
    :param c1: Value for the c_1 hyperparameter
    :param cc: Value for the c_c hyperparameter
    :param cmu: Value for the c_mu hyperparameter
    :param budget: Maximum number of evaluations per run
    :param target_idx: Custom target-index to stop the optimization (distance 10^(2-target_idx/5))
    :return: The ERT over all 'reps' runs on all instances in iids
"""
def runStaticWithHyperparameter(fid, dim, conf_nr, iids, reps, c1, cc, cmu, budget, target):
    if target < 0:
        target = np.load(f"{datapath}targets.npy")[fid-1]
    return (single_split_hyperparam_single(fid = fid, dim = dim, rep1 = conf_nr, rep2 = conf_nr, target_idx = target, iid = iids,
                                       rep_nr = reps, hyperparams = {'c_1': c1, 'c_c': cc, 'c_mu': cmu},
                                       budget = budget, split_idx = 51))
    # else:
    #     single_split_with_hyperparams_parallel(fid, dim, int(conf_nr), int(conf_nr), target_idx = 51, iids = iids,
    #                                        num_reps = reps, hyperparams = {'c_1': c1, 'c_c': cc, 'c_mu': cmu},
    #                                        budget = budget, split_idx = 51)

def run_one_search_space(fid, target, budget, seed):
    if target < 1:
        target = np.load(f"{datapath_npy}targets.npy")[fid-1]
    opt = OneSearchSpaceOptimizer(n_init_sample=250, max_iter=750)
    opt(fid, target, budget, data_file=f"{datapath}Data_onesearch/F{fid}_{seed}", seed=seed)

def rerun_configs():
    # data = np.load(f"{datapath_npy}rerun_confs_def.npy")
    targets = np.load(f"{datapath_npy}targets.npy")
    # erts_def = []
    # params_full = ['c_1', 'c_c', 'c_mu']
    # for (fid, conf) in data[nr*2:(1+nr)*2]:
    #     fid = int(fid)
    #     conf = int(conf)
    #     print(fid, conf)
    #     target_idx = targets[fid - 1]
    #     optimizer0 = hyperparameterOptimizer(params=params_full, n_init_sample=20, max_iter=180, part_to_optimize=-1)
    #     optimizer0(fid, 5, conf, conf, 51, target_idx=target_idx, budget=25000,
    #                num_reps=5, data_file=f"{datapath}Data_mip/F{fid}_best_mip_{conf}")
        # ert = single_split_with_hyperparams_parallel(fid, 5, conf, conf, target_idx, False, target_idx,
        #                                              [1, 2, 3, 4, 5], 50, None, 25000, None, None, None, False)
        # print(ert)
        # erts_def.append(ert)
    # np.save(f"{datapath_npy}default_erts_per_fid", erts_def)
    erts_mip_ego = []
    data2 = np.load(f"{datapath_npy}rerun_F12_C4374.npy")
    for (fid, conf, c1, cc, cmu) in data2:
        fid = int(fid)
        conf = int(conf)
        print(fid,conf)
        target_idx = targets[fid-1]
        params = {}
        params['c_1'] = c1
        params['c_c'] = cc
        params['c_mu'] = cmu
        ert = single_split_with_hyperparams_parallel(fid, 5, conf, conf, target_idx, False, target_idx,
                                               [1, 2, 3, 4, 5], 50, params, 25000, params, None, None, False)
        print(ert)
        erts_mip_ego.append(ert)
    np.save(f"{datapath_npy}C4374_reruns", erts_mip_ego)


def run_MIP_EGO_experiment(nr):
    print(nr)
    conf_pairs = np.load(f"{datapath_npy}to_opt_confs_per_fid.npy")
    targets = np.load(f"{datapath_npy}targets.npy")
    print("Loaded everything")
    print(len(conf_pairs))
    for (fid, config) in conf_pairs[48*nr:48*(nr+1)]:
        print(fid, config)
        params_full = ['c_1', 'c_c', 'c_mu']
        optimizer0 = hyperparameterOptimizer(params=params_full, n_init_sample=20, max_iter=180, part_to_optimize=-1)
        optimizer0(fid, 5, config, config, 51, target_idx=targets[fid-1], budget=25000, num_reps=5,
                          data_file=f"{datapath}Data_MIP_full/F{fid}_{config}", opt_split=False)

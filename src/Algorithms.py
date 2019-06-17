import cocoex
from functools import partial
from itertools import product
from modea import Utils
from src.Utils import create_observer, get_target, runParallelFunction
import src.Config as Config
from src.Adaptive_CMAES import SingleSplitCMAES
from src.fitnessFunction import fitnessFunc
import numpy as np
from cma import CMAEvolutionStrategy
from src.local import datapath
import csv

def runSingleSplit(fid, dim, rep1, rep2, split_idx, iids=[1, 2, 3, 4, 5], num_reps=5, budget=None):
    """
        Function running single-split CMA-ES.

        :param fid: The function id (from bbob)
        :param dim: The dimension to run the bbob-problem in
        :param rep1: The configuration to run before the splitpoint
        :param rep2: The configuration to run after the splitpoint
        :param split_idx: The splitpoint-index at which to switch between rep1 and rep2
        :param iids: The instances of the bbob-function to run
        :param num_reps: The amount of repetitions to run
        :return: The ERT over all num_reps runs on all instances in iids
    """
    obs = create_observer(rep1, rep2, split_idx)
    suite = cocoex.Suite("bbob", "", f"dimensions:{dim}")
    if budget is None:
        budget = dim * Config.budget_factor
    HittingTimes = []
    succeeded = 0
    for i, iid in enumerate(iids):
        split_target = get_target(fid, iid) + 10 ** (2 - (split_idx / 5))
        for j in range(num_reps):
            fitness_function = suite.get_problem_by_function_dimension_instance(fid, dim, iid)
            fitness_function.observe_with(obs)
            f = fitnessFunc(fitness_function)
            cma = SingleSplitCMAES(
                dim, f, budget, representation=Utils.intToRepr(rep1),
                representation2=Utils.intToRepr(rep2), seed=j,
                split=split_target
            )
            cma.run_optimizer()
            HittingTimes.append(cma.used_budget)
            succeeded += fitness_function.final_target_hit
            fitness_function.free()
    return sum(HittingTimes) / max(succeeded, 1)


def single_split_with_hyperparams(fid, dim, rep1, rep2, split_idx, record_runs=False,
                                  iids=[1, 2, 3, 4, 5], num_reps=5, hyperparams=None):
    """
        Function running single-split CMA-ES with specific hyperparameters.

        :param fid: The function id (from bbob)
        :param dim: The dimension to run the bbob-problem in
        :param rep1: The configuration to run before the splitpoint
        :param rep2: The configuration to run after the splitpoint
        :param split_idx: The splitpoint-index at which to switch between rep1 and rep2
        :param iids: The instances of the bbob-function to run
        :param num_reps: The amount of repetitions to run
        :param record_runs: Whether or not to record a .dat-file during the runs of the CMA-ES
        :param hyperparams: Dictionary of the hyperparameters to use
        :return: The ERT over all num_reps runs on all instances in iids
    """
    if record_runs:
        obs = create_observer(rep1, rep2, split_idx)
    suite = cocoex.Suite("bbob", "", f"dimensions:{dim}")
    budget = dim * Config.budget_factor
    hittingtimes = []
    succeeded = 0
    for i, iid in enumerate(iids):
        split_target = get_target(fid, iid) + 10 ** (2 - (split_idx / 5))
        for j in range(num_reps):
            fitness_function = suite.get_problem_by_function_dimension_instance(fid, dim, iid)
            if record_runs:
                fitness_function.observe_with(obs)
            f = fitnessFunc(fitness_function)
            cma = SingleSplitCMAES(
                dim, f, budget, representation=Utils.intToRepr(rep1),
                representation2=Utils.intToRepr(rep2), seed=j,
                split=split_target
            )
            cma.set_hyperparameters(hyperparams)
            cma.run_optimizer()
            hittingtimes.append(cma.used_budget)
            print(cma.used_budget)
            succeeded += fitness_function.final_target_hit
            fitness_function.free()

    return sum(hittingtimes) / max(succeeded, 1)


def single_split_hyperparam_single(iid, rep_nr=None, hyperparams=None, hyperparams2=None, rep1=None, rep2=None,
                                   fid=None, split_idx=None, dim=None, record_runs=None, budget=None,
                                   target_idx = None, lambda_=None, lambda2_=None, opt_split=False):
    """
        Function handling single runs of single-split CMA-ES during hyperparameter optimization.

        :param fid: The function id (from bbob)
        :param dim: The dimension to run the bbob-problem in
        :param rep1: The configuration to run before the splitpoint
        :param rep2: The configuration to run after the splitpoint
        :param split_idx: The splitpoint-index at which to switch between rep1 and rep2
        :param iid: The instance of the bbob-function to run
        :param rep_nr: The repetition number (used for controlled randomness)
        :param record_runs: Whether or not to record a .dat-file during the runs of the CMA-ES
        :param hyperparams: Dictionary of the hyperparameters to use for C1
        :param hyperparams2: Dictionary of the hyperparameters to use for C2
        :param lambda_: Population size
        :return: The budget used and whether or not the final target was hit
    """
    if rep_nr is None:
        if len(iid) == 2:
            rep_nr = iid[1]
            iid = iid[0]
        else:
            raise Exception("Missing rep_nr parameter")

    if record_runs:
        obs = create_observer(rep1, rep2, split_idx)
    suite = cocoex.Suite("bbob", "", f"dimensions:{dim}")
    if budget is None:
        budget = dim * Config.budget_factor
    split_target = get_target(fid, iid, split_idx)
    fitness_function = suite.get_problem_by_function_dimension_instance(fid, dim, iid)
    if record_runs:
        fitness_function.observe_with(obs)
    if target_idx is not None:
        target = get_target(fid, iid, target_idx)
    else:
        target = None
    f = fitnessFunc(fitness_function, target)
    cma = SingleSplitCMAES(
        dim, f, budget, representation=Utils.intToRepr(rep1),
        representation2=Utils.intToRepr(rep2), seed=rep_nr,
        split=split_target, hyperparams=hyperparams,
        hyperparams2=hyperparams2, lambda_=lambda_, lambda2_=lambda2_
    )
    # cma.set_hyperparameters(hyperparams)
    cma.run_optimizer()
    target_hit = f.final_target_hit
    fitness_function.free()
    # print(cma.used_budget, target_hit)
    if opt_split:
        return cma.used_budget_at_split, cma.switched, cma.used_budget, target_hit
    return cma.used_budget, target_hit


def single_split_with_hyperparams_parallel(fid, dim, rep1, rep2, split_idx, record_runs=False, target_idx=None,
                                           iids=[1, 2, 3, 4, 5], num_reps=5, hyperparams=None, budget=None,
                                           hyperparams2=None, lambda_=None, lambda2_=None, opt_split=False):
    """
        Function handling parallel execution of single-split CMA-ES during hyperparameter optimization.

        :param fid: The function id (from bbob)
        :param dim: The dimension to run the bbob-problem in
        :param rep1: The configuration to run before the splitpoint
        :param rep2: The configuration to run after the splitpoint
        :param split_idx: The splitpoint-index at which to switch between rep1 and rep2
        :param iids: The instances of the bbob-function to run
        :param num_reps: The amount of repetitions to run
        :param record_runs: Whether or not to record a .dat-file during the runs of the CMA-ES
        :param hyperparams: Dictionary of the hyperparameters to use for C1
        :param hyperparams2: Dictionary of the hyperparameters to use for C2
        :param budget: Maximum number of evaluations per run
        :param target_idx: Custom target-index to stop the optimization (distance 10^(2-target_idx/5))
        :param lambda_: Population size
        :return: The ERT over all num_reps runs on all instances in iids
    """
    runFunction = partial(single_split_hyperparam_single, fid=fid, split_idx=split_idx, dim=dim, target_idx=target_idx,
                          record_runs=record_runs, rep1=rep1, rep2=rep2, hyperparams=hyperparams, budget=budget,
                          hyperparams2=hyperparams2, lambda_=lambda_, lambda2_=lambda2_, opt_split=opt_split)
    arguments = list(product(iids, range(num_reps)))
    run_data = runParallelFunction(runFunction, arguments)
    if opt_split:
        used_split, split_hit, used_budgets, hits = zip(*run_data)
        print((sum(used_budgets) / max(sum(hits), 1)), sum(hits))
        if hyperparams is None:
            append_str = "default"
        else:
            append_str = "optimized"
        with open(f"{datapath}Data_Full/F{fid}_rep{rep1}_{append_str}_split.csv", 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(used_split)
        with open(f"{datapath}Data_Full/F{fid}_rep{rep1}_{append_str}.csv", 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(used_budgets)
        return sum(used_split) / max(sum(split_hit), 1)
    else:
        used_budgets, hits = zip(*run_data)
        print((sum(used_budgets) / max(sum(hits), 1)), sum(hits))
        with open(f"{datapath}Data_Full/F{fid}_rep{rep1}.csv", 'a') as fd:
            writer = csv.writer(fd)
            writer.writerow(used_budgets)
        return sum(used_budgets) / max(sum(hits), 1)


def runSingleSplit_pycma(fid, dim, iids=[1, 2, 3, 4, 5], num_reps=5):
    """
        Function running single-split CMA-ES.

        :param fid: The function id (from bbob)
        :param dim: The dimension to run the bbob-problem in
        :param rep1: The configuration to run before the splitpoint
        :param rep2: The configuration to run after the splitpoint
        :param split_idx: The splitpoint-index at which to switch between rep1 and rep2
        :param iids: The instances of the bbob-function to run
        :param num_reps: The amount of repetitions to run
        :return: The ERT over all num_reps runs on all instances in iids
    """
    obs = create_observer("Pycma", None, None)
    suite = cocoex.Suite("bbob", "", f"dimensions:{dim}")
    HittingTimes = []
    for i, iid in enumerate(iids):
        for j in range(num_reps):
            fitness_function = suite.get_problem_by_function_dimension_instance(fid, dim, iid)
            fitness_function.observe_with(obs)
            f = fitnessFunc(fitness_function)
            cma = CMAEvolutionStrategy([0]*5, 0.5)
            cma.optimize(f)
            HittingTimes.append(cma.result.evaluations)
            print(cma.result.xbest)
            fitness_function.free()
    return np.mean(HittingTimes)

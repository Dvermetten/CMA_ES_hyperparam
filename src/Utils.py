from src.local import datapath
import cocoex
import numpy as np
from functools import partial
from multiprocessing import Pool
import src.Config as Config
from src.Adaptive_CMAES import SingleSplitCMAES
from modea.Utils import intToRepr
from src.local import datapath_npy, use_MPI

targets = np.load(f"{datapath_npy}/targets_bbob.npy")


def create_observer(rep1, rep2=None, split=None):
    """
        Creates a cocoex-observer and specifies the correct datapaths and algorithm name / info
        :return: a cocoex-observer
    """
    if rep2 is not None and split is not None:
        opts = f"result_folder:{datapath}/SingleSplit/Rep{rep1}To{rep2}At{split} " \
            f"algorithm_name:Single_split_{rep1}_To_{rep2} " \
            f"algorithm_info:Splitpoint is {split}"
    else:
        opts = f"result_folder:{datapath}/Static/Rep{rep1} " \
            f"algorithm_name: Static_{rep1}"
    obs = cocoex.Observer("bbob", opts.__str__())
    return obs


def get_target(fid, iid, target_idx=None):
    """
        Loads the bbob-targets for the specified (function,instance) pair
        This is required to determine splitpoints in single-split CMA-ES
        :return: the target for (fid,iid)
    """
    target = targets[fid - 1][iid]
    if target_idx is not None:
        target = target + 10**(2-target_idx/5)
    return(target)


def get_default_hyperparameter_values(params, dim, rep, budget=1000):
    """
        Reads the inital hyperparameter values from a single-split CMA-ES

        :param params: list of the required parameters
        :param dim: dimensionality of the problem
        :param rep: the algorithm variant
        :param budget: optional budget (should not change parameter values?)
        :return: the inital hyperparameter values for all keys in params
    """
    alg = SingleSplitCMAES(dim, None, representation=intToRepr(rep),
                           representation2=None, split=None, budget=budget, seed=0)
    return alg.get_default_values(params)


def runParallelFunction(runFunction, arguments):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this system

        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :return:
    """
    if use_MPI and Config.evaluate_parallel:
        return runMPI(runFunction, arguments)
    elif Config.allow_parallel and Config.evaluate_parallel:
        return runPool(runFunction, arguments)
    else:
        return runSingleThreaded(runFunction, arguments)


# Inline function definition to allow the passing of multiple arguments to 'runFunction' through 'Pool.map'
def func_star(a_b, func):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return func(*a_b)


def runPool(runFunction, arguments):
    """
        Small overhead-function to handle multi-processing using Python's built-in multiprocessing.Pool

        :param runFunction: The (``partial``) function to run in parallel, accepting ``arguments``
        :param arguments:   The arguments to passed distributedly to ``runFunction``
        :return:            List of any results produced by ``runFunction``
    """
    p = Pool(min(Config.num_threads, len(arguments)))

    local_func = partial(func_star, func=runFunction)
    results = p.map(local_func, arguments)
    p.close()
    return results


def runSingleThreaded(runFunction, arguments):
    """
        Small overhead-function to iteratively run a function with a pre-determined input arguments

        :param runFunction: The (``partial``) function to run, accepting ``arguments``
        :param arguments:   The arguments to passed to ``runFunction``, one run at a time
        :return:            List of any results produced by ``runFunction``
    """
    results = []
    for arg in arguments:
        results.append(runFunction(*arg))
    return results


def runMPI(runFunction, arguments):
    from schwimmbad import MPIPool
    with MPIPool() as pool:
        results = pool.map(runFunction, arguments)
    return results

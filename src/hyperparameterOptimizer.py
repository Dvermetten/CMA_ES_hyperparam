import cocoex
import numpy as np

# from mipego import mipego
# from mipego.Surrogate import RandomForest
# from mipego.SearchSpace import ContinuousSpace, NominalSpace

from GaussianProcess import GaussianProcess
from GaussianProcess.trend import constant_trend
from BayesOpt import BO
from BayesOpt.Surrogate import RandomForest
from BayesOpt.SearchSpace import ContinuousSpace, OrdinalSpace, ProductSpace, NominalSpace
from src.Algorithms import single_split_with_hyperparams_parallel
from src.Utils import get_default_hyperparameter_values
from modea.Utils import reprToInt


class hyperparameterOptimizer():
    def __init__(self, params=None, max_iter=None, n_init_sample=None, ub=None, lb=None,
                 eval_budget=None, n_random_start=None, part_to_optimize=1, param_vals=None, n_point=None):
        """
            Creates a hyperparameter optimizer

            :param fid: The function id (from bbob)
            :param dim: The dimension to run the bbob-problem in
            :param rep1: The configuration to run before the splitpoint
            :param rep2: The configuration to run after the splitpoint
            :param split_idx: The splitpoint-index at which to switch between rep1 and rep2
            :param iids: The instances of the bbob-function to run
            :param num_reps: The amount of repetitions to run
            :param part_to_optimize: Which part of the adaptive configuration to optimize. Can be 1, 2 or -1. The -1
            option optimizes both parts, with the same parameter values for both. To better optimize a complete
             configuration, first optimize part 1, then part 2 using the optimial value for part1 in the param_val
             argument.
            :param param_val: The parameter values for the part of the configuration which is not optimized here.
            :return: The result of the hyper-parameter optimzation
        """
        if params is None:
            self.params = ['c_1', 'c_c', 'c_mu']
        else:
            self.params = params

        if param_vals is not None:
            self.param_vals = param_vals
        else:
            self.param_vals = None
        self.part_to_optimize = part_to_optimize

        self.dim_hyperparams = len(self.params)

        if max_iter is None:
            self.max_iter = 100
        else:
            self.max_iter = max_iter
        if n_init_sample is None:
            self.n_init_sample = 20
        else:
            self.n_init_sample = n_init_sample
        if eval_budget is None:
            self.eval_budget = 20
        else:
            self.eval_budget = eval_budget
        if n_random_start is None:
            self.n_random_start = 5
        else:
            self.n_random_start = n_random_start
        if n_point is None:
            self.n_point = 1
        else:
            self.n_point = n_point
        self.set_bounds(lb, ub, params)
        if "lambda_" in self.params:
            print("Contains discrete variable (lambda_)")
            self.contains_discrete = True
            search_space_discrete = OrdinalSpace(list(zip([4], [250])))
            search_space_cont = ContinuousSpace(list(zip(self.lb, self.ub)))
            self.search_space = search_space_cont + search_space_discrete
            self.lb = np.append(self.lb, [4])
            self.ub = np.append(self.ub, [250])
        else:
            self.contains_discrete = False
            self.search_space = ContinuousSpace(list(zip(self.lb, self.ub)), var_name=params)
        # trend function of GPR
        # this is a standard setting. no need to change
        self.mean = constant_trend(self.dim_hyperparams, beta=0)

        # autocorrelation parameters of GPR
        self.thetaL = 1e-10 * (self.ub - self.lb) * np.ones(self.dim_hyperparams)
        self.thetaU = 2 * (self.ub - self.lb) * np.ones(self.dim_hyperparams)
        np.random.seed(0)
        self.theta0 = np.random.rand(self.dim_hyperparams) * (self.thetaU - self.thetaL) + self.thetaL

    def __call__(self, fid, dim, rep1, rep2, split_idx, iids=[1, 2, 3, 4, 5], num_reps=5, budget=None, target_idx=None,
                 sol_points=None, seed=0, verbose=False, log_file=None, data_file=None, opt_split=False):
        np.random.seed(seed)
        params = self.params
        if self.part_to_optimize == 1 or self.part_to_optimize == -1:
            initial_point = get_default_hyperparameter_values(params, dim, rep1, budget)
        else:
            initial_point = get_default_hyperparameter_values(params, dim, rep2, budget)
        if sol_points is None:
            initial_points = [initial_point]
        else:
            if isinstance(sol_points[0], list):
                initial_points = sol_points.append(initial_point)
            else:
                if initial_point != sol_points:
                    initial_points = [sol_points, initial_point]
                else:
                    initial_points = [initial_point]
        if self.param_vals is not None and self.param_vals not in initial_points:
            initial_points = initial_points.append(self.param_vals)

        def obj_func(x):
            if self.contains_discrete:
                lambda1_ = x[-1]
                lambda2_ = x[-1]
                x = x[:-1]
                params_i = [x for x in self.params if x != "lambda_"]
            else:
                lambda1_ = None
                lambda2_ = None
                params_i = self.params
            if self.part_to_optimize == 1 or self.part_to_optimize == -1:
                c1 = (params_i, x)
            elif self.param_vals is not None:
                if self.contains_discrete:
                    lambda1_ = self.param_vals[-1]
                    c1 = (params_i, self.param_vals[:-1])
                else:
                    c1 = (params_i, self.param_vals)
            else:
                c1 = None
            if self.part_to_optimize == 2 or self.part_to_optimize == -1:
                c2 = (params_i, x)
            elif self.param_vals is not None:
                if self.contains_discrete:
                    lambda2_ = self.param_vals[-1]
                    c2 = (params_i, self.param_vals[:-1])
                else:
                    c2 = (params_i, self.param_vals)
            else:
                c2 = None
            print(c1,c2,lambda1_,lambda2_)
            return single_split_with_hyperparams_parallel(fid, dim, rep1, rep2, split_idx,
                                                          iids=iids, num_reps=num_reps,
                                                          hyperparams=c1, hyperparams2=c2, budget=budget,
                                                          target_idx=target_idx, lambda_=lambda1_, lambda2_=lambda2_, opt_split=opt_split)
        if self.contains_discrete:
            model = RandomForest()

            opt = BO(self.search_space, obj_func, model, max_iter=self.max_iter,
                     n_init_sample=self.n_init_sample, minimize=True, verbose=verbose,
                     wait_iter=10, init_sol_points=initial_points, random_seed=seed, n_point=self.n_point,
                     optimizer='MIES', log_file=log_file, data_file=data_file
                     )
        else:
            model = GaussianProcess(mean=self.mean, corr='matern',
                                    theta0=self.theta0, thetaL=self.thetaL, thetaU=self.thetaU,
                                    nugget=1e-10, noise_estim=False,
                                    optimizer='BFGS', wait_iter=5, random_start=10 * self.dim_hyperparams,
                                    likelihood='concentrated', eval_budget=self.eval_budget, random_state=seed)

            opt = BO(self.search_space, obj_func, model, max_iter=self.max_iter,
                     n_init_sample=self.n_init_sample, minimize=True, verbose=verbose,
                     wait_iter=10, init_sol_points=initial_points, random_seed=seed, n_point=self.n_point,
                     optimizer='BFGS', log_file=log_file, data_file=data_file  # when using GPR model, 'BFGS' is faster than 'MIES'
                     )
        return opt.run()

    def set_bounds(self, lb, ub, params):
        params_cont = [x for x in params if x != "lambda_"]
        if lb is not None and len(lb) == len(params_cont):
            self.lb = lb
        else:
            self.lb = np.zeros(len(params_cont))
        if ub is not None and len(ub) == len(params_cont):
            self.ub = ub
        else:
            self.ub = [(get_default_hyperparameter_values(params_cont, 5, 0)[i]+0.1)*2 for i in range(len(params_cont))]


class OneSearchSpaceOptimizer:
    def __init__(self, max_iter=None, n_init_sample=None,
                 eval_budget=None, n_random_start=None, n_point=None):
        """
                    Creates a hyperparameter optimizer

                    :param fid: The function id (from bbob)
                    :param dim: The dimension to run the bbob-problem in
                    :param rep1: The configuration to run before the splitpoint
                    :param rep2: The configuration to run after the splitpoint
                    :param split_idx: The splitpoint-index at which to switch between rep1 and rep2
                    :param iids: The instances of the bbob-function to run
                    :param num_reps: The amount of repetitions to run
                    :param part_to_optimize: Which part of the adaptive configuration to optimize. Can be 1, 2 or -1. The -1
                    option optimizes both parts, with the same parameter values for both. To better optimize a complete
                     configuration, first optimize part 1, then part 2 using the optimial value for part1 in the param_val
                     argument.
                    :param param_val: The parameter values for the part of the configuration which is not optimized here.
                    :return: The result of the hyper-parameter optimzation
                """

        self.params = ['c_1', 'c_c', 'c_mu']#, 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'm8', 'm9', 'm10', 'm11']
        self.dim_hyperparams = len(self.params)

        if max_iter is None:
            self.max_iter = 1000
        else:
            self.max_iter = max_iter
        if n_init_sample is None:
            self.n_init_sample = 250
        else:
            self.n_init_sample = n_init_sample
        if eval_budget is None:
            self.eval_budget = 20
        else:
            self.eval_budget = eval_budget
        if n_random_start is None:
            self.n_random_start = 5
        else:
            self.n_random_start = n_random_start
        if n_point is None:
            self.n_point = 1
        else:
            self.n_point = n_point

        self.lb = np.zeros((3, 1))
        self.ub = [0.35,1,0.35]
        p1 = [[0, 1]] * 9
        p1.append([0, 1, 2])
        p1.append([0, 1, 2])

        search_space_nominal = NominalSpace(p1)
        # search_space_discrete = OrdinalSpace(list(zip([0,0,0,0,0,0,0,0,0,0,0], [2,2,2,2,2,2,2,2,2,3,3])))
        search_space_cont = ContinuousSpace(list(zip([0,0,0], [0.35,1,0.35])))
        self.search_space = search_space_cont + search_space_nominal

        self.mean = constant_trend(self.dim_hyperparams, beta=0)

        # autocorrelation parameters of GPR
        self.thetaL = 1e-10 * (self.ub - self.lb) * np.ones(self.dim_hyperparams)
        self.thetaU = 2 * (self.ub - self.lb) * np.ones(self.dim_hyperparams)
        np.random.seed(0)
        self.theta0 = np.random.rand(self.dim_hyperparams) * (self.thetaU - self.thetaL) + self.thetaL

    def __call__(self, fid, target, budget, dim=5, seed=0, log_file=None, data_file=None, verbose=True):

        def obj_func(x):
            print(x)
            conf = reprToInt(x[3:])
            return single_split_with_hyperparams_parallel(fid, dim, conf, conf, 51,
                                                          iids=[1,2,3,4,5], num_reps=5,
                                                          hyperparams=[self.params, x[:3]], hyperparams2=None,
                                                          budget=budget, target_idx=target)

        model = RandomForest()

        opt = BO(self.search_space, obj_func, model, max_iter=self.max_iter,
                 n_init_sample=self.n_init_sample, minimize=True, verbose=verbose,
                 wait_iter=10, random_seed=seed, n_point=self.n_point,
                 optimizer='MIES', log_file=log_file, data_file=data_file
                 )

        return opt.run()

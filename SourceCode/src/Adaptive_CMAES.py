from functools import partial
import numpy as np
import pickle
from modea import (
    Algorithms,
    Utils,
    Selection,
    Sampling,
    Mutation
)
from modea.Parameters import Parameters


class OnlineCMAOptimizer(Algorithms.CustomizedES):
    '''
    Base class for using an optimizer to optimize the structure of an CMA:

    Omitting local restart functionality for now:
        ~ rep[10] = None always

    '''
    _required_methods = ["update_structure_optimizer"]

    def __init_subclass__(cls):
        ''' 
        Method thats checks for correct subclass implementation
        '''
        body = dir(cls)
        for method in OnlineCMAOptimizer._required_methods:
            if not method in body:
                raise TypeError("{cls} doesn't implement {method}".format(**locals()))

    def __init__(self, n, fitness_function, budget, mu=None, lambda_=None, seed=None):
        '''
        Calls super init from Algorithms.CustomizedES
        ~ sets mutate parameters to adaptcovarance
        ~ set objective function target. 
        ~ Sets some default parameters
        ~ Stores mu/lamda in cache, in order to reset these 
        '''
        self.seed = seed
        np.random.seed(self.seed)
        self._lambda_ = lambda_
        self._mu = mu
        super(OnlineCMAOptimizer, self).__init__(
            n, fitness_function, budget, mu, lambda_
        )

        self.mutateParameters = self.parameters.adaptCovarianceMatrix
        self.representation = self.default_representation

        self.validate_parameters()

    @property
    def target_reached(self):
        return self.target - self.target_threshold >= self.best_individual.fitness

    @property
    def default_representation(self):
        return [0] * len(Utils.options)

    @property
    def default_parameters(self):
        return [None] * (len(Utils.initializable_parameters))

    @property
    def default_options(self):
        if self.representation is not None:
            return Utils.getOpts(self.representation)
        else:
            return Utils.getOpts(self.default_representation)

    @property
    def default_container(self):
        '''
        This property returns a default list container
        based on the number of choices/modules available
        '''
        return [[0] * x for x in Utils.num_options_per_module]

    @property
    def default_values(self):
        return Utils.getVals(self.default_parameters)

    def run_optimizer(self):
        # print("Running standard")
        '''
        Main method running the optimizer
        update structure optimizer is a requires subclass 
        method.
        '''
        while self.used_budget < self.budget and not self.fitnessFunction.final_target_hit:
            self.run_one_generation()
            self.record_statistics()
            self.update_structure_optimizer()
        return self

    def run_one_generation(self):
        '''Rename of function'''
        super(OnlineCMAOptimizer, self).runOneGeneration()

    def record_statistics(self):
        '''Rename of function'''
        super(OnlineCMAOptimizer, self).recordStatistics()

    def validate_parameters(self):
        # TODO: Check which parameters are required
        pass

    def get_sampler(self, options):
        '''
        Method for getting a correct sampler 
        based on a dictionary of options.
        '''
        if options["base-sampler"] == 'quasi-sobol':
            sampler = Sampling.QuasiGaussianSobolSampling(self.parameters.n)
        elif options["base-sampler"] == 'quasi-halton' and Sampling.halton_available:
            sampler = Sampling.QuasiGaussianHaltonSampling(self.parameters.n)
        else:
            sampler = Sampling.GaussianSampling(self.parameters.n)

        if options["orthogonal"]:
            orth_lambda = self.parameters.eff_lambda
            if options['mirrored']:
                orth_lambda = max(orth_lambda // 2, 1)
            sampler = Sampling.OrthogonalSampling(self.parameters.n, lambda_=orth_lambda, base_sampler=sampler)

        if options['mirrored']:
            sampler = Sampling.MirroredSampling(self.parameters.n, base_sampler=sampler)
        return sampler

    def change_parameter_weights(self):
        '''
        Reset hyperparameters after a configuration switch.
        '''

        self.parameters.weights = self.parameters.getWeights(self.parameters.weights_option)
        self.parameters.mu_eff = 1 / np.sum(np.square(self.parameters.weights))
        n = self.parameters.n
        mu_eff = self.parameters.mu_eff

        self.parameters.c_sigma = (mu_eff + 2) / (mu_eff + n + 5)

        self.parameters.c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)

        self.parameters.c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)

        self.parameters.c_mu = min(
            1 - self.parameters.c_1, self.parameters.alpha_mu * ((mu_eff - 2 + 1 / mu_eff) /
                                                                 ((
                                                                              self.parameters.n + 2) ** 2 + self.parameters.alpha_mu * mu_eff / 2)))

        self.parameters.damps = (
                1 + 2 * np.max([0, np.sqrt((mu_eff - 1) /
                                           (self.parameters.n + 1)) - 1]) + self.parameters.c_sigma)

        self.seq_cutoff = self.parameters.mu_int * self.parameters.seq_cutoff

        if self.parameters.active:
            # print("active")
            self.parameters.c_c = 2 / (n + np.sqrt(2)) ** 2

        self.parameters.flat_fitness_index = int(min(
            [np.ceil(0.1 + self.parameters.lambda_ / 4.0), self.parameters.mu_int - 1]))

        self.parameters.threshold = (
                self.parameters.init_threshold * self.parameters.diameter *
                ((1 - 0) / 1) ** self.parameters.decay_factor
        )

    def set_modules(self):
        '''
        Set modules after configuration switch.
        '''

        options = Utils.getOpts(self.representation)
        self.parameters.__dict__.update(**options)
        if self.parameters.selection != 'pairwise':
            selector = Selection.best
        else:
            selector = Selection.pairwise
            if options['sequential']:
                options['seq_cutoff'] = 2
                self.parameters.seq_cutoff = 2
        if self.lambda2_ is not None:
            self.parameters.lambda_ = self.lambda2_
        self.parameters.lambda_, \
        self.parameters.eff_lambda, \
        self.parameters.mu = self.calculateDependencies(
            options, self.parameters.lambda_, self.parameters.mu
        )
        self.select = lambda pop, new_pop, _, param: selector(pop, new_pop, param)
        self.mutate = partial(
            Mutation.CMAMutation,
            sampler=self.get_sampler(options),
            threshold_convergence=options["threshold"]
        )
        self.change_parameter_weights()

    def init_local_restart(self):
        parameter_opts = self.parameters.getParameterOpts()
        if self.representation[10] == 1:
            parameter_opts['local_restart'] = 'IPOP'
        elif self.representation[10] == 2:
            parameter_opts['local_restart'] = 'BIPOP'

        self.parameters.local_restart = parameter_opts['local_restart']
        if parameter_opts['lambda_']:
            self.lambda_init = parameter_opts['lambda_']
        elif parameter_opts['local_restart'] in ['IPOP', 'BIPOP']:
            self.lambda_init = int(4 + np.floor(3 * np.log(parameter_opts['n'])))
        else:
            self.lambda_init = None
        parameter_opts['lambda_'] = self.lambda_init

        # BIPOP Specific parameters
        self.lambda_ = {'small': None, 'large': self.lambda_init}
        self.budgets = {'small': None, 'large': None}
        self.regime = 'first'  # Later alternates between 'large' and 'small'

    def run_sub_optimizer(self, check_restart=True):
        # print("Running sub-optimizer")
        # The main evaluation loop
        while self.used_budget < self.budget \
                and not self.fitnessFunction.final_target_hit \
                and not (self.parameters.checkLocalRestartConditions(self.used_budget) and check_restart):
            self.run_one_generation()
            self.record_statistics()
            if self.update_structure_optimizer():
                return True
        return False

    def run_local_restart(self, stop_after_switch):
        # print("Running local restart")
        parameter_opts = self.parameters.getParameterOpts()
        while not self.fitnessFunction.final_target_hit and not self.total_used_budget >= self.total_budget:
            # Every local restart needs its own parameters, so parameter update/mutation must also be linked every time
            self.parameters = Parameters(**parameter_opts)
            if self.hyperparams is not None:
                self.set_hyperparameters(self.hyperparams)
            self.seq_cutoff = self.parameters.mu_int * self.parameters.seq_cutoff
            self.mutateParameters = self.parameters.adaptCovarianceMatrix

            self.initializePopulation()
            parameter_opts['wcm'] = self.population[0].genotype
            self.new_population = self.recombine(self.population, self.parameters)

            # Run the actual algorithm
            if self.run_sub_optimizer():
                self.total_used_budget += self.used_budget
                if stop_after_switch:
                    self.used_budget += self.total_used_budget
                    return True
            else:
                self.total_used_budget += self.used_budget
                # Increasing Population Strategies
                if parameter_opts['local_restart'] == 'IPOP':
                    parameter_opts['lambda_'] *= 2

                elif parameter_opts['local_restart'] == 'BIPOP':
                    try:
                        self.budgets[self.regime] -= self.used_budget
                        self.determineRegime()
                    except KeyError:  # Setup of the two regimes after running regularily for the first time
                        remaining_budget = self.total_budget - self.used_budget
                        self.budgets['small'] = remaining_budget // 2
                        self.budgets['large'] = remaining_budget - self.budgets['small']
                        self.regime = 'large'

                    if self.regime == 'large':
                        self.lambda_['large'] *= 2
                        parameter_opts['sigma'] = 2
                    elif self.regime == 'small':
                        rand_val = np.random.random() ** 2
                        self.lambda_['small'] = int(
                            np.floor(self.lambda_init * (.5 * self.lambda_['large'] / self.lambda_init) ** rand_val))
                        parameter_opts['sigma'] = 2e-2 * np.random.random()

                    self.budget = self.budgets[self.regime]
                    self.used_budget = 0
                    parameter_opts['budget'] = self.budget
                    parameter_opts['lambda_'] = self.lambda_[self.regime]
        #TODO Clean up used_budget and total_used budget implementation
        self.used_budget = self.total_used_budget


class StaticCMA(OnlineCMAOptimizer):
    '''class that runs only one configuration'''

    def __init__(self, *args, **kwargs):
        representation = kwargs.pop("representation")
        super(StaticCMA, self).__init__(*args, **kwargs)
        if representation:
            self.representation = representation
            self.set_modules()

    def update_structure_optimizer(self):
        '''Dummy method to prevent typeerror'''
        pass


class SingleSplitCMAES(OnlineCMAOptimizer):
    def __init__(self, *args, **kwargs):
        representation = kwargs.pop("representation")
        if representation:
            self.representation = representation
        self.representation2 = kwargs.pop("representation2")
        self.splitpoint = kwargs.pop("split")
        if "hyperparams" in kwargs:
            self.hyperparams = kwargs.pop("hyperparams")
        else:
            self.hyperparams = None
        if "hyperparams2" in kwargs:
            self.hyperparams2 = kwargs.pop("hyperparams2")
        else:
            self.hyperparams2 = None
        if "lambda2_" in kwargs:
            self.lambda2_ = kwargs.pop("lambda2_")
        else:
            self.lambda2_ = None
        super(SingleSplitCMAES, self).__init__(*args, **kwargs)
        if representation:
            self.representation = representation
            self.set_modules()
        self.switched = False
        self.set_hyperparameters(self.hyperparams)
        self.used_budget_at_split = self.budget

    def set_hyperparameters(self, hyperparameters):
        if hyperparameters is None:
            return
        if type(hyperparameters) is not dict:
            keys = hyperparameters[0]
            values = hyperparameters[1]
            hyperparameters = dict(zip(keys, values))
        for key in hyperparameters.keys():
            self.parameters.__dict__[key] = hyperparameters[key]

    def splitpoint_reached(self):
        return self.splitpoint >= self.best_individual.fitness

    def update_structure_optimizer(self):
        if not self.switched and self.splitpoint_reached():
            self.used_budget_at_split = self.used_budget
            self.representation = self.representation2
            self.set_modules()
            self.switched = True
            np.random.seed(self.seed)
            self.set_hyperparameters(self.hyperparams2)
            # print("Switching")
            return True

    def get_default_values(self, params):
        values = []
        for key in params:
            if key == "lambda":
                values.append(self.parameters.lambda_)
            else:
                values.append(self.parameters.__dict__[key])
        return values

    def save_status(self, filename):
        pickle.dump(self, filename)

    def run_optimizer(self):
        if self.representation[10] == 0 and self.representation2[10] == 0:
            super(SingleSplitCMAES, self).run_optimizer()
        elif self.representation[10] > 0 and self.representation2[10] > 0:
            self.init_local_restart()
            self.run_local_restart(False)
        else:
            if self.representation[10] > 0:
                self.init_local_restart()
                self.run_local_restart(True)
                super(SingleSplitCMAES, self).run_optimizer()
            else:
                self.run_sub_optimizer(False)
                self.init_local_restart()
                self.run_local_restart(False)

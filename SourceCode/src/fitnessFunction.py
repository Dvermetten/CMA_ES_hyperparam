
class fitnessFunc():
    def __init__(self, problem, custom_target=None):
        self.func = problem
        if custom_target is None:
            self.has_custom_target = False
        else:
            self.has_custom_target = True
            self.target = custom_target

    def __call__(self, *args, **kwargs):
        return self.func(args[0])

    @property
    def final_target_hit(self):
        if self.has_custom_target:
            return self.target > self.func.best_observed_fvalue1
        else:
            return self.func.final_target_hit




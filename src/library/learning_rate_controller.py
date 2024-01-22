import math
import bisect


class LearningRateController:
    """
    Learning rate controller, based on iteration or epoch round.
    """

    def __init__(self, name):
        self.name = name
        self.init_lr = None

    def set_init_lr(self, init_lr):
        self.init_lr = init_lr

    def get_lr(self, iteration, local_iteration, epoch, local_epoch, extra=0.1):
        """
        At each iteration or epoch, it gets learning rate through this function.
        Args:
            iteration (int): If epoch_or_iteration=="iteration", this means the current iteration round.
            local_iteration (int): The node local iteration, we do not use this information in the currently available
                                    LR controllers.
            epoch (int): If epoch_or_iteration=="epoch", this means the current epoch rounds.
            local_epoch (int): The node local epoch, we do not use this information in the currently available
                                LR controllers.
            extra (float): Some extra information you want to use.

        Return:
            learning rate (float): return current learning rate.
        """
        raise NotImplementedError


class ConstantLr(LearningRateController):
    """
    Constant Learning Rate.
    """

    def __init__(self, init_lr, epoch_or_iteration, *args, **kwargs):
        super(ConstantLr, self).__init__(name='constant_lr')
        self.set_init_lr(init_lr)
        self.epoch_or_iteration = epoch_or_iteration

    def get_lr(self, iteration=0, local_iteration=1, epoch=0, local_epoch=1, extra=0.1):
        return self.init_lr


class OneOverSqrtKLr(LearningRateController):
    """
    O(1/sqrt(k)) decreasing step size
    we choose proper constant so that variable 'decreasing_factor'
    is 1 at iteration 0 and 'final_proportion' at iteration 'total_iteration'
    """

    def __init__(self, init_lr, total_iteration=5000, total_epoch=10, epoch_or_iteration="iteration",
                 final_proportion=1 / 10, a=None, b=None, *args, **kwargs):
        # we choose proper constant so that 
        # variable 'decreasing_factor' is 1 at iteration 1 and
        # final_proportion at iteration 'total_iteration'
        super(OneOverSqrtKLr, self).__init__(name='one_over_sqrt_k_lr')
        self.set_init_lr(init_lr)
        self.epoch_or_iteration = epoch_or_iteration
        total = total_epoch if epoch_or_iteration == "epoch" else total_iteration
        if a is None or b is None:
            b = (total * final_proportion ** 2)  # / (1 - final_proportion ** 2)
            a = math.sqrt(b+1)
        self.a = a
        self.b = b

    def get_lr(self, iteration=0, local_iteration=1, epoch=0, local_epoch=1, extra=0.1):
        # a / sqrt(k+b) learning rate
        k = epoch if self.epoch_or_iteration == "epoch" else iteration
        decreasing_factor = self.a / math.sqrt(k + self.b)
        return self.init_lr * decreasing_factor


class OneOverKLr(LearningRateController):
    """
    O(1/k) decreasing step size
    we choose proper constant so that variable 'decreasing_factor'
    is 1 at iteration 1 and 1/10 at iteration 'total_iteration'
    """

    def __init__(self, init_lr, total_iteration=5000, total_epoch=10, epoch_or_iteration="iteration",
                 final_proportion=1 / 10, a=None, b=None, *args, **kwargs):
        # we choose proper constant so that 
        # variable 'decreasing_factor' is 1 at iteration 1 and
        # 1/final_proportion at iteration 'total_iteration'
        super(OneOverKLr, self).__init__(name='one_over_k_lr')
        self.set_init_lr(init_lr)
        self.epoch_or_iteration = epoch_or_iteration
        total = total_epoch if epoch_or_iteration == "epoch" else total_iteration
        if a is None or b is None:
            b = (total * final_proportion - 1)  # / (1 - final_proportion)
            if b <= 1:
                b = 1
            a = b + 1
        self.a = a
        self.b = b

    def get_lr(self, iteration=0, local_iteration=1, epoch=0, local_epoch=1, extra=0.1):
        # a / (k+b) learning rate
        k = epoch if self.epoch_or_iteration == "epoch" else iteration
        decreasing_factor = self.a / (k + self.b)
        return self.init_lr * decreasing_factor


class LadderLr(LearningRateController):
    """
    Customisation step size.
    """

    def __init__(self, init_lr, epoch_or_iteration="iteration",
                 decreasing_iter_ls=None, proportion_ls=None, *args, **kwargs):
        if proportion_ls is None:
            proportion_ls = []
        if decreasing_iter_ls is None:
            decreasing_iter_ls = []
        assert len(decreasing_iter_ls) == len(proportion_ls)
        super(LadderLr, self).__init__(name='ladder_lr')
        self.set_init_lr(init_lr)
        self.epoch_or_iteration = epoch_or_iteration
        self.decreasing_iter_ls = decreasing_iter_ls.copy()
        self.proportion_ls = proportion_ls.copy()
        if len(self.decreasing_iter_ls) == 0 or self.decreasing_iter_ls[0] != 0:
            self.decreasing_iter_ls.insert(0, 0)
            self.proportion_ls.insert(0, 1)

    def get_lr(self, iteration=0, local_iteration=1, epoch=0, local_epoch=1, extra=0.1):
        k = epoch if self.epoch_or_iteration == "epoch" else iteration
        pos = bisect.bisect_right(self.decreasing_iter_ls, k)
        return self.proportion_ls[pos - 1] * self.init_lr


class ConstantThenDecreasingLr(LearningRateController):
    """
    The learning rate is constant firstly, after few iteration or epoch, it becomes ration/t。
    """

    def __init__(self, init_lr, boundary_iteration=1000, boundary_epoch=2, epoch_or_iteration="iteration",
                 ratio=1, *args, **kwargs):
        super(ConstantThenDecreasingLr, self).__init__(name="constant_then_decreasing")
        self.set_init_lr(init_lr)
        self.boundary_iteration = boundary_iteration
        self.boundary_epoch = boundary_epoch
        self.epoch_or_iteration = epoch_or_iteration
        self.ratio = ratio
        self.boundary = boundary_epoch if epoch_or_iteration == "epoch" else boundary_iteration

    def get_lr(self, iteration=0, local_iteration=1, epoch=0, local_epoch=1, extra=0.1):
        current = epoch if self.epoch_or_iteration == "epoch" else iteration
        if current < self.boundary:
            return self.init_lr
        else:
            return min(current, self.ratio) / float(current)


class DecreasingStepLr(LearningRateController):
    """
    The learning rate is decreasing lr*step_ration every step interval.
    """

    def __init__(self, init_lr, step_interval_interation=30, step_interval_epoch=1, epoch_or_iteration="iteration",
                 step_ratio=0.9, nodes_cnt=10, *args, **kwargs):
        super(DecreasingStepLr, self).__init__(name="decreasing_step_lr")
        self.set_init_lr(init_lr)
        self.step_interval_interation = step_interval_interation
        self.step_interval_epoch = step_interval_epoch
        self.epoch_or_iteration = epoch_or_iteration
        self.step_ratio = step_ratio
        self.step_interval = step_interval_epoch if epoch_or_iteration == "epoch" else step_interval_interation
        self.lr = init_lr

    def get_lr(self, iteration=0, local_iteration=1, epoch=0, local_epoch=1, extra=0.1):
        current = epoch if self.epoch_or_iteration == "epoch" else iteration
        if current % self.step_interval == 0:
            self.lr = self.lr * self.step_ratio
            return self.lr
        else:
            return self.lr


class FollowOne(LearningRateController):
    """
    Follow the input rate, same with it or multiply by a factor. This is created for momentum rate, it can follow the
    learning rate.
    """

    def __init__(self, init_lr, epoch_or_iteration="iteration", multiple_ratio=1, *args, **kwargs):
        super(FollowOne, self).__init__(name="follow_one")
        self.set_init_lr(init_lr)
        self.multiple_ratio = multiple_ratio
        self.epoch_or_iteration = epoch_or_iteration

    def get_lr(self, iteration=0, local_iteration=1, epoch=0, local_epoch=1, extra=0.1):
        return extra * self.multiple_ratio

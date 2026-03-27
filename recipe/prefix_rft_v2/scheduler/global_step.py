import math
import torch
import numpy as np

class ConstController:

    def __init__(self, init=0, **kwargs):
        self.c = init

    def value(self, **kwargs):
        if "avg_scores" in kwargs:
            x = kwargs["avg_scores"]
            return torch.ones_like(x) * self.c
        return self.c

    def __str__(self):
        return f"Constant Controller with constant={self.c}"
    

class LinearDecayController:
    def __init__(self, init=1., target=0.0, n_steps=10000, warmup_ratio=0.1, **kwargs):
        # print(f"Using Linear Decay with init={init}, target={target}, n_steps={n_steps}, warmup_ratio={warmup_ratio}")
        self.n_steps = n_steps
        self.init = init
        self.target = target
        self.warmup_steps = int(n_steps * warmup_ratio)
        assert init > target

    def __str__(self):
        return f"Linear Decay with init={self.init}, target={self.target}, n_steps={self.n_steps}, warmup_steps={self.warmup_steps}"
    
    
    def value(self, **kwargs):
        cur_global_step = kwargs.get("global_step", None)
        assert cur_global_step is not None
         # Warmup phase: linearly increase from 0 to init
        if cur_global_step < self.warmup_steps:
            return (cur_global_step / self.warmup_steps) * self.init

        
        cur_step_after_warmup = cur_global_step - self.warmup_steps
        # If the current step exceeds the total steps, return the target value
        if cur_step_after_warmup >= self.n_steps:
            return self.target
        
        # Calculate the linear decay ratio
        decay_ratio = 1.0 - (cur_step_after_warmup / self.n_steps)
        
        # Interpolate between init and target based on the decay ratio
        return self.target + decay_ratio * (self.init - self.target)

# controllers based on global_steps
# class PeriodicConstantController:

#     def __init__(self, a, b, Ta, Tb, left_shift=0):
#         self.a = a
#         self.b = b
#         self.Ta = Ta
#         self.Tb = Tb
#         self.T = Ta + Tb
#         self.left_shift = left_shift

#     def value(self, **kwargs):
#         cur_global_step = kwargs.get("global_step", None)
#         x = cur_global_step + self.left_shift
#         x = x % self.T

#         if 0 <= x < self.Ta:
#             return self.a
#         elif self.Ta <= x < self.T:
#             return self.b
#         else:
#             raise NotImplementedError

# class ExponentialDecayController:
#     def __init__(self, init=1., target=0.0, decay_rate=0.999, n_steps=10000, warmup_ratio=0.1):
#         """
#         Initialize the exponential decay controller.

#         Parameters:
#         - init: Initial value, default is 1.0.
#         - target: Target value, default is 0.0.
#         - decay_rate: Exponential decay rate, in the range (0, 1]. 
#                       The closer to 1, the slower the decay; the smaller the value, the faster the decay.
#         - n_steps: Total number of steps for decay. After this number of steps, the target value is returned.
#         """
#         self.init = init
#         self.target = target
#         self.decay_rate = decay_rate
#         self.n_steps = n_steps
#         self.warmup_steps = int(warmup_ratio * n_steps) 
#         assert init > target, "Initial value must be greater than the target value"
#         assert 0 < decay_rate <= 1, "Decay rate must be within the range (0, 1]"

#     def value(self, **kwargs):
#         """
#         Calculate the exponential decay value based on the current step.

#         Parameters:
#         - global_step: Current global step, provided via kwargs.

#         Returns:
#         - The decayed value corresponding to the current step.
#         """
#         cur_global_step = kwargs.get("global_step", None)
#         assert cur_global_step is not None, "The global_step parameter must be provided"

#         # Warmup phase: linearly increase from 0 to init
#         if cur_global_step < self.warmup_steps:
#             return (cur_global_step / self.warmup_steps) * self.init

#         # Exp decay phase
#         cur_step_after_warmup = cur_global_step - self.warmup_steps
#         if cur_global_step > self.n_steps:
#             return self.target

#         # Calculate exponential decay
#         decayed_value = self.init * (self.decay_rate ** cur_step_after_warmup)
#         return max(decayed_value, self.target)


# class CosineDecayController:
#     def __init__(self, init=1., target=0.0, n_steps=10000, warmup_ratio=0.1):
#         self.n_steps = n_steps
#         self.init = init
#         self.target = target
#         self.warmup_steps = int(warmup_ratio * n_steps) 
#         assert init > target
#         # assert warmup_steps < n_steps  # Ensure warmup steps are less than total steps

#     def value(self, **kwargs):
#         cur_global_step = kwargs.get("global_step", None)
#         assert cur_global_step is not None

#         # Warmup phase: linearly increase from 0 to init
#         if cur_global_step < self.warmup_steps:
#             return (cur_global_step / self.warmup_steps) * self.init

#         # Cosine decay phase
#         cur_step_after_warmup = cur_global_step - self.warmup_steps
#         if cur_step_after_warmup > self.n_steps:
#             return self.target

#         decay_ratio = 0.5 * (1 + math.cos(math.pi * cur_step_after_warmup / self.n_steps))
#         y = self.target + decay_ratio * (self.init - self.target)
#         return y

class CosineDecayController:
    def __init__(self, init=0.1, target=1.0, n_steps=10000, warmup_ratio=0.0, **kwargs):
        print(f"Using Cosine Decay with init={init}, target={target}, n_steps={n_steps}, warmup_ratio={warmup_ratio}")
        self.n_steps = n_steps
        self.init = init
        self.target = target
        self.warmup_steps = int(warmup_ratio * n_steps)
        
        assert init != target, "init and target must be different"
        if init > target:
            self.mode = 'decay'
        else:
            self.mode = 'rise'

    def __str__(self):
        return f"Cosine Decay with init={self.init}, target={self.target}, n_steps={self.n_steps}, warmup_steps={self.warmup_steps}"

    def value(self, **kwargs):
        cur_global_step = kwargs.get("global_step", None)
        assert cur_global_step is not None

        # Warmup phase: linear increase from 0 to init
        if cur_global_step < self.warmup_steps:
            return (cur_global_step / self.warmup_steps) * self.init

        # Cosine phase
        cur_step_after_warmup = cur_global_step - self.warmup_steps
        if cur_step_after_warmup > self.n_steps:
            return self.target

        cos_progress = math.pi * cur_step_after_warmup / self.n_steps
        decay_ratio = 0.5 * (1 + math.cos(cos_progress))  # always goes from 1 -> 0

        if self.mode == 'decay':
            y = self.target + decay_ratio * (self.init - self.target)
        else:  # 'rise'
            y = self.init + (1 - decay_ratio) * (self.target - self.init)

        return y

CTRL_MAPPING = {
    "cosine_decay": CosineDecayController,
    "linear_decay": LinearDecayController,
    "const": ConstController
}

class DelayWrapper:

    def __init__(self, ctrl, delay_steps=0, delay_val=1.0, **kwargs):
        print(f"Using delay wrapper with delay_steps={delay_steps} and delay_val={delay_val}")
        self.ctrl = ctrl
        self.delay_steps = delay_steps
        self.delay_val = delay_val

    def __str__(self):
        return f"Delay ctrl with the base ctrl={self.ctrl}, delay_steps={self.delay_steps}, delay_val={self.delay_val}"

    def value(self, **kwargs):
        cur_global_step = kwargs.get("global_step", None)
        assert cur_global_step is not None
        if cur_global_step < self.delay_steps:
            return self.delay_val
        else:
            kwargs["global_step"] = cur_global_step - self.delay_steps
            out = self.ctrl.value(**kwargs)
            return out

class IDWrapper:

    def __init__(self, ctrl, **kwargs):
        self.ctrl = ctrl

    def value(self, **kwargs):
        return self.ctrl.value(**kwargs)

CTRL_WRAPPER_MAPPING = {
    "delay": DelayWrapper,
    "id": IDWrapper
}

# class UniformRandomWrapper:

#     def __init__(self, ctrl, left):
#         self.ctrl = ctrl
#         self.left = left

#     def value(self, **kwargs):
#         ctrl_out = self.ctrl.value(**kwargs)
#         # out = np.random.uniform(self.left, 1) * ctrl_out
#         out = np.random.uniform(self.left, ctrl_out)
#         return out

# class NoisyWrapper:

#     def __init__(self, ctrl, noise_scale=0.1):
#         self.ctrl = ctrl
#         self.noise_scale = noise_scale

#     def value(self, **kwargs):
#         ctrl_out = self.ctrl.value(**kwargs)
#         # ctrl_out = np.random.rand() * self.noise_scale + ctrl_out
#         ctrl_out = np.random.uniform(-1, 1) * self.noise_scale + ctrl_out
#         if ctrl_out > 0.95:
#             ctrl_out = 0.95
#         if ctrl_out < 0:
#             ctrl_out = 0.0
#         return ctrl_out

class BetaSampler:

    def __init__(self, low_ctrl, high_ctrl, alpha=1.0, beta=1.0,**kwargs):
        print(f"Using Beta sampleer with low_crl={low_ctrl}, ")
        self.low_ctrl = low_ctrl
        self.high_ctrl = high_ctrl
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return f"Beta sampler with low_ctrl={self.low_ctrl}, high_ctrl={self.high_ctrl}, alpha={self.alpha}, beta={self.beta}"

    def value(self, **kwargs):                                                                                          
        low = self.low_ctrl.value(**kwargs)
        high = self.high_ctrl.value(**kwargs)
        out = np.random.beta(self.alpha, self.beta)
        lower = min(low, high)
        higher = max(low, high)
        out = lower + (higher - lower) * out
        return out, lower, higher

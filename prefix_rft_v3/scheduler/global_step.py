import math

import numpy as np
import torch


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
    def __init__(self, init=1.0, target=0.0, n_steps=10000, warmup_ratio=0.1, **kwargs):
        self.n_steps = n_steps
        self.init = init
        self.target = target
        self.warmup_steps = int(n_steps * warmup_ratio)
        assert init > target

    def __str__(self):
        return (
            f"Linear Decay with init={self.init}, target={self.target}, "
            f"n_steps={self.n_steps}, warmup_steps={self.warmup_steps}"
        )

    def value(self, **kwargs):
        cur_global_step = kwargs.get("global_step", None)
        assert cur_global_step is not None
        if cur_global_step < self.warmup_steps:
            return (cur_global_step / self.warmup_steps) * self.init

        cur_step_after_warmup = cur_global_step - self.warmup_steps
        if cur_step_after_warmup >= self.n_steps:
            return self.target

        decay_ratio = 1.0 - (cur_step_after_warmup / self.n_steps)
        return self.target + decay_ratio * (self.init - self.target)


class CosineDecayController:
    def __init__(self, init=0.1, target=1.0, n_steps=10000, warmup_ratio=0.0, **kwargs):
        self.n_steps = n_steps
        self.init = init
        self.target = target
        self.warmup_steps = int(warmup_ratio * n_steps)
        assert init != target
        self.mode = "decay" if init > target else "rise"

    def __str__(self):
        return (
            f"Cosine Decay with init={self.init}, target={self.target}, "
            f"n_steps={self.n_steps}, warmup_steps={self.warmup_steps}"
        )

    def value(self, **kwargs):
        cur_global_step = kwargs.get("global_step", None)
        assert cur_global_step is not None

        if cur_global_step < self.warmup_steps:
            return (cur_global_step / self.warmup_steps) * self.init

        cur_step_after_warmup = cur_global_step - self.warmup_steps
        if cur_step_after_warmup > self.n_steps:
            return self.target

        cos_progress = math.pi * cur_step_after_warmup / self.n_steps
        decay_ratio = 0.5 * (1 + math.cos(cos_progress))

        if self.mode == "decay":
            return self.target + decay_ratio * (self.init - self.target)
        return self.init + (1 - decay_ratio) * (self.target - self.init)


CTRL_MAPPING = {
    "cosine_decay": CosineDecayController,
    "linear_decay": LinearDecayController,
    "const": ConstController,
}


class DelayWrapper:
    def __init__(self, ctrl, delay_steps=0, delay_val=1.0, **kwargs):
        self.ctrl = ctrl
        self.delay_steps = delay_steps
        self.delay_val = delay_val

    def __str__(self):
        return (
            f"Delay ctrl with the base ctrl={self.ctrl}, delay_steps={self.delay_steps}, "
            f"delay_val={self.delay_val}"
        )

    def value(self, **kwargs):
        cur_global_step = kwargs.get("global_step", None)
        assert cur_global_step is not None
        if cur_global_step < self.delay_steps:
            return self.delay_val
        kwargs["global_step"] = cur_global_step - self.delay_steps
        return self.ctrl.value(**kwargs)


class IDWrapper:
    def __init__(self, ctrl, **kwargs):
        self.ctrl = ctrl

    def value(self, **kwargs):
        return self.ctrl.value(**kwargs)


CTRL_WRAPPER_MAPPING = {
    "delay": DelayWrapper,
    "id": IDWrapper,
}


class BetaSampler:
    def __init__(self, low_ctrl, high_ctrl, alpha=1.0, beta=1.0, **kwargs):
        self.low_ctrl = low_ctrl
        self.high_ctrl = high_ctrl
        self.alpha = alpha
        self.beta = beta

    def __str__(self):
        return (
            f"Beta sampler with low_ctrl={self.low_ctrl}, high_ctrl={self.high_ctrl}, "
            f"alpha={self.alpha}, beta={self.beta}"
        )

    def value(self, **kwargs):
        low = self.low_ctrl.value(**kwargs)
        high = self.high_ctrl.value(**kwargs)
        out = np.random.beta(self.alpha, self.beta)
        lower = min(low, high)
        higher = max(low, high)
        out = lower + (higher - lower) * out
        return out, lower, higher

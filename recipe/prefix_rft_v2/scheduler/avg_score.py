import torch
import torch.nn.functional as F
# controller based on the avg_scores of the current rollout
# could be used for contrl the weight of the sft loss, kl loss, and entropy loss
class AvgScoreConstController:

    def __init__(self, c: float =0):
        self.c = c

    def value(self, **kwargs):
        if "avg_scores" in kwargs:
            x = kwargs["avg_scores"]
            if x is not None:
                return torch.ones_like(x) * self.c
        return self.c

class AvgScoreStepController:

    def __init__(self, left=0, right=1, thereshold=0.5):
        self.left = left
        self.right = right
        self.thereshold = thereshold

    def value(self, **kwargs):
        x = kwargs.get('avg_scores', None)
        assert x is not None
        return torch.where(
            x <= self.thereshold,
            torch.tensor(self.left, dtype=x.dtype),
            torch.tensor(self.right, dtype=x.dtype)
        )

class AvgScoreSigmoidController:

    def __init__(self, shift=0.0, scale=1.0, temp=1.0, bias=0.0):
        self.shift = shift
        self.scale = scale
        self.temp = temp
        self.bias = bias

    def value(self, **kwargs):
        x = kwargs.get('avg_scores', None)
        assert x is not None
        return torch.sigmoid(self.temp * (x - self.shift)) * self.scale + self.bias

class MetaAdvController:

    def __init__(self, out_func):
        self.out_func = out_func

    def value(self, **kwargs):
        x = kwargs.get("meta_adv", None)
        assert x is not None
        if self.out_func.lower() == "id":
            return x
        elif self.out_func.lower() == "relu":
            return F.relu(x)

    
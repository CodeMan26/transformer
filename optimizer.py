import math

import torch


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamWOptimizer, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m, v = state["m"], state["v"]
                g = p.grad.data  # gradient
                m = m * beta1 + (1 - beta1) * g
                v = v * beta2 + (1 - beta2) * g
                t = state.get("t", 0)
                bias1 = 1 - beta1**t
                bias2 = 1 - beta2**t
                alpha_t = (lr * math.sqrt(bias2)) / bias1
                p.data.addvdiv_(m, (v.sqrt() + eps), value=-alpha_t)
                if weight_decay > 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

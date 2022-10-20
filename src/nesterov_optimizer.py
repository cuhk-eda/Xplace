# We accelerate the nesterov optimizer in DREAMPlace by removing some 
# unused var and pre-computing step size

import torch
from torch.optim.optimizer import Optimizer, required
import numpy as np

import numba
import logging
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.WARNING)
@numba.jit(numba.float32[:](numba.int32), nopython=True, nogil=True, cache=True)
def calc_nesterov_step_size32(N):
    a_k = np.empty(N + 1, dtype=np.float32)
    a_k[0] = 1.0
    for i in range(1, len(a_k)):
        a_k[i] = (1 + np.sqrt(4 * np.power(a_k[i-1], 2) + 1)) / 2
    a_kp1 = np.roll(a_k, -1)[:-1]
    a_k = a_k[:-1]
    coef = (a_k - 1) / a_kp1
    return coef


class NesterovOptimizer(Optimizer):
    """
    @brief Follow the Nesterov's implementation of e-place algorithm 2
    http://cseweb.ucsd.edu/~jlu/papers/eplace-todaes14/paper.pdf
    """
    def __init__(self, params, lr=required):
        """
        @brief initialization
        @param params variable to optimize
        @param lr learning rate
        """
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # u_k is major solution
        # v_k is reference solution
        # obj_k is the objective at v_k
        # alpha_k is the step size
        # v_k_1 is previous reference solution
        # g_k_1 is gradient to v_k_1
        # obj_k_1 is the objective at v_k_1
        defaults = dict(lr=lr, u_k=[], v_k=[], g_k=[], obj_k=[], alpha_k=[], v_kp1 = [])
        super().__init__(params, defaults)
        self.max_cache_steps = 10000
        self.steps = 0
        coef = calc_nesterov_step_size32(self.max_cache_steps)
        self.last_coef = coef[-1].item()
        self.coef = torch.from_numpy(coef)

        if len(self.param_groups) != 1:
            raise ValueError("Only parameters with single tensor is supported")

    def __setstate__(self, state):
        super().__setstate__(state)

    def step(self, closure):
        """
        @brief Performs a single optimization step.
        @param closure A callable closure function that reevaluates the model and returns the loss.
        """
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                if not group['u_k']:
                    group['u_k'].append(p.data.clone())
                    # directly use p as v_k to save memory
                    group['v_k'].append(p)
                    obj, grad = closure(group['v_k'][i])
                    group['g_k'].append(grad.data.clone()) # must clone
                    group['obj_k'].append(obj.data.clone())
                u_k = group['u_k'][i]
                v_k = group['v_k'][i]
                g_k = group['g_k'][i]
                obj_k = group['obj_k'][i]
                if not group['alpha_k']:
                    # init alpha_k
                    self.coef = self.coef.to(g_k.device)
                    v_k_1 = (group['v_k'][i] - group['lr'] * g_k).detach().requires_grad_(True)
                    obj_k_1, g_k_1 = closure(v_k_1)
                    group['alpha_k'].append((v_k-v_k_1).norm(p=2) / (g_k-g_k_1).norm(p=2))
                alpha_k = group['alpha_k'][i]

                if not group['v_kp1']:
                    group['v_kp1'].append(torch.zeros_like(v_k, requires_grad=True))
                v_kp1 = group['v_kp1'][i]

                if self.steps < self.max_cache_steps:
                    coef = self.coef[self.steps]
                else:
                    coef = self.last_coef
                alpha_kp1 = 0
                backtrack_cnt = 0
                max_backtrack_cnt = 10

                while True:
                    u_kp1 = v_k - alpha_k * g_k
                    v_kp1.data.copy_(u_kp1 + coef * (u_kp1 - u_k))

                    f_kp1, g_kp1 = closure(v_kp1)

                    alpha_kp1 = (v_kp1.data-v_k.data).norm(2) / (g_kp1.data-g_k.data).norm(2)
                    backtrack_cnt += 1

                    if alpha_kp1 > 0.95 * alpha_k or backtrack_cnt >= max_backtrack_cnt:
                        break
                    alpha_k.data.copy_(alpha_kp1.data)

                obj = obj_k.data.clone()

                u_k.data.copy_(u_kp1.data)
                v_k.data.copy_(v_kp1.data)
                g_k.data.copy_(g_kp1.data)
                obj_k.data.copy_(f_kp1.data)
                alpha_k.data.copy_(alpha_kp1.data)

                self.steps += 1

                # although the solution should be u_k
                # we need the gradient of v_k
                # the update of density weight also requires v_k
                # I do not know how to copy u_k back to p when exit yet
                #p.data.copy_(v_k.data)

        return obj

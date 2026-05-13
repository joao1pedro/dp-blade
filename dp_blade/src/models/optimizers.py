import math
import numpy as np
from collections import deque
import torch
from torch.optim import Optimizer

class DopplerFilter:
    def __init__(self, params, a_coeffs=None, b_coeffs=None):
        self.params = list(params)
        if a_coeffs is None: a_coeffs = [-9.0/11.0]
        if b_coeffs is None: b_coeffs = [1.0/11.0, 1.0/11.0]
        self.a_coeffs = a_coeffs
        self.b_coeffs = b_coeffs
        self.na = len(a_coeffs)
        self.nb = len(b_coeffs) - 1
        self.m_history = [deque(maxlen=self.na) for _ in self.params]
        self.g_history = [deque(maxlen=self.nb + 1) for _ in self.params]
        self.c_a_history = deque(maxlen=self.na)
        self.c_b_history = deque(maxlen=self.nb + 1)
        for _ in range(self.na): self.c_a_history.append(0.0)
        for _ in range(self.nb + 1): self.c_b_history.append(0.0)

    def step(self):
        c_b_t = 1.0
        self.c_b_history.appendleft(c_b_t)
        term_a = sum(a * self.c_a_history[i] for i, a in enumerate(self.a_coeffs) if i < len(self.c_a_history))
        term_b = sum(b * self.c_b_history[i] for i, b in enumerate(self.b_coeffs) if i < len(self.c_b_history))
        c_a_t = -term_a + term_b
        self.c_a_history.appendleft(c_a_t)

        for i, p in enumerate(self.params):
            if p.grad is None: continue
            g_t = p.grad.data.clone()
            self.g_history[i].appendleft(g_t)
            sum_a = torch.zeros_like(g_t)
            for tau_idx, a_val in enumerate(self.a_coeffs):
                if tau_idx < len(self.m_history[i]): sum_a += a_val * self.m_history[i][tau_idx]
            sum_b = torch.zeros_like(g_t)
            for tau, b_val in enumerate(self.b_coeffs):
                if tau < len(self.g_history[i]): sum_b += b_val * self.g_history[i][tau]
            m_t = -sum_a + sum_b
            self.m_history[i].appendleft(m_t)
            m_hat = m_t / (c_a_t + 1e-12)
            p.grad.data = m_hat

class DopplerDPOptimizer(Optimizer):
    def __init__(self, dp_optimizer, model):
        self.defaults = {}
        self.dp_optimizer = dp_optimizer
        self.original_optimizer = dp_optimizer.original_optimizer
        self.filter = DopplerFilter([p for p in model.parameters() if p.requires_grad])

    def __getattr__(self, name):
        if name in ['dp_optimizer', 'original_optimizer', 'filter', 'defaults']: raise AttributeError
        return getattr(self.dp_optimizer, name)

    def zero_grad(self, set_to_none: bool = False):
        self.dp_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        self.dp_optimizer.step()
        self.filter.step()
        self.original_optimizer.step()
        return loss

class DiSKOptimizer(Optimizer):
    def __init__(self, dp_optimizer, model, gamma=0.5, kappa=0.7):
        self.defaults = {}
        self.dp_optimizer = dp_optimizer
        self.original_optimizer = dp_optimizer.original_optimizer
        self.model = model
        self.gamma = gamma
        self.kappa = kappa
        self.c1 = (1.0 - kappa) / (kappa * gamma)
        self.c2 = 1.0 - self.c1
        norm_factor = math.sqrt(self.c1**2 + self.c2**2)
        self.dp_optimizer.noise_multiplier = self.dp_optimizer.noise_multiplier / norm_factor
        self.state = {}
        for p in self.model.parameters():
            if p.requires_grad:
                self.state[p] = {
                    'd_prev': torch.zeros_like(p.data),
                    'g_kalman': torch.zeros_like(p.data)
                }

    def __getattr__(self, name):
        if name in ['dp_optimizer', 'original_optimizer', 'model', 'state', 'gamma', 'kappa', 'c1', 'c2', 'defaults']:
            raise AttributeError
        return getattr(self.dp_optimizer, name)

    def zero_grad(self, set_to_none: bool = False):
        self.dp_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure):
        original_weights = {}
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad:
                    original_weights[p] = p.data.clone()
                    if p in self.state:
                        p.data.add_(self.state[p]['d_prev'], alpha=self.gamma)
        self.dp_optimizer.zero_grad()
        loss = closure()
        grads_shifted = {}
        for p in self.model.parameters():
            if p.requires_grad and hasattr(p, "grad_sample") and p.grad_sample is not None:
                grads_shifted[p] = p.grad_sample.detach().clone()
        self.dp_optimizer.zero_grad()
        with torch.no_grad():
            for p, w_orig in original_weights.items():
                p.data.copy_(w_orig)
        loss = closure()
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad and p in grads_shifted:
                    if hasattr(p, "grad_sample") and p.grad_sample is not None:
                        p.grad_sample.mul_(self.c2).add_(grads_shifted[p], alpha=self.c1)
                    del grads_shifted[p]
            del grads_shifted

        if self.dp_optimizer.pre_step():
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.requires_grad and p.grad is not None:
                        state = self.state[p]
                        g_noisy = p.grad.data
                        g_pred = state['g_kalman']
                        state['g_kalman'].lerp_(g_noisy, weight=self.kappa)
                        p.grad.data.copy_(state['g_kalman'])
            self.original_optimizer.step()
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.requires_grad and p in self.state:
                        self.state[p]['d_prev'].copy_(p.data - original_weights[p])
        return loss

class IAKFBlockwiseDiSKOptimizer(Optimizer):
    def __init__(self, dp_optimizer, model, gamma=0.5, kappa_shift=0.7, A=1.0, alpha=0.05, q_scale=1e-2, kappa_min=0.01, kappa_max=0.99, eps=1e-8):
        self.defaults = {}
        self.dp_optimizer = dp_optimizer
        self.original_optimizer = dp_optimizer.original_optimizer
        self.model = model
        self.gamma, self.kappa_shift = gamma, kappa_shift
        self.A, self.alpha, self.q_scale = A, alpha, q_scale
        self.kappa_min, self.kappa_max, self.eps = kappa_min, kappa_max, eps
        self.base_noise_multiplier = getattr(dp_optimizer, "noise_multiplier", 1.0)
        self.max_grad_norm = getattr(dp_optimizer, "max_grad_norm", 1.0)
        self.sigma_dp_sq = (self.base_noise_multiplier * self.max_grad_norm) ** 2
        self._update_disk_coefficients(self.kappa_shift, self.gamma)
        self.dp_optimizer.noise_multiplier = self.base_noise_multiplier
        self.state = {}
        for p in self.model.parameters():
            if p.requires_grad:
                self.state[p] = {
                    "d_prev": torch.zeros_like(p.data),
                    "g_filt": torch.zeros_like(p.data),
                    "p_t": self.sigma_dp_sq,
                    "q_t": self.q_scale * self.sigma_dp_sq,
                    "C_t": self.sigma_dp_sq
                }
        self.step_stats = {'k_t': [], 'nu_sq_norm': [], 'p_t': [], 'q_t': [], 'C_t': []}

    def _update_disk_coefficients(self, kappa_val, gamma_val):
        gamma_safe = max(gamma_val, 1e-8)
        kappa_safe = max(min(kappa_val, 1.0 - 1e-8), 1e-8)
        self.c1 = (1.0 - kappa_safe) / (kappa_safe * gamma_safe)
        self.c2 = 1.0 - self.c1

    def __getattr__(self, name):
        if name in ["dp_optimizer", "original_optimizer", "model", "state", "defaults", "gamma", "kappa_shift", "A", "alpha", "c1", "c2", "base_noise_multiplier", "max_grad_norm", "sigma_dp_sq", "q_scale", "kappa_min", "kappa_max", "eps", "step_stats"]:
            raise AttributeError
        return getattr(self.dp_optimizer, name)

    def zero_grad(self, set_to_none: bool = False):
        self.dp_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure):
        original_weights = {}
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad:
                    original_weights[p] = p.data.clone()
                    p.data.add_(self.state[p]["d_prev"], alpha=self.gamma)
        self.dp_optimizer.zero_grad()
        loss = closure()
        grads_shifted = {}
        for p in self.model.parameters():
            if p.requires_grad and hasattr(p, "grad_sample") and p.grad_sample is not None:
                grads_shifted[p] = p.grad_sample.detach().clone()
        self.dp_optimizer.zero_grad()
        with torch.no_grad():
            for p, w_orig in original_weights.items():
                p.data.copy_(w_orig)
        loss = closure()
        with torch.no_grad():
            for p in self.model.parameters():
                if p.requires_grad and p in grads_shifted:
                    if hasattr(p, "grad_sample") and p.grad_sample is not None:
                        p.grad_sample.mul_(self.c2).add_(grads_shifted[p], alpha=self.c1)
        del grads_shifted

        if self.dp_optimizer.pre_step():
            layer_k_t, layer_nu_sq, layer_p_t, layer_q_t, layer_c_t = [], [], [], [], []
            with torch.no_grad():
                for p in self.model.parameters():
                    if not p.requires_grad or p.grad is None:
                        continue
                    state = self.state[p]
                    z_t = p.grad.data
                    g_prev = state["g_filt"]
                    g_pred = self.A * g_prev
                    p_pred = state["p_t"] + state["q_t"]
                    nu_t = z_t - g_pred
                    d_l = p.numel()
                    nu_sq_norm = nu_t.pow(2).sum().item() / d_l
                    state["C_t"] = self.alpha * state["C_t"] + (1.0 - self.alpha) * nu_sq_norm
                    state["q_t"] = max(0.0, state["C_t"] - self.sigma_dp_sq - state["p_t"])
                    k_t = p_pred / (p_pred + self.sigma_dp_sq + self.eps)
                    k_t = max(self.kappa_min, min(self.kappa_max, k_t))
                    state["g_filt"].copy_(g_pred + k_t * nu_t)
                    p.grad.data.copy_(state["g_filt"])
                    state["p_t"] = max((1.0 - k_t) * p_pred, self.eps)
                    layer_k_t.append(k_t if isinstance(k_t, float) else k_t.item())
                    layer_nu_sq.append(nu_sq_norm)
                    layer_p_t.append(state["p_t"])
                    layer_q_t.append(state["q_t"])
                    layer_c_t.append(state["C_t"])
            self.step_stats['k_t'].append(np.mean(layer_k_t) if layer_k_t else 0)
            self.step_stats['nu_sq_norm'].append(np.mean(layer_nu_sq) if layer_nu_sq else 0)
            self.step_stats['p_t'].append(np.mean(layer_p_t) if layer_p_t else 0)
            self.step_stats['q_t'].append(np.mean(layer_q_t) if layer_q_t else 0)
            self.step_stats['C_t'].append(np.mean(layer_c_t) if layer_c_t else 0)
            self.original_optimizer.step()
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.requires_grad:
                        self.state[p]["d_prev"].copy_(p.data - original_weights[p])
        del original_weights
        return loss

    def get_and_reset_epoch_stats(self):
        if not self.step_stats['k_t']: return {}
        res = {k: np.mean(v) for k, v in self.step_stats.items()}
        self.step_stats = {k: [] for k in self.step_stats}
        return res
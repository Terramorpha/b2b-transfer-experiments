"""Shared MLP actor-critic + PPO helpers (no EnergyPlus/b2b dependency).

Imported by both the BC clone (bc_fit_mlp.py, pure supervised) and the PPO
trainer/fine-tuner (train_mlp_specialist.py). Keeping the model definition in one
place is what lets a BC-cloned checkpoint deserialise back into the trainer.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # keep std in [~0.007, ~7.4] -> no runaway


class MLP(eqx.Module):
    layers: list

    def __init__(self, sizes, key):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [eqx.nn.Linear(a, b, key=k)
                       for a, b, k in zip(sizes[:-1], sizes[1:], keys)]

    def __call__(self, x):
        for lin in self.layers[:-1]:
            x = jax.nn.tanh(lin(x))
        return self.layers[-1](x)


class ActorCritic(eqx.Module):
    actor: MLP
    critic: MLP
    log_std: jax.Array

    def __init__(self, obs_dim, act_dim, hidden, key):
        ka, kc = jax.random.split(key)
        self.actor = MLP([obs_dim, *hidden, act_dim], ka)
        self.critic = MLP([obs_dim, *hidden, 1], kc)
        self.log_std = jnp.full((act_dim,), -0.5)  # std ~0.6


def gauss_logprob(raw, mean, log_std):
    var = jnp.exp(2.0 * log_std)
    return jnp.sum(-0.5 * ((raw - mean) ** 2 / var) - log_std
                   - 0.5 * jnp.log(2.0 * jnp.pi))


def squash_logdet(raw):
    return jnp.sum(jnp.log(1.0 - jnp.tanh(raw) ** 2 + 1e-6))


@eqx.filter_jit
def act(model, obs_n, key):
    """Sample: squashed action in [-1,1], the RAW pre-tanh sample (stored verbatim
    for the update -- arctanh reconstruction clips saturated actions and corrupts
    the PPO ratio), its log-prob, and the value."""
    ls = jnp.clip(model.log_std, LOG_STD_MIN, LOG_STD_MAX)
    mean = model.actor(obs_n)
    raw = mean + jnp.exp(ls) * jax.random.normal(key, mean.shape)
    logp = gauss_logprob(raw, mean, ls) - squash_logdet(raw)
    return jnp.tanh(raw), raw, logp, model.critic(obs_n)[0]


@eqx.filter_jit
def value(model, obs_n):
    return model.critic(obs_n)[0]


@eqx.filter_jit
def mean_action(model, obs_n):
    return jnp.tanh(model.actor(obs_n))  # deterministic eval action in [-1,1]


class RunningNorm:
    """Welford running mean/var over observations (numpy, CPU-side)."""

    def __init__(self, dim):
        self.mean = np.zeros(dim, np.float64)
        self.var = np.ones(dim, np.float64)
        self.count = 1e-4

    def update(self, x):
        b_mean = x.mean(0); b_var = x.var(0); b_n = x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean += delta * b_n / tot
        m_a = self.var * self.count
        m_b = b_var * b_n
        self.var = (m_a + m_b + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot

    def norm(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)


def compute_gae(rew, val, ep_end, boot_val, last_val, gamma, lam):
    T = len(rew)
    adv = np.zeros(T, np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        if ep_end[t]:
            nextv = boot_val[t]
            delta = rew[t] + gamma * nextv - val[t]
            gae = delta
        else:
            nextv = last_val if t == T - 1 else val[t + 1]
            delta = rew[t] + gamma * nextv - val[t]
            gae = delta + gamma * lam * gae
        adv[t] = gae
    return adv, adv + val


def make_update(opt):
    @eqx.filter_jit
    def update(model, opt_state, obs, act_raw, old_logp, adv, ret, clip, ent_c, vf_c):
        def loss_fn(m):
            ls = jnp.clip(m.log_std, LOG_STD_MIN, LOG_STD_MAX)
            def per(o, ar):
                mean = m.actor(o)
                logp = gauss_logprob(ar, mean, ls) - squash_logdet(ar)
                v = m.critic(o)[0]
                ent = jnp.sum(ls + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))
                return logp, v, ent
            logp, v, ent = jax.vmap(per)(obs, act_raw)
            ratio = jnp.exp(logp - old_logp)
            a = (adv - adv.mean()) / (adv.std() + 1e-8)
            pg = -jnp.mean(jnp.minimum(ratio * a,
                                       jnp.clip(ratio, 1 - clip, 1 + clip) * a))
            vf = jnp.mean((v - ret) ** 2)
            entropy = jnp.mean(ent)
            return pg + vf_c * vf - ent_c * entropy, (pg, vf, entropy)
        (loss, aux), g = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        upd, opt_state = opt.update(g, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, upd)
        return model, opt_state, loss, aux
    return update

"""
Tabular control: MC, SARSA, Q-learning, n-step SARSA, SARSA(λ)
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Literal, Optional

import gymnasium as gym
import numpy as np
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """experiment name"""
    seed: int = 1
    """random seed"""
    track: bool = False
    """log to W&B"""
    wandb_project_name: str = "tabular-control"
    """W&B project name"""
    wandb_entity: Optional[str] = None
    """W&B entity / team"""
    capture_video: bool = False
    """record videos of evaluation episodes"""

    # Env
    env_id: str = "FrozenLake-v1"
    """gymnasium env id (Discrete observation + action spaces required)"""

    # Training
    total_timesteps: int = 500_000
    """total environment steps"""
    learning_rate: float = 0.1
    """step size α"""
    gamma: float = 0.99
    """discount factor γ"""

    # Exploration (linear ε schedule, like CleanRL DQN)
    start_e: float = 1.0
    """initial ε"""
    end_e: float = 0.05
    """final ε"""
    exploration_fraction: float = 0.5
    """fraction of total_timesteps over which ε decays linearly"""

    # Algorithm
    algo: Literal["mc", "sarsa", "q_learning", "n_step_sarsa", "sarsa_lambda"] = "q_learning"
    """which control algorithm to run"""
    n_step: int = 4
    """n for n-step SARSA"""
    lambda_: float = 0.9
    """λ for SARSA(λ)"""
    trace_type: Literal["accumulating", "replacing"] = "replacing"
    """eligibility trace style for SARSA(λ)"""

    # Eval / logging
    eval_frequency: int = 10_000
    """evaluation period in env steps (0 disables eval)"""
    n_eval_episodes: int = 50
    """number of greedy episodes per evaluation"""


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    if capture_video and idx == 0:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id)
    env.action_space.seed(seed)
    return env


def linear_schedule(start_e: float, end_e: float, duration: float, t: int) -> float:
    slope = (end_e - start_e) / duration
    return max(end_e, start_e + slope * t)


def epsilon_greedy(Q: np.ndarray, s: int, eps: float, n_actions: int, rng) -> int:
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    qs = Q[s]
    best = np.flatnonzero(qs == qs.max()) # tie-break uniformly
    return int(rng.choice(best))


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------

def run_episode_mc(env, Q, args, eps, rng):
    """Every-visit on-policy MC control with ε-greedy policy."""
    s, _ = env.reset()
    states, actions, rewards = [], [], []
    ep_return = 0.0
    while True:
        a = epsilon_greedy(Q, s, eps, env.action_space.n, rng)
        s_next, r, term, trunc, _ = env.step(a)
        states.append(s); actions.append(a); rewards.append(r)
        ep_return += r
        s = s_next
        if term or trunc:
            break

    G = 0.0
    for t in reversed(range(len(states))):
        G = rewards[t] + args.gamma * G
        Q[states[t], actions[t]] += args.learning_rate * (G - Q[states[t], actions[t]])
    return ep_return, len(states)


def run_episode_sarsa(env, Q, args, eps, rng):
    """SARSA"""
    s, _ = env.reset()
    a = epsilon_greedy(Q, s, eps, env.action_space.n, rng)
    ep_return = 0.0
    ep_len = 0
    while True:
        s_next, r, term, trunc, _ = env.step(a)
        ep_return += r; ep_len += 1
        if term or trunc:
            target = r
            Q[s, a] += args.learning_rate * (target - Q[s, a])
            break
        a_next = epsilon_greedy(Q, s_next, eps, env.action_space.n, rng)
        target = r + args.gamma * Q[s_next, a_next]
        Q[s, a] += args.learning_rate * (target - Q[s, a])
        s, a = s_next, a_next
    return ep_return, ep_len


def run_episode_q_learning(env, Q, args, eps, rng):
    """Q-learning"""
    s, _ = env.reset()
    ep_return = 0.0
    ep_len = 0
    while True:
        a = epsilon_greedy(Q, s, eps, env.action_space.n, rng)
        s_next, r, term, trunc, _ = env.step(a)
        ep_return += r; ep_len += 1
        bootstrap = 0.0 if term else float(np.max(Q[s_next]))
        Q[s, a] += args.learning_rate * (r + args.gamma * bootstrap - Q[s, a])
        if term or trunc:
            break
        s = s_next
    return ep_return, ep_len


def run_episode_n_step_sarsa(env, Q, args, eps, rng):
    """n-step SARSA (Sutton & Barto §7.2)"""
    n, gamma, alpha = args.n_step, args.gamma, args.learning_rate
    n_actions = env.action_space.n

    s, _ = env.reset()
    a = epsilon_greedy(Q, s, eps, n_actions, rng)
    states, actions, rewards = [s], [a], []  # rewards[j] = R_{j+1}

    T = float("inf")
    t = 0
    ep_return = 0.0
    while True:
        if t < T:
            s_next, r, term, trunc, _ = env.step(actions[t])
            rewards.append(r); states.append(s_next); ep_return += r
            if term or trunc:
                T = t + 1
            else:
                actions.append(epsilon_greedy(Q, s_next, eps, n_actions, rng))
        tau = t - n + 1
        if tau >= 0:
            end = min(tau + n, T)  # exclusive
            G = 0.0
            for j in range(tau, end):
                G += gamma ** (j - tau) * rewards[j]
            if tau + n < T:
                G += gamma ** n * Q[states[tau + n], actions[tau + n]]
            Q[states[tau], actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])
        if tau == T - 1:
            break
        t += 1
    return ep_return, int(T)


def run_episode_sarsa_lambda(env, Q, E, args, eps, rng):
    """SARSA(λ), implemented with eligibility traces (Sutton & Barto §12.7)"""
    n_actions = env.action_space.n
    E.fill(0.0)
    s, _ = env.reset()
    a = epsilon_greedy(Q, s, eps, n_actions, rng)
    ep_return = 0.0
    ep_len = 0
    while True:
        s_next, r, term, trunc, _ = env.step(a)
        ep_return += r; ep_len += 1

        if term or trunc:
            delta = r - Q[s, a]
            a_next = None
        else:
            a_next = epsilon_greedy(Q, s_next, eps, n_actions, rng)
            delta = r + args.gamma * Q[s_next, a_next] - Q[s, a]

        if args.trace_type == "accumulating":
            E[s, a] += 1.0
        else:
            E[s, a] = 1.0

        Q += args.learning_rate * delta * E
        E *= args.gamma * args.lambda_

        if term or trunc:
            break
        s, a = s_next, a_next
    return ep_return, ep_len


# ---------------------------------------------------------------------------
# Eval (greedy w.r.t. Q)
# ---------------------------------------------------------------------------

def evaluate(env_id: str, Q: np.ndarray, n_episodes: int, seed: int):
    env = gym.make(env_id)
    returns = np.zeros(n_episodes)
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + 10_000 + ep)
        ret = 0.0
        while True:
            a = int(np.argmax(Q[s]))
            s, r, term, trunc, _ = env.step(a)
            ret += r
            if term or trunc:
                break
        returns[ep] = ret
    env.close()
    return float(returns.mean()), float(returns.std())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALGOS = {
    "mc": run_episode_mc,
    "sarsa": run_episode_sarsa,
    "q_learning": run_episode_q_learning,
    "n_step_sarsa": run_episode_n_step_sarsa,
    "sarsa_lambda": run_episode_sarsa_lambda,
}


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.algo}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = make_env(args.env_id, args.seed, 0, args.capture_video, run_name)
    assert isinstance(env.observation_space, gym.spaces.Discrete), \
        "tabular control requires a Discrete observation space"
    assert isinstance(env.action_space, gym.spaces.Discrete), \
        "tabular control requires a Discrete action space"
    env.reset(seed=args.seed)

    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    E = np.zeros_like(Q) if args.algo == "sarsa_lambda" else None

    run_fn = ALGOS[args.algo]
    needs_E = args.algo == "sarsa_lambda"

    global_step = 0
    episode = 0
    last_eval = 0
    start_time = time.time()
    return_window: list[float] = [] # rolling buffer of last 100 episode returns

    while global_step < args.total_timesteps:
        eps = linear_schedule(
            args.start_e, args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        if needs_E:
            ep_return, ep_len = run_fn(env, Q, E, args, eps, rng)
        else:
            ep_return, ep_len = run_fn(env, Q, args, eps, rng)

        global_step += ep_len
        episode += 1
        return_window.append(ep_return)
        if len(return_window) > 100:
            return_window.pop(0)

        writer.add_scalar("charts/episodic_return", ep_return, global_step)
        writer.add_scalar("charts/episodic_length", ep_len, global_step)
        writer.add_scalar("charts/epsilon", eps, global_step)
        writer.add_scalar("charts/episodic_return_mean100",
                          float(np.mean(return_window)), global_step)

        if args.eval_frequency > 0 and global_step - last_eval >= args.eval_frequency:
            mean_ret, std_ret = evaluate(args.env_id, Q, args.n_eval_episodes, args.seed)
            sps = int(global_step / (time.time() - start_time))
            writer.add_scalar("eval/mean_return", mean_ret, global_step)
            writer.add_scalar("eval/std_return", std_ret, global_step)
            writer.add_scalar("charts/SPS", sps, global_step)
            print(
                f"step={global_step:>8d}  episode={episode:>6d}  "
                f"eps={eps:.3f}  train_ret(100)={np.mean(return_window):.3f}  "
                f"eval_ret={mean_ret:.3f}±{std_ret:.3f}  sps={sps}"
            )
            last_eval = global_step

    env.close()
    writer.close()
    if args.track:
        wandb.finish()
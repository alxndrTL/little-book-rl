"""
Vanilla Policy Gradient (VPG) as described in the book.

Few implementation details not described in the book:
- specific initialization scheme for the policy network weights
- vectorized environment interaction (allows to collect multiple trajectories in parallel)
- Adam optimizer instead of gradient descent
- GAE (explained in the book) for both policy and critic updates
- multiple critic epochs per iteration
- advantage normalization, very common
"""

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    actor_learning_rate: float = 1e-2
    """the learning rate alpha for the policy network"""
    critic_learning_rate: float = 1e-2
    """the learning rate beta for the critic network"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_trajectories: int = 16
    """the number N of complete trajectories collected per gradient update"""
    gamma: float = 0.99
    """the discount factor gamma"""
    use_gae: bool = False
    """if toggled, use GAE-lambda for advantages instead of A_t = G_t - V(s_t)"""
    gae_lambda: float = 0.95
    """the lambda for GAE (only used if use_gae=True)"""
    critic_epochs: int = 1
    """number of critic gradient steps per iteration (pseudocode uses 1)"""
    norm_adv: bool = False
    """if toggled, normalize advantages to mean=0, std=1 before the policy update"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_dim = int(np.array(envs.single_observation_space.shape).prod())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_action(self, x):
        logits = self.actor(x)
        return Categorical(logits=logits).sample()

    def log_prob(self, x, action):
        logits = self.actor(x)
        return Categorical(logits=logits).log_prob(action)

    def value(self, x):
        return self.critic(x).squeeze(-1)


def reward_to_go(rewards, gamma):
    """G_t = sum_{k=t}^{T-1} gamma^{k-t} r_k."""
    G = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


def compute_gae(rewards, values, gamma, lam):
    """
    A_t = sum_{l>=0} (gamma * lam)^l * delta_{t+l} with delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) (computed backward)

    Treats the trajectory as ending at a terminal state: V(s_T) = 0.
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0  # V(terminal) = 0
        delta = rewards[t] + gamma * next_value - values[t]
        advantages[t] = last_gae = delta + gamma * lam * last_gae
    return advantages


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.actor_learning_rate)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.critic_learning_rate)

    # Per-env in-progress trajectory buffers (carried across iterations).
    ep_obs = [[] for _ in range(args.num_envs)]
    ep_actions = [[] for _ in range(args.num_envs)]
    ep_rewards = [[] for _ in range(args.num_envs)]

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)

    iteration = 0
    while global_step < args.total_timesteps:
        iteration += 1

        # =====================================================
        # 1. Collect N complete trajectories under pi_theta
        # =====================================================
        trajectories = []
        while len(trajectories) < args.num_trajectories:
            global_step += args.num_envs

            with torch.no_grad():
                action = agent.get_action(next_obs)

            new_next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminations, truncations)

            for i in range(args.num_envs):
                ep_obs[i].append(next_obs[i].cpu())
                ep_actions[i].append(action[i].cpu())
                ep_rewards[i].append(float(reward[i]))

                if done[i]:
                    rewards_arr = np.array(ep_rewards[i], dtype=np.float32)
                    G_t = reward_to_go(ep_rewards[i], args.gamma)
                    trajectories.append({
                        "obs": torch.stack(ep_obs[i]).to(device),
                        "actions": torch.stack(ep_actions[i]).to(device),
                        "rewards": rewards_arr,
                        "G_t": G_t,
                        "length": len(ep_rewards[i]),
                        "undisc_return": float(rewards_arr.sum()),
                        "disc_return": float(G_t[0]),
                    })
                    ep_obs[i].clear()
                    ep_actions[i].clear()
                    ep_rewards[i].clear()

            next_obs = torch.Tensor(new_next_obs).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        trajectories = trajectories[: args.num_trajectories]
        batch_obs = torch.cat([t["obs"] for t in trajectories])
        batch_actions = torch.cat([t["actions"] for t in trajectories])

        # =====================================================
        # 2. Compute advantages A_t (critic target stays G_t)
        # =====================================================
        with torch.no_grad():
            batch_V_old = agent.value(batch_obs).cpu().numpy() # baseline values, no grad

        adv_chunks = []
        offset = 0
        for traj in trajectories:
            T = traj["length"]
            V_traj = batch_V_old[offset:offset + T]
            if args.use_gae:
                adv = compute_gae(traj["rewards"], V_traj, args.gamma, args.gae_lambda)
            else:
                adv = traj["G_t"] - V_traj # A_t = G_t - V_phi(s_t)
            adv_chunks.append(adv)
            offset += T

        advantages = torch.from_numpy(np.concatenate(adv_chunks)).to(device)
        # Critic target: lambda-return G_t^lambda = A_t + V(s_t) under GAE, else G_t
        if args.use_gae:
            value_targets = torch.from_numpy(
                np.concatenate(adv_chunks) + batch_V_old
            ).to(device)
        else:
            value_targets = torch.from_numpy(
                np.concatenate([t["G_t"] for t in trajectories])
            ).to(device)

        if args.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # =====================================================
        # 3. Policy update
        #    theta <- theta + alpha * (1/N) * sum_i sum_t A_t^i * grad log pi(a_t^i | s_t^i)
        # =====================================================
        log_probs = agent.log_prob(batch_obs, batch_actions)
        actor_loss = -(advantages * log_probs).sum() / args.num_trajectories
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # =====================================================
        # 4. Critic update
        #    phi <- phi - beta * grad (1/N) * sum_i sum_t (1/2) (G_t - V_phi(s_t))^2
        # =====================================================
        for _ in range(args.critic_epochs):
            V_pred = agent.value(batch_obs)
            critic_loss = 0.5 * ((value_targets - V_pred) ** 2).sum() / args.num_trajectories
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

        # =====================================================
        # 5. Logging
        # =====================================================
        mean_undisc = float(np.mean([t["undisc_return"] for t in trajectories]))
        mean_disc = float(np.mean([t["disc_return"] for t in trajectories]))
        mean_len = float(np.mean([t["length"] for t in trajectories]))
        # Explained variance of V_phi against the true MC returns G_t
        batch_G_t = np.concatenate([t["G_t"] for t in trajectories])
        var_y = float(np.var(batch_G_t))
        explained_var = float("nan") if var_y == 0 else 1.0 - float(np.var(batch_G_t - batch_V_old)) / var_y

        writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
        writer.add_scalar("losses/critic_loss", critic_loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/mean_undiscounted_return", mean_undisc, global_step)
        writer.add_scalar("charts/mean_discounted_return", mean_disc, global_step)
        writer.add_scalar("charts/mean_episodic_length", mean_len, global_step)
        writer.add_scalar("charts/mean_value", float(batch_V_old.mean()), global_step)
        writer.add_scalar("charts/mean_advantage", advantages.mean().item(), global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"iter={iteration}, step={global_step}, "
              f"return={mean_undisc:.2f}, EV={explained_var:.2f}, SPS={sps}")

    envs.close()
    writer.close()
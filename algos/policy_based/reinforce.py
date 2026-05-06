"""
REINFORCE as described in the book.

Few implementation details not described in the book:
- specific initialization scheme for the policy network weights
- vectorized environment interaction (allows to collect multiple trajectories in parallel)
- Adam optimizer instead of gradient descent
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
    learning_rate: float = 1e-2
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_trajectories: int = 16
    """the number N of complete trajectories collected per gradient update"""
    gamma: float = 0.99
    """the discount factor gamma"""


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
    """No critic — only the policy network pi_theta."""

    def __init__(self, envs):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_action(self, x):
        logits = self.actor(x)
        return Categorical(logits=logits).sample()

    def log_prob(self, x, action):
        logits = self.actor(x)
        return Categorical(logits=logits).log_prob(action)


def reward_to_go(rewards, gamma):
    """G_t = sum_{k=t}^{T-1} gamma^{k-t} r_k (computed backward)"""
    G = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        G[t] = running
    return G


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

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate)

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
                    G_t = reward_to_go(ep_rewards[i], args.gamma)  # shape (T,)
                    trajectories.append({
                        "obs": torch.stack(ep_obs[i]).to(device),
                        "actions": torch.stack(ep_actions[i]).to(device),
                        "G_t": torch.from_numpy(G_t).to(device),  # per-timestep reward-to-go
                        "length": len(ep_rewards[i]),
                        "undisc_return": float(np.sum(ep_rewards[i])),
                        "disc_return": float(G_t[0]),  # G_0 = discounted return of the trajectory
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

        # Trim to exactly N (we may have collected up to num_envs-1 extra)
        trajectories = trajectories[: args.num_trajectories]

        # =====================================================
        # 2. REINFORCE gradient step
        #    g_hat = (1/N) * sum_i sum_t G_t^i * grad log pi(a_t^i | s_t^i)
        #    minimize  L(theta) = -(1/N) * sum_i sum_t G_t^i * log pi(a_t^i | s_t^i)
        # =====================================================
        batch_obs = torch.cat([t["obs"] for t in trajectories])
        batch_actions = torch.cat([t["actions"] for t in trajectories])
        # Each timestep gets its own reward-to-go G_t (as opposed to the complete trajectory return G in SPG)
        batch_G = torch.cat([t["G_t"] for t in trajectories])

        log_probs = agent.log_prob(batch_obs, batch_actions)
        loss = -(batch_G * log_probs).sum() / args.num_trajectories

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # =====================================================
        # 3. Logging
        # =====================================================
        mean_undisc = float(np.mean([t["undisc_return"] for t in trajectories]))
        mean_disc = float(np.mean([t["disc_return"] for t in trajectories]))
        mean_len = float(np.mean([t["length"] for t in trajectories]))
        writer.add_scalar("losses/policy_loss", loss.item(), global_step)
        writer.add_scalar("charts/mean_undiscounted_return", mean_undisc, global_step)
        writer.add_scalar("charts/mean_discounted_return", mean_disc, global_step)
        writer.add_scalar("charts/mean_episodic_length", mean_len, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        print(f"iter={iteration}, step={global_step}, mean_return={mean_undisc:.2f}, SPS={sps}")

    envs.close()
    writer.close()
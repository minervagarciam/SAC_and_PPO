import os
import random
import time
from dataclasses import dataclass
import csv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import shimmy  # registers dm_control environments with gymnasium


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
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "dm_control/pendulum-swingup-v0"
    """the environment id of the task"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient"""
    batch_size: int = 256
    """the batch size of sample from the replay memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """entropy regularization coefficient"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    max_grad_norm: float = 1.0
    """the maximum norm for gradient clipping"""


class RawRewardTracker(gym.Wrapper):
    """Accumulates raw episodic return before any reward-modifying wrappers.
    Stores the full episode return in info['raw_episodic_return'] at episode end.
    """
    def __init__(self, env):
        super().__init__(env)
        self._ep_return = 0.0

    def reset(self, **kwargs):
        self._ep_return = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_return += reward
        if terminated or truncated:
            info["raw_episodic_return"] = self._ep_return
            self._ep_return = 0.0
        return obs, reward, terminated, truncated, info


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = RawRewardTracker(env)  # must be before any reward-modifying wrappers
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(
            env,
            lambda obs: np.clip(obs, -10, 10),
            env.observation_space,
        )
        env.action_space.seed(seed)
        return env

    return thunk


class ReplayBuffer:
    """Simple replay buffer that stores transitions and samples random batches."""
    def __init__(self, buffer_size, obs_shape, action_shape, device):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        self.observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((buffer_size, *action_shape), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size):
        max_idx = self.buffer_size if self.full else self.pos
        indices = np.random.randint(0, max_idx, size=batch_size)

        return (
            torch.tensor(self.observations[indices], dtype=torch.float32).to(self.device),
            torch.tensor(self.next_observations[indices], dtype=torch.float32).to(self.device),
            torch.tensor(self.actions[indices], dtype=torch.float32).to(self.device),
            torch.tensor(self.rewards[indices], dtype=torch.float32).to(self.device),
            torch.tensor(self.dones[indices], dtype=torch.float32).to(self.device),
        )

    def __len__(self):
        return self.buffer_size if self.full else self.pos


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        action_dim = np.prod(env.single_action_space.shape)
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        # action rescaling to match environment's action space
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (np.clip(env.single_action_space.high, -1, 1) - np.clip(env.single_action_space.low, -1, 1)) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (np.clip(env.single_action_space.high, -1, 1) + np.clip(env.single_action_space.low, -1, 1)) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    args = tyro.cli(Args)
    safe_env_id = args.env_id.replace("/", "_")
    run_name = f"{safe_env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        envs.single_action_space.shape,
        device,
    )

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    csv_path = f"{run_name}_results.csv"

    try:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestep", "episodic_return"])

        for global_step in range(args.total_timesteps):
            # select action
            if global_step < args.learning_starts:
                actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
                actions = actions.detach().cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            # log episodic return at episode end
            for i, (term, trunc) in enumerate(zip(terminations, truncations)):
                if term or trunc:
                    ep_return = float(infos["raw_episodic_return"][i])
                    print(f"global_step={global_step}, episodic_return={ep_return:.2f}")
                    csv_writer.writerow([global_step, ep_return])
                    csv_file.flush()

            # handle truncation: use real next_obs for the buffer, not the reset obs
            real_next_obs = next_obs.copy()
            for idx, trunc in enumerate(truncations):
                if trunc and "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]

            # store transition in replay buffer
            rb.add(obs[0], real_next_obs[0], actions[0], rewards[0], terminations[0])

            obs = next_obs

            # training
            if global_step >= args.learning_starts:
                observations, next_observations, act, rews, dones = rb.sample(args.batch_size)

                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(next_observations)
                    qf1_next_target = qf1_target(next_observations, next_state_actions)
                    qf2_next_target = qf2_target(next_observations, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = rews + (1 - dones) * args.gamma * min_qf_next_target.view(-1)

                qf1_a_values = qf1(observations, act).view(-1)
                qf2_a_values = qf2(observations, act).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(qf1.parameters()) + list(qf2.parameters()), args.max_grad_norm
                )
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    for _ in range(args.policy_frequency):
                        pi, log_pi, _ = actor.get_action(observations)
                        qf1_pi = qf1(observations, pi)
                        qf2_pi = qf2(observations, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(observations)
                            alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                            a_optimizer.zero_grad()
                            alpha_loss.backward()
                            a_optimizer.step()
                            alpha = log_alpha.exp().item()

                if global_step % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                if global_step % 1000 == 0:
                    print("SPS:", int(global_step / (time.time() - start_time)))

    finally:
        envs.close()
        csv_file.close()
import gymnasium as gym 
from custom_envs import *
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
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
    episodes_captured: int = 1
    """how many episodes to capture"""
    freq_videos: int = 5_000
    """capture video every `freq_videos` steps"""

    # Algorithm specific arguments
    env_id: str = "Maze-Easy-v1"
    """the environment id of the task"""
    num_envs: int = 1
    """the number of parallel game environments to run"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.001
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # plasticity specific arguments
    eigen_vector: tuple = (1, 0)
    """the eigen vector of the covariance matrix"""
    lambda_plasticity: float = 0.1
    """the plasticity coefficient"""
    delta: int = 4
    """the time step between st and st+delta_t"""
    tau_covariance: float = 0.01
    """the covariance smoothing coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod()*2 + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, dx_delta,  a):
        x = torch.cat([x, dx_delta, a], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod()*2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x, dx_delta):
        x = torch.cat([x, dx_delta], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, dx_delta):
        mean, log_std = self(x, dx_delta)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

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
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    C = torch.zeros((np.prod(envs.single_observation_space.shape), np.prod(envs.single_observation_space.shape)))
    eigen_vector = torch.tensor(args.eigen_vector, dtype=torch.float32)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    dobs_delta = obs*0
    time_step = 0
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device), torch.Tensor(dobs_delta).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        time_step += 1

        # intrinsic reward
        intrinsic_reward = 0
        if rb.pos > args.delta:
            dobs = next_obs - obs
            dobs_delta =  rb.next_observations[rb.pos - args.delta + 1] - rb.observations[rb.pos - args.delta + 1]
            c = np.einsum('bi,bj->bij', dobs, dobs_delta)
            intrinsic_reward = np.abs(np.sum(np.einsum('...ij,...j->...i', c, dobs) * eigen_vector.numpy(), axis=1))
            rewards = rewards*0.0 + intrinsic_reward*1000
        



        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break
            time_step = 0

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            # data = rb.sample(args.batch_size)
            batch_inds = np.random.randint(args.delta, rb.pos, args.batch_size)
            batch_inds_delta = batch_inds - args.delta
            batch_inds_envs = np.random.randint(0, args.num_envs, args.batch_size)
            observations = torch.tensor(rb.observations[batch_inds, batch_inds_envs]).to(device)
            next_observations = torch.tensor(rb.next_observations[batch_inds, batch_inds_envs]).to(device)
            actions = torch.tensor(rb.actions[batch_inds, batch_inds_envs]).to(device)
            rewards = torch.tensor(rb.rewards[batch_inds, batch_inds_envs]).to(device)
            dones = torch.tensor(rb.dones[batch_inds, batch_inds_envs]).to(device)
            # -delta 
            observations_delta = torch.tensor(rb.observations[batch_inds_delta, batch_inds_envs]).to(device)
            next_observations_delta = torch.tensor(rb.next_observations[batch_inds_delta, batch_inds_envs]).to(device)
            d_obs_delta = next_observations_delta - observations_delta
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(next_observations, d_obs_delta)
                qf1_next_target = qf1_target(next_observations, d_obs_delta, next_state_actions)
                qf2_next_target = qf2_target(next_observations, d_obs_delta, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                # update C
                # d_obs = next_observations - observations
                # d_obs_delta = next_observations_delta - observations_delta
                # d_obs = d_obs - d_obs.mean(dim=0, keepdim=True)
                # d_obs_delta = d_obs_delta - d_obs_delta.mean(dim=0, keepdim=True)
                # cov_d_obs = torch.einsum('bi,bj->bij', d_obs, d_obs_delta)
                # C = (1 - args.tau_covariance) * C + args.tau_covariance * cov_d_obs.mean(dim=0)
                # intrinsic_reward= torch.abs(torch.sum(torch.einsum('ij,bj->bi', C, d_obs) * eigen_vector, dim=1))
                # rewards = rewards.flatten()*0.0 + intrinsic_reward*1e5
                # C = (1 - args.tau_covariance) * C + args.tau_covariance * torch.ger(observations.flatten(), observations.flatten())


                next_q_value = rewards.flatten() + (1 - dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(observations, d_obs_delta, actions).view(-1)
            qf2_a_values = qf2(observations, d_obs_delta, actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(observations, d_obs_delta)
                    qf1_pi = qf1(observations, d_obs_delta,  pi)
                    qf2_pi = qf2(observations, d_obs_delta,  pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(observations, d_obs_delta)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                print('mean reward :',rewards.mean().item())
                print('max intrinsic reward :',rewards.max().item())
                print('min reward :',rewards.min().item())
                # print('C :', C)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)
        if global_step % args.freq_videos == 0 and global_step > 0:
            env_plot = gym.make(args.env_id, render_mode=None, render = True)
            frames = []
            for k in range(args.episodes_captured): 
                print('episode played :', k)
                state, _ = env_plot.reset()
                d = False 
                obs_list = []
                next_obs_list = []
                dobs_delta = state*0
                dobs_delta = np.expand_dims(dobs_delta, axis=0) 
                while not d: 
                    dobs_delta = np.expand_dims(next_obs_list[-args.delta+1] - obs_list[-args.delta+1], axis = 0) if len(obs_list) > args.delta else dobs_delta
                    a, _, _= actor.get_action(torch.Tensor(state).to(device).unsqueeze(0), torch.Tensor(dobs_delta).to(device))
                    obs_list.append(state)
                    state, _, d, _ , _= env_plot.step(a[0].detach().cpu().numpy())
                    next_obs_list.append(state)
                    frames.append(env_plot.render())
            env_plot.save_video(frames = frames, filename=f"videos/{run_name}_episode_{global_step}.mp4", fps=4)
            env_plot.close()


    envs.close()
    writer.close()

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from torch.distributions.kl import kl_divergence


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
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRLTest"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.01 #1.0
    """the target network update rate"""
    target_network_frequency: int = 10 #500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    tau_soft : float = 0.5 #0.03 original implem
    """the temperature parameter for the soft-max policy"""
    alpha : float = 0.9 #0.9
    """the entropy regularization parameter"""
    l_0 : float = -1.0
    """the lower bound of the weighted log probability"""
    epsilon_tar : float = 1e-6
    """the epsilon term for numerical stability"""
    # TRANSFORMER SPECIFIC
    num_heads: int = 4
    """the number of attention heads"""
    attention_dim: int = 8
    """the dimension of the attention layer"""
    sequence_length: int = 2
    """the length of the sequence"""
    representation_dim: int = 4
    """the dimension of the representation layer"""
    lambda_forward : float = 0.2
    """the weight of the forward loss"""
    lambda_backward : float = 0.2
    """the weight of the backward loss"""



def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, attention_dim):
        super(MultiHeadAttention, self).__init__()
        assert attention_dim % num_heads == 0, "attention_dim doit être divisible par num_heads"
        
        self.num_heads = num_heads
        self.attention_dim_per_head = attention_dim // num_heads
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        
        self.final_linear = nn.Linear(attention_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Transformation des entrées en requêtes, clés et valeurs
        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.attention_dim_per_head)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.attention_dim_per_head)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.attention_dim_per_head)

        # Transposition pour obtenir la dimension des têtes en avant
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_length, attention_dim_per_head)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Calcul des scores d'attention avec scaling
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.attention_dim_per_head, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Application des scores d'attention aux valeurs
        weighted_values = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_length, attention_dim_per_head)
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        # Appliquer une dernière couche linéaire
        output = self.final_linear(weighted_values)

        return output, attention_weights


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, num_heads, attention_dim, sequence_length, representation_dim=32):
        super().__init__()
        self.mutli_head_attention = MultiHeadAttention(np.array(env.single_observation_space.shape).prod()-1, num_heads, attention_dim)
        self.layer_representation = nn.Linear(sequence_length*(np.array(env.single_observation_space.shape).prod()-1), representation_dim)
        self.network = nn.Sequential(
            nn.Linear(representation_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, env.single_action_space.n),
        )

    def forward(self, x):
        x= self.representation(x)
        # print('representation', x)
        # input('press')
        return self.network(x)

    def representation(self, x):
        x, _ = self.mutli_head_attention(x)
        x = x.view(x.size(0), -1)
        return self.layer_representation(x)
    
def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def soft_max(q_values, tau):
    return F.softmax(q_values / tau, dim=-1)



# def _get_samples(batch_inds, rb, env = None):
#     # Sample randomly the env idx
#     env_indices = np.random.randint(0, high=rb.n_envs, size=(len(batch_inds),))
#     if rb.optimize_memory_usage:
#         next_obs = rb._normalize_obs(rb.observations[(batch_inds + 1) % rb.buffer_size, env_indices, :], env)
#     else:
#         next_obs = rb._normalize_obs(rb.next_observations[batch_inds, env_indices, :], env)
#     data = (
#         rb._normalize_obs(rb.observations[batch_inds, env_indices, :], env),
#         rb.actions[batch_inds, env_indices, :],
#         next_obs,
#         # Only use dones that are not due to timeouts
#         # deactivated by default (timeouts is initialized as an array of False)
#         (rb.dones[batch_inds, env_indices] * (1 - rb.timeouts[batch_inds, env_indices])).reshape(-1, 1),
#         rb._normalize_reward(rb.rewards[batch_inds, env_indices].reshape(-1, 1), env),
#     )
#     return ReplayBufferSamples(*tuple(map(rb.to_torch, data)))

if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
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
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # bypass the attention_dim and sequence_length
    # args.attention_dim = envs.single_observation_space.shape[0] - 2 
    q_network = QNetwork(envs, num_heads=args.num_heads, attention_dim=args.attention_dim, sequence_length=args.sequence_length, representation_dim=args.representation_dim).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs, num_heads=args.num_heads, attention_dim=args.attention_dim, sequence_length=args.sequence_length,representation_dim=args.representation_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    action_dim = 1 if isinstance(envs.single_action_space, Discrete) else envs.single_action_space.shape[0]
    P_FORWARD = torch.nn.Parameter(torch.rand(args.representation_dim, args.representation_dim + action_dim), requires_grad=True).to(device)
    P_BACKWARD = torch.nn.Parameter(torch.rand(action_dim, args.representation_dim*2), requires_grad=True).to(device)
    rb = ReplayBuffer(
        args.buffer_size,
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(args.sequence_length, *envs.single_observation_space.shape), dtype=np.float32),
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    state = np.zeros((args.num_envs, args.sequence_length, *envs.single_observation_space.shape)) 
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    state[:, -1] = obs
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        # softmax policy
        with torch.no_grad():
            q_values = q_network(torch.Tensor(state[:,:, [0,2,3]]).to(device))
            # q_values = q_network(torch.tensor(state, dtype=torch.float32).to(device))
            # print('q_values', q_values) if global_step > args.learning_starts else None
            policy = soft_max(q_values, args.tau_soft)
            # print('policy', policy) if global_step > args.learning_starts else None
            # input('press') if global_step > args.learning_starts else None
            actions = torch.multinomial(policy, 1).squeeze(-1).cpu().numpy()
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)


        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        # update state
        next_state = np.roll(state, shift=-1, axis=1)
        next_state[:, -1] = next_obs.copy()
        
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
            if terminations[idx]:
                # reset state
                next_state[idx] = np.zeros((args.sequence_length, *envs.single_observation_space.shape)) 
                next_state[idx, -1] = real_next_obs[idx].copy() 
        rb.add(state, next_state, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        state = next_state

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                # custom sampling 
                batch_inds = np.random.randint(args.sequence_length, rb.pos, args.batch_size) if not rb.full else (np.random.randint(1+args.sequence_length, rb.buffer_size, size=args.batch_size) + rb.pos) % rb.buffer_size
                batch_state = torch.tensor(rb.observations[batch_inds]).squeeze(1).to(device)
                last_obs = batch_state[:,-1]
                batch_state = batch_state[:,:, [0,2,3]]
                batch_next_state = torch.tensor(rb.next_observations[batch_inds]).squeeze(1).to(device)
                batch_next_state = batch_next_state[:,:, [0,2,3]]
                batch_rewards = torch.tensor(rb.rewards[batch_inds]).to(device)
                batch_actions = torch.tensor(rb.actions[batch_inds]).squeeze(-1).to(device)
                batch_dones = torch.tensor(rb.dones[batch_inds]).to(device)
                with torch.no_grad():
                    # Q-Learning with Munchausen RL
                    target_q_values = target_network(batch_next_state)
                    target_policy = soft_max(target_q_values, args.tau_soft)
                    target_next_q_values = target_network(batch_next_state)
                    target_next_policy = soft_max(target_next_q_values, args.tau_soft)
                    red_term = args.alpha*(args.tau_soft*torch.log(target_policy.gather(1, batch_actions))+args.epsilon_tar).clamp(args.l_0, 0.0)
                    bleu_term = -args.tau_soft * torch.log(target_next_policy+args.epsilon_tar)
                    munchausen_target = batch_rewards + red_term + \
                                        args.gamma * (1 - batch_dones) * (target_next_policy * (target_next_q_values + bleu_term)).sum(dim=-1).unsqueeze(-1)
                    td_target = munchausen_target.squeeze()
                old_val = q_network(batch_state).gather(1, batch_actions).squeeze()
                loss = F.mse_loss(td_target, old_val)
                # Representation Learning
                r_s= q_network.representation(batch_state)
                r_ns_hat_a = torch.cat([r_s, batch_actions.float()], dim=1)
                r_ns_hat = torch.mm(r_ns_hat_a, P_FORWARD.t())
                # with torch.no_grad():
                r_ns= q_network.representation(batch_next_state)
                action_hat = torch.mm(torch.cat([r_s, r_ns], dim=1), P_BACKWARD.t())
                backward_loss = F.mse_loss(action_hat, batch_actions.float())
                forward_loss = F.mse_loss(r_ns_hat, r_ns)
                sheet_loss = F.mse_loss(r_s, last_obs)
                loss += args.lambda_forward*forward_loss + args.lambda_backward * backward_loss
                # loss += args.lambda_r*sheet_loss
                

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("losses/td_target", td_target.mean().item(), global_step)
                    writer.add_scalar("losses/log_policy", red_term.mean().item(), global_step)
                    writer.add_scalar("losses/entropy", bleu_term.mean().item(), global_step)
                    writer.add_scalar("losses/munchausen_target", munchausen_target.mean().item(), global_step)
                    writer.add_scalar("losses/forward_loss", forward_loss, global_step)
                    writer.add_scalar("losses/backward_loss", backward_loss, global_step)
                    writer.add_scalar("losses/sheet_loss", sheet_loss, global_step)
                    # print('representation_loss', representation_loss)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.05,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()

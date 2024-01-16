import numpy as np
import torch
import torch.distributions as tdist
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from typing import List, Tuple, Optional
from torch import nn
import warnings
from deephive.environment.utils import StdController
warnings.filterwarnings('ignore')

# set device for mps
# if not torch.backends.mps.is_available():
#     if not torch.backends.mps.is_built():
#         print("MPS not available because the current PyTorch install was not "
#               "built with MPS enabled.")
# else:
    # use cuda if available

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using device: {device}")

from deephive.environment.utils import parse_config
random_seed = 47

# Single Agent Memory Buffer
class Memory: 
    """
    Memory Buffer for storing the experiences of the agent.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        self.logprobs = []
        self.std_obs = []
    
    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.logprobs[:]
        del self.std_obs[:]


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int,  layer_size: List, 
                action_std_init : float = 0.2, variable_std: bool = True,):
        super(ActorCritic, self).__init__()
        """ 
        Actor Critic Network for the agent.
        Args:
            obs_dim: Dimension of the observation vector.
            action_dim: Dimension of the action vector.
            layer_size: List of layer sizes for the actor and critic networks.
            action_std_init: Initial standard deviation for the action distribution.
            variable_std: Whether the standard deviation should be fixed or variable.  
        """

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, layer_size[0]),
            nn.Tanh(),
            nn.Linear(layer_size[0], layer_size[1]),
            nn.Tanh(),
            nn.Linear(layer_size[1], action_dim),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, layer_size[0]),
            nn.Tanh(),
            nn.Linear(layer_size[0], layer_size[1]),
            nn.Tanh(),
            nn.Linear(layer_size[1], 1),
        )

        self.action_dim = action_dim
        self.variable_std=variable_std
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)   #action_std is a constant
        self.init_action_std = action_std_init
          
    def set_action_std(self, new_action_std : float):
                self.action_var = torch.full(
                    (self.action_dim,), new_action_std * new_action_std).to(device)
    
    def forward(self):
        raise NotImplementedError

    def get_std(self, std_obs:np.ndarray) -> torch.Tensor:
        # convert the std_obs to variance and then return the tensor
        action_vars = std_obs.pow(2)
        return action_vars


    def act(self, state: torch.Tensor, std_obs: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """ 
        Function to sample actions from the policy given the state.
        Args:
            state: Current state of the environment.
            std_obs: Standard deviation of the action distribution.
        Returns:
            action: Action sampled from the policy.
            action_logprob: Log probability of the action sampled from the policy.
        """
        action_mean = self.actor(state)  
        if self.variable_std:
            action_var = self.get_std(std_obs) # type: ignore
            dist = tdist.Normal(action_mean, action_var)
        else:
            cov_mat = torch.diag(self.action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, std_obs: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_value = self.critic(state)
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_mean = self.actor(state)
        if self.variable_std:
            action_var = self.get_std(std_obs) # type: ignore
            dist = tdist.Normal(action_mean, action_var)
        else:
            cov_mat = torch.diag(self.action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
        action_logprob = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprob, torch.squeeze(state_value), dist_entropy


class MAPPO:
    def __init__(self, config_file, **kwargs):
        """
        Multi-Agent Proximal Policy Optimization Algorithm. 
        Args:
            n_agents: Number of agents in the environment.
            n_dim: Dimension of the state vector.
            state_dim: Dimension of the state vector.
            action_dim: Dimension of the action vector.
            action_std: Initial standard deviation for the action distribution.
            variable_std: Whether the standard deviation should be fixed or variable.
            layer_size: Size of the hidden layers.
            lr: Learning rate.
            beta: Beta parameter for the Adam optimizer.
            gamma: Discount factor.
            K_epochs: Number of epochs for the policy and value function updates.
            eps_clip: Epsilon for the clipping in the PPO algorithm.
            pretrained: Whether to load a pretrained model or not.
            ckpt_folder: Path to the folder containing the pretrained model.
            initialization: Type of initialization for the weights of the neural networks.
        """
        config = parse_config(config_file)
        self.lr = config["lr"]
        self.kwargs = kwargs
        self.beta = config["beta"]
        self.gamma = config["gamma"]
        self.K_epochs = config["K_epochs"]
        self.eps_clip = config["eps_clip"]
        self.n_agents = config["n_agents"]
        self.n_dim = config["n_dim"]
        self.action_std = config["action_std"]
        self.obs_dim = config["obs_dim"]
        self.action_dim = config["action_dim"]
        self.layer_size = config["layer_size"]
        self.variable_std = config["variable_std"] if "variable_std" in config else False
        self.pretrained = config["pretrained"]
        self.ckpt_folder = config["ckpt_folder"]
        self.initialization = None
        
        self.std_controller = StdController(num_agents=self.n_agents,
                                            n_dim=self.n_dim,
                                            role_std=config["role_std"],
                                            decay_rate=config["decay_rate"],
                                            min_std=config["std_min"],
                                            max_std=config["std_max"])
        
        self.device = device
        self.buffer = Memory()

        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
        
        self.policy = ActorCritic(self.obs_dim, self.action_dim, action_std_init=self.action_std, layer_size=self.layer_size,
                                 variable_std=self.variable_std).to(self.device)
        
        if self.pretrained:
            pretrained_model = torch.load(
                self.ckpt_folder, map_location=lambda storage, loc: storage)
            self.policy.load_state_dict(pretrained_model)
            print(f"Loading pretrained model from {self.ckpt_folder}...")
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

        # old policy
        self.old_policy = ActorCritic(
            self.obs_dim, self.action_dim, action_std_init=self.action_std, layer_size=self.layer_size,
            variable_std=self.variable_std).to(device)
        
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()


    def __str__(self):
             return f"MAPPO: {self.n_agents} agents, {self.n_dim} dimensions, {self.action_std} action std, {self.lr} lr, {self.beta} beta, {self.gamma} gamma, {self.K_epochs} epochs, {self.eps_clip} eps_clip"

    def select_action(self, state, std_obs):
        state = torch.FloatTensor(state).to(device)  # Flatten the state
        action, action_logprob = self.policy.act(state, std_obs)
        for i in range(self.n_agents):
            self.buffer.states.append(state[i])
            self.buffer.std_obs.append(std_obs[i])
            self.buffer.actions.append(action[i])
            self.buffer.logprobs.append(action_logprob[i])
        return action.detach().cpu().numpy().flatten()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.old_policy.set_action_std(new_action_std)
        
    def decay_action_std(self, action_std_decay_rate, min_action_std=0.001, variable_std=False):
        if variable_std:
            print(f"not decaying action std because variable_std is set to {variable_std}")
            pass
        else:
            self.action_std = self.action_std * action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                print(f"setting action std to min_action_std {min_action_std}")
                self.action_std = min_action_std
            else:
                print(f"setting action std to {self.action_std}")
            self.set_action_std(self.action_std)
        
    def __get_buffer_info(self, buffer):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(buffer.rewards), reversed(buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = torch.stack(buffer.states).to(self.device).detach()
        old_actions = torch.stack(buffer.actions).to(self.device).detach()
        old_logprobs = torch.stack(buffer.logprobs).to(self.device).detach()
        old_std_obs = buffer.std_obs
        return rewards, old_states, old_actions, old_logprobs, old_std_obs

    def _update_old_policy(self, policy, old_policy, optimizer, rewards, old_states, old_actions, old_logprobs, old_std_obs):
        loss = None
        if old_std_obs is not None:
            old_std_obs = np.array(old_std_obs)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = policy.evaluate(
                old_states, old_actions, old_std_obs)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MSE_loss(state_values, rewards) - 0.001*dist_entropy

            # take gradient step
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

        # Copy new weights into old policy:
        old_policy.load_state_dict(policy.state_dict())
        return loss.mean().item() if loss is not None else None

    def update(self):
        # update gen eral policy
        rewards, old_states, old_actions, old_logprobs, old_std_obs = self.__get_buffer_info(
            self.buffer)
        loss = self._update_old_policy(self.policy, self.old_policy, self.optimizer,
                                    rewards, old_states, old_actions, old_logprobs, old_std_obs)
        if loss is not None:
            self.buffer.clear_memory()
            assert len(self.buffer.states) == 0
        else:
            print(f'[ERROR]: policy update failed')

    #save policy
    def save(self, filename, episode=0):
        torch.save(self.policy.state_dict(), filename + "/policy-" + str(episode) + ".pth")
        print("Saved policy to: ", filename)
    
    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename, map_location=lambda storage, loc: storage))
        print("Loaded policy from: ", filename)

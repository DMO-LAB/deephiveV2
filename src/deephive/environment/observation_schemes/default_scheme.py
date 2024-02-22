from deephive.environment.observation_schemes import ObservationScheme
from deephive.environment.utils import ScalingHelper
import numpy as np

scaler = ScalingHelper()

class DefaultObservationScheme(ObservationScheme):
    
    def generate_observation(self, pbest, use_gbest=False, std_type="euclidean", ratio=0.5, include_gbest=False):
        pbest[:, :-1] = scaler.scale(pbest[:, :-1], self.env.min_pos, self.env.max_pos)
        pbest[:, -1] = scaler.scale(pbest[:, -1], self.env.worst_obj_value, self.env.best_obj_value)
        if self.env.optimization_type == "minimum":
            gbest = pbest[np.argmin(pbest[:, -1])]
        else:
            gbest = pbest[np.argmax(pbest[:, -1])]
        agent_obs = [[] for _ in range(self.env.n_dim)]
        std_obs = [[] for _ in range(self.env.n_dim)]
        nbs = self._get_agent_neighbors()

        for agent in range(self.env.n_agents):
            agent_nb = nbs[agent]
            std = self._calculate_std(agent, gbest, std_type)

            for dim in range(self.env.n_dim):
                obs_values = self._get_obs_for_dim(agent, agent_nb, dim, pbest, gbest, use_gbest, include_gbest)
                agent_obs[dim].append(np.array([obs_values]))
                std_obs[dim].append(std)

        obs_length = agent_obs[0][0].shape[1]
        obss = [np.array(agent_obs[i]).reshape(self.env.n_agents, obs_length) for i in range(self.env.n_dim)]
        std_obss = [np.array(std_obs[i]).reshape(self.env.n_agents, 1) for i in range(self.env.n_dim)]

        return obss, self.assign_agent_roles(std_obss, ratio)


    def assign_agent_roles(self, std_obss, ratio):
        """
        Assign roles to agents based on their distance to zero in each dimension, 
        using a specified ratio to determine the split between roles.

        Args:
        std_obss (List[np.ndarray]): List of arrays, each containing distances of agents to zero in each dimension.
        ratio (float): The ratio of agents to be assigned role 1 (exploiter). 
                    Must be between 0 and 1, where 1 means all agents are role 1.

        Returns:
        List[np.ndarray]: List of binary arrays representing the roles of agents in each dimension.
        """
        roles_per_dimension = []

        for std_obs in std_obss:
            # Flatten the array if it's 2D with one column
            if std_obs.ndim == 2 and std_obs.shape[1] == 1:
                std_obs = std_obs.flatten()
            
            # Calculate the number of agents
            n_agents = len(std_obs)
            
            # Sort agents by their distance and get their indices
            sorted_indices = np.argsort(std_obs)
            
            # Determine the split index based on the ratio
            split_index = int(n_agents * ratio)
            
            # Assign roles: 1 for the closer ratio of agents, 0 for the rest
            roles = np.zeros(n_agents)
            roles[sorted_indices[:split_index]] = 1
            
            # Append the roles for this dimension
            roles_per_dimension.append(roles)

        return roles_per_dimension

            

    def _get_agent_neighbors(self):
        nbs = []
        agents_nbs = list(range(self.env.n_agents))
        for agent in range(self.env.n_agents):
            nbs.append(agent)
            choices = [ag for ag in agents_nbs if ag not in nbs]
            if len(choices) == 0:
                choices = [ag for ag in agents_nbs if ag != agent]
            agent_nb = np.random.choice(choices)
            nbs[-1] = agent_nb
        return nbs

    def _calculate_std(self, agent, gbest, std_type):
        if std_type == "euclidean":
            return np.sqrt(np.sum((self.env.state[agent][:-1] - gbest[:-1]) ** 2))
        else:
            return abs(gbest - self.env.state[agent])

    def _get_obs_for_dim(self, agent, agent_nb, dim, pbest, gbest, use_gbest, include_gbest=False):
        obs = [ 
                self.env.state[agent][self.env.n_dim],
                (self.env.state[agent][dim] - self.env.prev_state[agent][dim]),
                (self.env.state[agent][self.env.n_dim] - self.env.prev_state[agent][self.env.n_dim]),
                (self.env.state[agent][dim] - pbest[agent][dim]),
                (self.env.state[agent][self.env.n_dim] - pbest[agent][self.env.n_dim]),
                (self.env.state[agent][dim] - self.env.state[agent_nb][dim]),
                (self.env.state[agent][self.env.n_dim] - self.env.state[agent_nb][self.env.n_dim]),
                (self.env.state[agent][dim] - pbest[agent_nb][dim]),
                (self.env.state[agent][self.env.n_dim] - pbest[agent_nb][self.env.n_dim]),
        ]
        if include_gbest:
            obs.append(self.env.state[agent][dim] - gbest[dim])
            obs.append(self.env.state[agent][self.env.n_dim] - gbest[self.env.n_dim])
        if use_gbest: 
            # replace the third and fourth elements with the distance to gbest
            obs[3] = self.env.state[agent][dim] - gbest[dim]
            obs[4] = self.env.state[agent][self.env.n_dim] - gbest[self.env.n_dim]
        return obs

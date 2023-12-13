from environment.observation_schemes import ObservationScheme
from environment.utils import ScalingHelper
import numpy as np


class ExplorersObservationScheme(ObservationScheme):
    
    def generate_observation(self):
        self.agent_pos = self.env._get_actual_state()[:, :-1]
        self.novelty_scores = self.env.gmm.assess_novelty(self.agent_pos)
        agent_obs = [[] for _ in range(self.env.n_dim)]
        std_obs = [[] for _ in range(self.env.n_dim)]
        nbs = self._get_agent_neighbors()
        
        for agent in range(self.env.n_agents):
            agent_nb = nbs[agent]


            for dim in range(self.env.n_dim):
                obs_values = self._get_obs_for_dim(agent, agent_nb, dim)
                agent_obs[dim].append(np.array([obs_values]))

        obs_length = agent_obs[0][0].shape[1]
        obss = [np.array(agent_obs[i]).reshape(self.env.n_agents, obs_length) for i in range(self.env.n_dim)]
        return obss, self.assign_agent_roles()

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

    def assign_agent_roles(self):
        """
        Assign roles to agents based on their distance to zero in each dimension.
        
        Args:

        Returns:
        List[np.ndarray]: List of binary arrays representing the roles of agents in each dimension.
        """
        roles_per_dimension = []

        for agent in range(self.env.n_agents):
            # Calculate the number of agents
            n_agents = self.env.n_agents
            
            # Sort agents by their distance and get their indices
            sorted_indices = np.argsort(self.novelty_scores)
            
            # Assign roles: 1 for closer half, 0 for farther half
            roles = np.zeros(n_agents)
            roles[sorted_indices[:n_agents // 2]] = 1
            
            # Append the roles for this dimension
            roles_per_dimension.append(roles)


        return roles_per_dimension

    def _get_obs_for_dim(self, agent, agent_nb, dim):
        obs = [ #self.env.state[agent][dim],
                self.agent_pos[agent][dim] - self.env.prev_agents_pos[agent][dim], # agent position
                self.novelty_scores[agent], # agent novelty score
                self.agent_pos[agent][dim] - self.agent_pos[agent_nb][dim], # neighbor position
                self.novelty_scores[agent_nb], # neighbor novelty score
                self.env.prev_novelty_scores[agent], # previous agent novelty score
        ]
        return obs

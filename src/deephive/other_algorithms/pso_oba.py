import numpy as np 


class PSOBSA:
    def __init__(self, fitness_function, dimension, swarm_size, inertia_weight, 
                 acc_coefficients, mix_rate, mutation_probability, neighborhood_size,
                 lower_bound=-100, upper_bound=100, scale = False, apply_mutation=False):
        self.scale_option = scale
        self.num_evaluations = 0
        self.lower_bound = lower_bound  
        self.upper_bound = upper_bound
        self.obj_function = fitness_function
        self.dimension = dimension
        self.apply_mutation = apply_mutation
        self.swarm_size = swarm_size
        self.inertia_weight = inertia_weight
        self.acc_coefficients = acc_coefficients  # This is a tuple (c1, c2)
        self.mix_rate = mix_rate
        self.mutation_probability = mutation_probability
        self.neighborhood_size = neighborhood_size
        self.positions = np.random.uniform(low=lower_bound, high=upper_bound, size=(swarm_size, dimension))
        if self.scale_option:
            self.positions = self.scale(self.positions)
        self.velocities = np.zeros((swarm_size, dimension))
        self.pbest_positions = np.copy(self.positions)
        self.gbest_position = self.positions[np.argmin(self.fitness_function(self.positions))]
        self.fitness_values = self.fitness_function(self.positions)
        self.pbest_values = np.copy(self.fitness_values)
        
    def fitness_function(self, positions):
        self.num_evaluations += len(positions)
        if self.scale_option:
            return self.obj_function(self.unscale(positions))
        return self.obj_function(positions)
        
    def scale(self, observation: np.ndarray) -> np.ndarray:
        low = self.lower_bound
        high = self.upper_bound
        # Scale the observation to the range [0, 1]
        scaled_obs = (observation - low) / (high - low)
        # Stretch and shift the [0, 1] interval to [-1, 1]
        return scaled_obs 
    
    def unscale(self, observation: np.ndarray) -> np.ndarray:
        low = self.lower_bound
        high = self.upper_bound
        # Shift and compress the [-1, 1] interval back to [0, 1]
        unscaled_obs = observation 
        # Unscale the observation back to the original range
        return unscaled_obs * (high - low) + low
    
    def initialize_neighborhoods(self):
        self.neighborhoods = []
        for i in range(self.swarm_size):
            neighborhood_indices = list(range(i - self.neighborhood_size, i)) + list(range(i + 1, i + 1 + self.neighborhood_size))
            neighborhood_indices = [index % self.swarm_size for index in neighborhood_indices]  # Ensure indices are within bounds
            self.neighborhoods.append(neighborhood_indices)
    
    def compute_lbest(self):
        self.lbest_positions = np.zeros((self.swarm_size, self.dimension))
        for i in range(self.swarm_size):
            neighborhood_fitnesses = self.fitness_values[self.neighborhoods[i]]
            best_neighbor_idx = self.neighborhoods[i][np.argmin(neighborhood_fitnesses)]
            self.lbest_positions[i] = self.positions[best_neighbor_idx]
    
    def update_velocity(self):
        r1, r2 = np.random.rand(self.swarm_size, self.dimension), np.random.rand(self.swarm_size, self.dimension)
        cognitive_component = self.acc_coefficients[0] * r1 * (self.pbest_positions - self.positions)
        social_component = self.acc_coefficients[1] * r2 * (self.gbest_position - self.positions)
        self.velocities = self.inertia_weight * self.velocities + cognitive_component + social_component
    
    def update_positions(self):
        positions += self.velocities
        # confirm the position is within the search space
        if self.scale_option:
            positions = np.clip(positions, 0, 1)
        else:
            positions = np.clip(positions, self.lower_bound, self.upper_bound)
        if self.scale_option:
            #print(positions)
            assert np.all((positions >= 0) & (positions <= 1))
        else:
           # print(positions)
            assert np.all((positions >= self.lower_bound) & (positions <= self.upper_bound))
        #Boundary conditions and velocity limits would be handled here

    def calculate_fitness(self, positions):
        fitness_values = self.fitness_function(positions)
        
        # update personal best, global best and fitness values
        better_mask = fitness_values < self.fitness_values
        self.pbest_positions[better_mask] = positions[better_mask]
        self.fitness_values[better_mask] = fitness_values[better_mask]
        if np.min(fitness_values) < self.fitness_function(self.gbest_position.reshape(1, -1)):
            self.gbest_position = positions[np.argmin(fitness_values)]
        return fitness_values
        

    def mutate(self, positions, lbest_positions):
        # Eq. (12) from the description:
        # Tij = Xij + A(φ × Lbest - Xij)
        A = 2 * np.random.rand(self.swarm_size, self.dimension)
        phi = np.random.rand(self.swarm_size, self.dimension)
        return positions + A * (phi * lbest_positions - positions)
    
    def crossover(self, positions, trial_positions):
        # For each dimension, we have a mix rate chance of taking the value from the trial position
        crossover_mask = np.random.rand(self.swarm_size, self.dimension) < self.mix_rate
        new_positions = np.where(crossover_mask, trial_positions, positions)
        return new_positions

    # Placeholder for the main optimization loop
    def optimize(self, iterations):
        self.initialize_neighborhoods()
        for _ in range(iterations):
            self.compute_lbest()  # Compute Lbest positions
            self.update_velocity()
            positions = self.update_positions()
            fitness_value = self.calculate_fitness(positions)
            self.fitness_values = fitness_value
            # Mutation and Crossover
            if self.apply_mutation:
                self.positions = self.mutate_and_crossover(self.positions, self.lbest_positions)
            else:
                self.positions = positions
        return self.gbest_position
    
    def mutate_and_crossover(self, positions, lbest_positions):
        # Mutation and Crossover
        for i in range(self.swarm_size):
            if np.random.rand() < self.mutation_probability:
                trial_positions = self.mutate(positions, lbest_positions)
            else:
                self.permute_lbest(i)
                trial_positions = self.mutate(positions, lbest_positions[i])

            new_positions = self.crossover(positions, trial_positions)
            new_fitness = self.fitness_function(new_positions)

            # Selection-II: Update the personal and global bests
            better_mask = new_fitness < self.fitness_values
            positions[better_mask] = new_positions[better_mask]
            self.pbest_positions[better_mask] = new_positions[better_mask]
            self.fitness_values[better_mask] = new_fitness[better_mask]

            if np.min(new_fitness) < self.fitness_function(self.gbest_position.reshape(1, -1)):
                self.gbest_position = new_positions[np.argmin(new_fitness)]
        return positions
    
    def permute_lbest(self, particle_index):
        # Apply a permutation to the lbest of the selected particle
        # This is a simplification; in reality, you might want to shuffle the entire neighborhood
        neighbor_indices = self.neighborhoods[particle_index]
        np.random.shuffle(neighbor_indices)
        self.lbest_positions[particle_index] = self.positions[neighbor_indices[0]]


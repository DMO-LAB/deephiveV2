import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
import time

plt.rcParams['figure.figsize'] = [7, 7]
plt.rcParams['figure.dpi'] = 80
plt.rcParams['font.size'] = 12

def normalize(x, lower, upper):
    return (x - lower) / (upper - lower)

def unnormalize(x, lower, upper):
    return x * (upper - lower) + lower


class ParticleSwarmOptimizer:
    def __init__(self, func, lb, ub, args=(), kwargs={}, swarmsize=100, omega=0.5, phip=0.5, 
                 phig=0.5, maxiter=25, minstep=1e-8, minfunc=1e-8, debug=False, minimize=True, use_net=False,
                 net=None, normalize=False):
        self.func = func
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.args = args
        self.kwargs = kwargs
        self.swarmsize = swarmsize
        self.omega = omega
        self.phip = phip
        self.phig = phig
        self.maxiter = maxiter
        self.minstep = minstep
        self.minfunc = minfunc
        self.debug = debug
        self.global_best_values = []
        self.collected_data = []
        self.minimize = minimize
        self.use_net = use_net
        self.net = net
        self.normalize = normalize

        self._validate_inputs()
        self._initialize_swarm()

    def _validate_inputs(self):
        assert len(self.lb) == len(self.ub), 'Lower- and upper-bounds must be the same length'
        assert np.all(self.ub > self.lb), 'All upper-bound values must be greater than lower-bound values'

    def _initialize_swarm(self):
        self.vhigh = np.abs(self.ub - self.lb)
        self.vlow = -self.vhigh

        if not self.normalize:
            self.x = np.random.uniform(low=0, high=1, size=(self.swarmsize, len(self.lb))) * (self.ub - self.lb) + self.lb
        else:
            self.x = np.random.uniform(low=0, high=1, size=(self.swarmsize, len(self.lb)))
        self.v = np.random.uniform(low=self.vlow, high=self.vhigh, size=(self.swarmsize, len(self.lb)))
        self.p = np.copy(self.x)
        self.fp = np.array([self._evaluate_objective(p) for p in unnormalize(self.p, self.lb, self.ub)])
        self.pv = self.fp.copy()
        self.g = self.p[np.argmin(self.fp)] if self.minimize else self.p[np.argmax(self.fp)]
        self.fg = np.min(self.fp) if self.minimize else np.max(self.fp)
        self.fw = np.max(self.fp) if self.minimize else np.min(self.fp)

    def _evaluate_objective(self, x):
        try:
            y =  self.func.evaluate(np.array([x]).reshape(1, -1))
        except Exception as e:
            y = self.func(x, *self.args, **self.kwargs)
        return y.item()

    def _update_velocity(self, i):
        rp = np.random.uniform(size=len(self.lb))
        rg = np.random.uniform(size=len(self.lb))
        self.v[i] = self.omega * self.v[i] + self.phip * rp * (self.p[i] - self.x[i]) + self.phig * rg * (self.g - self.x[i])
        for dim in range(len(self.v[i])):
            self.data_point[f"velocity_{dim}"] = self.v[i][dim]
            self.data_point[f"cognitive_velocity_{dim}"] = (self.p[i][dim] - self.x[i][dim])
            self.data_point[f"social_velocity_{dim}"] = (self.g[dim] - self.x[i][dim])
            self.data_point[f"position_{dim}"] = self.x[i][dim] if not self.normalize else unnormalize(self.x[i][dim], self.lb[dim], self.ub[dim])

    def _update_velocity_with_net(self, i):
        inference_input = []
        for dim in range(len(self.v[i])):
            inference_input.append(self.x[i][dim])
            inference_input.append(self.p[i][dim] - self.x[i][dim])
            inference_input.append(self.g[dim] - self.x[i][dim])
            inference_input.append(self.fp[i])
            inference_input.append(self.fg)
            inference_input.append((self.pv[i] - self.fw) / (self.fg - self.fw))

            inference_input = np.nan_to_num(inference_input, nan=0)
            inference_tensor = torch.tensor(inference_input).float().unsqueeze(0)  # Add batch dimension
            with torch.no_grad():  # Ensure no gradients are computed
                predicted_velocity_tensor = self.net(inference_tensor)
                self.data_point[f"velocity_{dim}"] = predicted_velocity_tensor.item()
            inference_input = []
        self.v[i] = predicted_velocity_tensor.item()

    def _update_position(self, i):
        self.x[i] += self.v[i]
        if not self.normalize:
            self.x[i] = np.clip(self.x[i], self.lb, self.ub)
        else:
            self.x[i] = np.clip(self.x[i], 0, 1)


    def _update_personal_best(self, i):
        fx = self._evaluate_objective(self.x[i]) if not self.normalize else self._evaluate_objective(unnormalize(self.x[i], self.lb, self.ub))
        self.pv[i] = fx
        if self.minimize:
            if fx < self.fp[i]:
                self.p[i] = self.x[i]
                self.fp[i] = fx
        else:
            if fx > self.fp[i]:
                self.p[i] = self.x[i]
                self.fp[i] = fx
        self.data_point["personal_best"] = self.fp[i]

    def _update_global_best(self, i):
        if self.minimize:
            if self.fp[i] < self.fg:
                self.g = self.p[i]
                self.fg = self.fp[i]
                if self.debug:
                    print(f'New best for swarm at iteration {self.it}: {self.g} {self.fg}')
            if self.fp[i] > self.fw:
                self.fw = self.fp[i]
        else:
            if self.fp[i] > self.fg:
                self.g = self.p[i]
                self.fg = self.fp[i]
                if self.debug:
                    print(f'New best for swarm at iteration {self.it}: {self.g} {self.fg}')
            if self.fp[i] < self.fw:
                self.fw = self.fp[i]
        self.data_point["global_best"] = self.fg

    def optimize(self, neptune_logger=None):
        self.it = 1
        self.pos_history = np.array([self.x]) if not self.normalize else np.array([unnormalize(self.x, self.lb, self.ub)])
        self.history = [self.fg]
        while self.it <= self.maxiter:
            for i in range(self.swarmsize):
                self.data_point = {}
                self.data_point["iteration"] = self.it
                self._update_velocity(i) if not self.use_net else self._update_velocity_with_net(i)
                self._update_position(i)
                self._update_personal_best(i)
                self._update_global_best(i)
            self.pos_history = np.append(self.pos_history, [self.x], axis=0) if not self.normalize else np.append(self.pos_history, [unnormalize(self.x, self.lb, self.ub)], axis=0)
            self.history.append(self.fg)
            if self.debug:
                print(f'Best after iteration {self.it}: {self.g} {self.fg}')
            if neptune_logger is not None:
                neptune_logger[f"test/pso_iteration/{self.it}_best"].log(self.fg)

            # if np.linalg.norm(self.g - self.p[i]) <= self.minstep and np.abs(self.fg - self.fp[i]) <= self.minfunc:
            #     if self.debug:
            #         print(f'Stopping search: Swarm best objective change less than {self.minfunc}')
            #     break

            self.it += 1
            self.collected_data.append(self.data_point)

        if self.debug:
            print(f'Stopping search: maximum iterations reached --> {self.maxiter}')
        return self.g, self.fg
    
    def multiple_run(self, runs=10, plot_particles=False, plot_history=False, fps=10, save_dir="", neptune_logger=None, log_interval=5):
        duration = 0
        for i in range(runs+1):
            self._initialize_swarm()
            start = time.time()
            self.optimize(neptune_logger=neptune_logger)
            end = time.time()
            duration += end - start
            self.global_best_values.append(self.history)
            # if dim is less than 3, plot the particles state
            if len(self.lb) < 3:
                if plot_particles:
                    self.plot_particles_state()
                if plot_history and i % log_interval == 0:
                    _ = self.plot_particles_history(fps=fps, save_dir=f"{save_dir}/pso_run_{i}.gif")
                    if neptune_logger is not None:
                        neptune_logger["test/pso_run"].log(f"{save_dir}/pso_run_{i}.gif")
        return self.global_best_values, duration
    
    def plot_particles_state(self):
        # plot all particles position in the 2D function space: if the function is 2D else raise an error
        if len(self.lb) != 2:
            raise ValueError("The function must be 2D")
        else:
            plt.figure(figsize=(10, 10))
            plt.scatter(self.x[:, 0], self.x[:, 1], marker='o', c='b', s=30, label='particles')
            plt.scatter(self.g[0], self.g[1], marker='*', c='r', s=60, label='global best')
            plt.legend(loc='upper right', fontsize=15)
            plt.xlabel('x1', fontsize=15)
            plt.ylabel('x2', fontsize=15)
            plt.title('Particles position', fontsize=15)
            plt.show()

    def plot_particles_history(self, save_dir=None, fps=10):
        # plot a gif of the particle positions at each time step
        if len(self.lb) != 2:
            raise ValueError("The function must be 2D")
        else:
            x = np.linspace(self.lb[0], self.ub[0], 1000)
            y = np.linspace(self.lb[1], self.ub[1], 1000)
            X, Y = np.meshgrid(x, y)
            Z = self.func.evaluate(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)

            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(xlim=(self.lb[0], self.ub[0]), ylim=(self.lb[1], self.ub[1]))
            
            ax.contour(X, Y, Z, 50)
            ax.scatter(self.g[0], self.g[1], marker='*', c='r', s=60, label='global best')
            ax.set_xlabel('x1', fontsize=15)
            ax.set_ylabel('x2', fontsize=15)
            ax.set_title('Particles position', fontsize=15)

            scat = ax.scatter([], [], c="red", s=100, marker="^", edgecolors="black")
            # add a text box to display the iteration number
            text = ax.text(0.05, 0.95, "", transform=ax.transAxes)

            def animate(i):
                scat.set_offsets(self.pos_history[i])
                text.set_text(f"Iteration: {i}")
                return scat, text
            print(f"Number of iterations: {len(self.pos_history)}")
            anim = animation.FuncAnimation(fig, animate, frames=len(self.pos_history), interval=100, blit=True)
            if save_dir is not None:
                anim.save(save_dir, writer="imagemagick", fps=fps)
            # close the animation figure to avoid displaying it below the cell
            plt.close()
            return anim

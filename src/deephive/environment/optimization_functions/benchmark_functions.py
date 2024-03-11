
import numpy as np

# Sphere Function
def sphere_function(x):
    return np.sum(x**2, axis=1)

def schwefel_1_2(x):
    return np.sum(np.cumsum(x, axis=1)**2, axis=1)

# Schwefel 2.20 Function
def schwefel_2_20(x):
    return np.sum(np.abs(x), axis=1)

# Schwefel 2.21 Function
def schwefel_2_21(x):
    return np.max(np.abs(x), axis=1)

# Schwefel 2.22 Function
def schwefel_2_22(x):
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)

# Schwefel 2.23 Function
def schwefel_2_23(x):
    return np.sum(np.abs(x)**10, axis=1)

# Step Function
def step_function(x):
    return np.sum((x + 0.5)**2, axis=1)

# Rosenbrock Function - Requires special handling for vectorization
def rosenbrock(x):
    return np.sum(100.0 * (x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2, axis=1)

# Sum Squares Function
def sum_squares(x):
    n = np.arange(1, x.shape[1] + 1)
    return np.sum(n * x**2, axis=1)

# Zakharov Function
def zakharov(x):
    n = np.arange(1, x.shape[1] + 1)
    return np.sum(x**2, axis=1) + np.sum(0.5 * n * x, axis=1)**2 + np.sum(0.5 * n * x, axis=1)**4

# Quartic Function
def quartic(x):
    n = np.arange(1, x.shape[1] + 1)
    return np.sum(n * x**4, axis=1)

# Powell Sum Function
def powell_sum(x):
    n = np.arange(1, x.shape[1] + 1)
    return np.sum(np.abs(x)**n, axis=1)

# Dixon-Price Function - Requires loop or vectorization trick for differences
def dixon_price(x):
    i = np.arange(2, x.shape[1] + 1)
    return (x[:, 0] - 1)**2 + np.sum(i * (2 * x[:, 1:]**2 - x[:, :-1])**2, axis=1)

def leon(x):
    return np.sum((x - 1)**2 * (1 + 10 * np.sin(np.pi * x + 1)**2), axis=1)

def booth(x):
    return (x[:, 0] + 2 * x[:, 1] - 7)**2 + (2 * x[:, 0] + x[:, 1] - 5)**2

def matyas(x):
    return 0.26 * sphere_function(x) - 0.48 * x[:, 0] * x[:, 1]

def perm(x):
    n = x.shape[1]
    p = np.arange(1, n + 1)
    return np.sum((np.sum((p + 1) * (x - p)**2, axis=1))**2, axis=1)

def ackley_2(x):
    n = x.shape[1]
    return -200 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / n))

def trid(x):
    n = x.shape[1]
    return np.sum((x - 1)**2, axis=1) - np.sum(x[:, 1:] * x[:, :-1], axis=1)

def Qing(x):
    return np.sum(x**2 - np.arange(1, x.shape[1] + 1)**2, axis=1)

def Alpine(x):
    return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1)

def Griewank(x):
    n = np.arange(1, x.shape[1] + 1)
    return np.sum(x**2 / 4000, axis=1) - np.prod(np.cos(x / np.sqrt(n)), axis=1) + 1

def Salomon(x):
    return 1 - np.cos(2 * np.pi * np.sqrt(np.sum(x**2, axis=1))) + 0.1 * np.sqrt(np.sum(x**2, axis=1))

def Ackley(x):
    n = x.shape[1]
    return -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / n)) - np.exp(np.sum(np.cos(2 * np.pi * x), axis=1) / n) + 20 + np.exp(1)

def Levy(x):
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w[:, 0])**2 + np.sum((w[:, :-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1)**2), axis=1) + (w[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * w[:, -1])**2)

def Powell(x):
    n = x.shape[1]
    return np.sum((x[:, 0::4] + 10 * x[:, 1::4])**2 + 5 * (x[:, 2::4] - x[:, 3::4])**2 + (x[:, 1::4] - 2 * x[:, 2::4])**4 + 10 * (x[:, 0::4] - x[:, 3::4])**4, axis=1)

def Rastrigin(x):
    return 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

def Penalized_1(x):
    n = x.shape[1]
    return 0.1 * (np.sin(3 * np.pi * x[:, 0])**2 + np.sum((x[:, :-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[:, 1:] + 1)**2), axis=1) + (x[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * x[:, -1])**2)) + np.sum((x - 1)**2 * (1 + np.sin(3 * np.pi * x + 1)**2), axis=1)

def Schwefel(x):
    return (-1/x.shape[1]) * np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

def Langermann(x):
    A = np.array([[3, 5, 2, 1, 7], [5, 2, 1, 4, 9], [2, 3, 4, 6, 8], [4, 6, 7, 3, 2]])
    C = np.array([1, 2, 5, 2, 3])
    return -np.sum(C * np.exp(-np.sum((x - A)**2, axis=1) / np.pi) * np.cos(np.pi * np.sum((x - A)**2, axis=1)), axis=1)

def Goldstein_Price(x):
    return (1 + (x[:, 0] + x[:, 1] + 1)**2 * (19 - 14 * x[:, 0] + 3 * x[:, 0]**2 - 14 * x[:, 1] + 6 * x[:, 0] * x[:, 1] + 3 * x[:, 1]**2)) * (30 + (2 * x[:, 0] - 3 * x[:, 1])**2 * (18 - 32 * x[:, 0] + 12 * x[:, 0]**2 + 48 * x[:, 1] - 36 * x[:, 0] * x[:, 1] + 27 * x[:, 1]**2))

def Bartels_Conn(x):
    return np.abs(x[:, 0]**2 + x[:, 1]**2 + x[:, 0] * x[:, 1]) + np.abs(np.sin(x[:, 0])) + np.abs(np.cos(x[:, 1]))

def Levy_13(x):
    w = 1 + (x - 1) / 4
    return np.sin(np.pi * w[:, 0])**2 + np.sum((w[:, :-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1)**2), axis=1) + (w[:, -1] - 1)**2 * (1 + np.sin(2 * np.pi * w[:, -1])**2)

def Himmelblau(x):
    return (x[:, 0]**2 + x[:, 1] - 11)**2 + (x[:, 0] + x[:, 1]**2 - 7)**2

def Egg_crate(x):
    return x[:, 0]**2 + x[:, 1]**2 + 25 * (np.sin(x[:, 0])**2 + np.sin(x[:, 1])**2)

def Three_hump_camel(x):
    return 2 * x[:, 0]**2 - 1.05 * x[:, 0]**4 + x[:, 0]**6 / 6 + x[:, 0] * x[:, 1] + x[:, 1]**2

def Beale(x):
    return (1.5 - x[:, 0] + x[:, 0] * x[:, 1])**2 + (2.25 - x[:, 0] + x[:, 0] * x[:, 1]**2)**2 + (2.625 - x[:, 0] + x[:, 0] * x[:, 1]**3)**2

def Colville(x):
    return 100 * (x[:, 0]**2 - x[:, 1])**2 + (x[:, 0] - 1)**2 + (x[:, 2] - 1)**2 + 90 * (x[:, 2]**2 - x[:, 3])**2 + 10.1 * ((x[:, 1] - 1)**2 + (x[:, 3] - 1)**2) + 19.8 * (x[:, 1] - 1) * (x[:, 3] - 1)

def Power_sum(x):
    n = np.arange(1, x.shape[1] + 1)
    return np.sum(np.sum(x**n - 1, axis=1)**2, axis=1)

def Bohachevsky_1(x):
    return x[:, 0]**2 + 2 * x[:, 1]**2 - 0.3 * np.cos(3 * np.pi * x[:, 0]) - 0.4 * np.cos(4 * np.pi * x[:, 1]) + 0.7

def Bohachevsky_2(x):
    return x[:, 0]**2 + 2 * x[:, 1]**2 - 0.3 * np.cos(3 * np.pi * x[:, 0]) * np.cos(4 * np.pi * x[:, 1]) + 0.3

def Bohachevsky_3(x):
    return x[:, 0]**2 + 2 * x[:, 1]**2 - 0.3 * np.cos(3 * np.pi * x[:, 0] + 4 * np.pi * x[:, 1]) + 0.3

def Schaffer_1(x):
    return 0.5 + (np.sin(x[:, 0]**2 - x[:, 1]**2)**2 - 0.5) / (1 + 0.001 * (x[:, 0]**2 + x[:, 1]**2))**2

def Schaffer_2(x):
    return 0.5 + (np.cos(np.sin(np.abs(x[:, 0]**2 - x[:, 1]**2)))**2 - 0.5) / (1 + 0.001 * (x[:, 0]**2 + x[:, 1]**2))**2

def Schaffer_3(x):
    return 0.5 + (np.sin(np.sin(x[:, 0]**2 - x[:, 1]**2)**2) - 0.5) / (1 + 0.001 * (x[:, 0]**2 + x[:, 1]**2))**2

def Schaffer_4(x):
    return 0.5 + (np.cos(np.sin(np.abs(x[:, 0]**2 - x[:, 1]**2)))**2 - 0.5) / (1 + 0.001 * (x[:, 0]**2 + x[:, 1]**2))**2

def Branin(x):
    return (x[:, 1] - 5.1 / (4 * np.pi**2) * x[:, 0]**2 + 5 / np.pi * x[:, 0] - 6)**2 + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[:, 0]) + 10

def Keane(x):
    return -np.sin(x[:, 0] - x[:, 1])**2 * np.sin(x[:, 0] + x[:, 1]) / np.sqrt(x[:, 0]**2 + x[:, 1]**2)

def Kowalik(x):
    return np.sum((y - (x[0] * (u**2 + v**2) / (u**2 + v**2 + x[1] * u + x[2] * v)))**2 for y, u, v in zip([4.0, 2.0, 1.0, 1.0], [0.1957, 0.1947, 0.1735, 0.16], [0.25, 0.287, 0.248, 0.21]))

def drop_wave(x):
    return -(1 + np.cos(12 * np.sqrt(x[:, 0]**2 + x[:, 1]**2))) / (0.5 * (x[:, 0]**2 + x[:, 1]**2) + 2)

def ackley_3(x):
    return -200 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1))) + 5 * np.exp(np.sum(np.cos(3 * x) + np.sin(3 * x)), axis=1)

def holder_table(x):
    return -np.abs(np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.exp(np.abs(1 - np.sqrt(x[:, 0]**2 + x[:, 1]**2) / np.pi)))

def shubert(x):
    return np.prod(np.sum(np.arange(1, 6) * np.cos((np.arange(1, 6) + 1) * x + np.arange(1, 6)), axis=1))

def shubert_3(x):
    return np.prod(np.sum(np.arange(1, 6) * np.sin((np.arange(1, 6) + 1) * x + np.arange(1, 6)), axis=1))

def shubert_4(x):
    return np.prod(np.sum(np.arange(1, 6) * np.cos((np.arange(1, 6) * x + np.arange(1, 6))**2), axis=1))

def egg_holder(x):
    return -(x[:, 1] + 47) * np.sin(np.sqrt(np.abs(x[:, 1] + x[:, 0] / 2 + 47)) - x[:, 0] * np.sin(np.sqrt(np.abs(x[:, 0] - (x[:, 1] + 47)))))
                                    
def six_hump_camel(x):
    return (4 - 2.1 * x[:, 0]**2 + x[:, 0]**4 / 3) * x[:, 0]**2 + x[:, 0] * x[:, 1] + (-4 + 4 * x[:, 1]**2) * x[:, 1]**2

def bird(x):
    return np.sin(x[:, 0]) * np.exp((1 - np.cos(x[:, 1]))**2) + np.cos(x[:, 1]) * np.exp((1 - np.sin(x[:, 0]))**2) + (x[:, 0] - x[:, 1])**2
    
def adjiman(x):
    return np.cos(x[:, 0]) * np.sin(x[:, 1]) - x[:, 0] / (x[:, 1]**2 + 1) 
    
    


class FunctionSelector:
    def __init__(self):
        self.functions = {
            'f01': {
                'func': sphere_function,
                'dimension': 30,
                'domain': [-100, 100],
                'global_min': 0
            },
            'f02': {
                'func': schwefel_1_2,
                'dimension': 30,
                'domain': [-100, 100],
                'global_min': 0
            },
            'f03': {
                'func': schwefel_2_20,
                'dimension': 30,
                'domain': [-100, 100],
                'global_min': 0
            },
            # Add other functions here following the same structure
            'f04': {
                'func': schwefel_2_21,
                'dimension': 30,
                'domain': [-100, 100],
                'global_min': 0
            },
            'f05': {
                'func': schwefel_2_22,
                'dimension': 30,
                'domain': [-10, 10],
                'global_min': 0
            },
            'f06': {
                'func': schwefel_2_23,
                'dimension': 30,
                'domain': [-100, 100],
                'global_min': 0
            },
            'f07': {
                'func': step_function,
                'dimension': 30,
                'domain': [-10, 10],
                'global_min': 0
            },
            'f08': {
                'func': rosenbrock,
                'dimension': 30,
                'domain': [-30, 30],
                'global_min': 0
            },
            'f09': {
                'func': sum_squares,
                'dimension': 30,
                'domain': [-10, 10],
                'global_min': 0
            },
            'f10': {
                'func': zakharov,
                'dimension': 30,
                'domain': [-5, 10],
                'global_min': 0
            },
            'f11': {
                'func': quartic,
                'dimension': 30,
                'domain': [-1.28, 1.28],
                'global_min': 0
            },
            'f12': {
                'func': powell_sum,
                'dimension': 30,
                'domain': [-1, 1],
                'global_min': 0
            },
            'f13': {
                'func': dixon_price,
                'dimension': 2,
                'domain': [-10, 10],
                'global_min': 0
            },
            'f14': {
                'func': leon,
                'dimension': 2,
                'domain': [0, 10],
                'global_min': 0
            },
            'f15': {
                'func': booth,
                'dimension': 2,
                'domain': [-10, 10],
                'global_min': 0
            },
            'f16': {
                'func': matyas,
                'dimension': 2,
                'domain': [-10, 10],
                'global_min': 0
            },
            'f17': {
                'func': perm,
                'dimension': 2,
                'domain': [2, 2],
                'global_min': 0
            },
            'f18': {
                'func': ackley_2,
                'dimension': 2,
                'domain': [-32, 32],
                'global_min': -200
            },
            'f19': {
                'func': trid,
                'dimension': 10,
                'domain': [-100, 100],
                'global_min': -210
            },
            'f20': {
                'func': Qing,
                'dimension': 30,
                'domain': [-500, 500],
                'global_min': 0
            },
            'f21': {
                'func': Alpine,
                'dimension': 30,
                'domain': [0, 10],
                'global_min': 0
            },
            
            'f22': {
                'func': Griewank,
                'dimension': 30,
                'domain': [-600, 600],
                'global_min': 0
            },
            
            'f23': {
                'func': Salomon,
                'dimension': 30,
                'domain': [-100, 100],
                'global_min': 0
            },
            'f24': {
                'func': Ackley,
                'dimension': 30,
                'domain': [-32, 32],
                'global_min': 0
            },
            'f25': {
                'func': Levy,
                'dimension': 30,
                'domain': [-10, 10],
                'global_min': 0
            },
            'f26': {
                'func': Powell,
                'dimension': 30,
                'domain': [-4, 5],
                'global_min': 0
            },
            'f27': {
                'func': Rastrigin,
                'dimension': 30,
                'domain': [-5.12, 5.12],
                'global_min': 0
            },
            'f28': {
                'func': Penalized_1,
                'dimension': 30,
                'domain': [-50, 50],
                'global_min': 0
            },
            'f29': {
                'func': Schwefel,
                'dimension': 30,
                'domain': [-500, 500],
                'global_min': -12569.5
            },
            'f30': {
                'func': Langermann,
                'dimension': 30,
                'domain': [0, 10],
                'global_min': -4.155809
            },
             
            'f31': {
                'func': Goldstein_Price,
                'dimension': 2,
                'domain': [-2, 2],
                'global_min': 3
            },
            'f32': {
                'func': Bartels_Conn,
                'dimension': 2,
                'domain': [-500, 500],
                'global_min': 1
            },
            'f33': {
                'func': Levy_13,
                'dimension': 2,
                'domain': [-10, 10],
                'global_min': 0
            },
            'f34': {
                'func': Himmelblau,
                'dimension': 2,
                'domain': [-6, 6],
                'global_min': 0
            },
             
            'f35': {
                'func': Egg_crate,
                'dimension': 2,
                'domain': [-5, 5],
                'global_min': 0
            },
            'f36': {
                'func': Three_hump_camel,
                'dimension': 2,
                'domain': [-5, 5],
                'global_min': 0
            },
            'f37': {
                'func': Beale,
                'dimension': 2,
                'domain': [-4.5, 4.5],
                'global_min': 0
            },
            'f38': {
                'func': Colville,
                'dimension': 4,
                'domain': [-10, 10],
                'global_min': 0
            },
            'f39': {
                'func': Power_sum,
                'dimension': 4,
                'domain': [0, 4],
                'global_min': 0
            },
            'f40': {
                'func': Bohachevsky_1,
                'dimension': 2,
                'domain': [-100, 100],
                'global_min': 0
            },
            'f41': {
                'func': Bohachevsky_2,
                'dimension': 2,
                'domain': [-100, 100],
                'global_min': 0
            },
            'f42': {
                'func': Bohachevsky_3,
                'dimension': 2,
                'domain': [-100, 100],
                'global_min': 0
            },
            'f43': {
                'func': Schaffer_1,
                'dimension': 2,
                'domain': [-100, 100],
                'global_min': 0
            },
            'f44': {
                'func': Schaffer_2,
                'dimension': 2,
                'domain': [-100, 100],
                'global_min': 0
            },
            'f45': {
                'func': Schaffer_3,
                'dimension': 2,
                'domain': [-100, 100],
                'global_min': 0.0016
            },
            'f46': {
                'func': Schaffer_4,
                'dimension': 2,
                'domain': [-100, 100],
                'global_min': 0.292579
            },
            'f47': {
                'func': Branin,
                'dimension': 2,
                'domain': [-5, 5],
                'global_min': 0.397887
            },
            'f48': {
                'func': Keane,
                'dimension': 2,
                'domain': [-0, 10],
                'global_min': -0.6737
            },
            'f49': {
                'func': Kowalik,
                'dimension': 4,
                'domain': [-5, 5],
                'global_min': 0.0003075
            },
            'f50': {
                'func': drop_wave,
                'dimension': 2,
                'domain': [-5.12, 5.12],
                'global_min': -1
            }
            
        }

    
    def get_function(self, identifier):
        if identifier in self.functions:
            func_info = self.functions[identifier]
            return func_info
        else:
            raise ValueError("Function identifier not found.")

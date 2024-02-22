# cec2017.utils
# Author: Duncan Tilley
# Additional functions for graphing and benchmarking

def surface_plot(function, domain=(-100,100), points=30, dimension=2, ax=None):
    """
    Creates a surface plot of a function.

    Args:
        function (function): The objective function to be called at each point.
        domain (num, num): The inclusive (min, max) domain for each dimension.
        points (int): The number of points to collect on each dimension. A total
            of points^2 function evaluations will be performed.
        dimension (int): The dimension to pass to the function. If this is more
            than 2, the elements after the first 2 will simply be zero,
            providing a slice at x_3 = 0, ..., x_n = 0.
        ax (matplotlib axes): Optional axes to use (must have projection='3d').
            Note, if specified plt.show() will not be called.
    """
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import numpy as np

    # create points^2 tuples of (x,y) and populate z
    xys = np.linspace(domain[0], domain[1], points)
    xys = np.transpose([np.tile(xys, len(xys)), np.repeat(xys, len(xys))])
    # zs = np.zeros(points*points)

    if dimension > 2:
        # concatenate remaining zeros
        tail = np.zeros((xys.shape[0], dimension - 2))
        x = np.concatenate([xys, tail], axis=1)
        zs = function(x)
        # for i in range(0, xys.shape[0]):
        #     zs[i] = function(np.concatenate([xys[i], tail]))
    else:
        zs = function(xys)
        # for i in range(0, xys.shape[0]):
        #     zs[i] = function(xys[i])

    # create the plot
    ax_in = ax
    if ax is None:
        ax = plt.axes(projection='3d')

    X = xys[:,0].reshape((points, points))
    Y = xys[:,1].reshape((points, points))
    Z = zs.reshape((points, points))
    ax.plot_surface(X, Y, Z, cmap='gist_ncar', edgecolor='none')
    ax.set_title(function.__name__)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if ax_in is None:
        plt.show()
        
        
def contour_plot_1(function, domain=(-100,100), points=30, ax=None, dimension=2):
    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    import numpy as np
    # create points^2 tuples of (x,y) and populate z
    xys = np.linspace(domain[0], domain[1], points)
    xys = np.transpose([np.tile(xys, len(xys)), np.repeat(xys, len(xys))])

    if dimension > 2:
        # concatenate remaining zeros
        tail = np.zeros((xys.shape[0], dimension - 2))
        x = np.concatenate([xys, tail], axis=1)
        zs = function(x)
    else:
        zs = function(xys)

    fig = plt.figure()
    ax = fig.gca()

    X = xys[:,0].reshape((points, points))
    Y = xys[:,1].reshape((points, points))
    Z = zs.reshape((points, points))
    cont = ax.contourf(X, Y, Z, lw = 1, levels=20, cmap='plasma')
    ax.contour(X, Y, Z, colors="k", linestyles="solid")
    cbar = fig.colorbar(cont, shrink=0.5, aspect=5, pad=0.15, label='')
#     cbar = fig.colorbar(ax.collections[0], shrink=0.5, aspect=5, pad=0.15, label='')

    ax.set_title(function.__name__)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # Set the background color
    ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
    plt.show()
    
    return fig
    

def time(function, domain=(-100,100), points=30):
    """
    Returns the time in seconds to calculate points^2 evaluations of the
    given function.

    function
        The objective function to be called at each point.
    domain
        The inclusive (min, max) domain for each dimension.
    points
        The number of points to collect on each dimension. A total of points^2
        function evaluations will be performed.
    """
    from time import time
    import numpy as np

    # create points^2 tuples of (x,y) and populate z
    xys = np.linspace(domain[0], domain[1], points)
    xys = np.transpose([np.tile(xys, len(xys)), np.repeat(xys, len(xys))])
    zs = np.zeros(points*points)

    before = time()
    for i in range(0, xys.shape[0]):
        zs[i] = function(xys[i])
    return time() - before

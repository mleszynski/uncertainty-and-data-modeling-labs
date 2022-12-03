# iPyParallel - Intro to Parallel Programming
from ipyparallel import Client
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from numpy.core.defchararray import title

# Problem 1
def initialize():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as spar on
    all engines. Return the DirectView.
    """
    # prepare client, create and return directview #############################
    client = Client()
    dview = client[:]
    #print(dview.targets)
    dview.execute("import scipy.sparse as sparse")
    return dview
    

# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    # create a dview ###########################################################
    dview = initialize()
    dview.block = True

    # push variables ###########################################################
    #print("initial values: ")
    #print(dx)

    dview.push(dx)
    #print("values in engines: ")
    #for key in dx:
    #    print(dview.pull(key))


# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    # create a dview ###########################################################
    dview = initialize()
    dview.block = True

    # initialize variables in engines ##########################################
    dview.execute("import numpy as np")
    dview.push({'n_val':n})

    # sample normal distribution and calculate means, mins, maxs ###############
    dview.execute("draws = np.random.normal(size = n_val)")
    dview.execute("means = np.mean(draws)")
    dview.execute("mins = draws.min()")
    dview.execute("maxs = draws.max()")

    # pull data from engines and return wanted values ##########################
    means = dview.pull('means')
    mins = dview.pull('mins')
    maxs = dview.pull('maxs')
    #print(means)
    #print(mins)
    #print(maxs)

    return means, mins, maxs


# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    # create helper function to perform serial draws ###########################
    def serial_draws(n):
        means = []
        mins = []
        maxs = []
        for i in range(8):
            draws = np.random.normal(size = n)
            means.append(np.mean(draws))
            mins.append(draws.min())
            maxs.append(draws.max())

        return means, mins, maxs
    
    # initialize necessary values ##############################################
    lengths = [1000000, 5000000, 10000000, 15000000]
    parallel_times = []
    serial_times = []

    # perform and time draws ###################################################
    for n in lengths:
        # time parallel ########################################################
        p_start = timer()
        prob3(n)
        p_end = timer()
        parallel_times.append(p_end - p_start)

        # time serial ##########################################################
        s_start = timer()
        serial_draws(n)
        s_end = timer()
        serial_times.append(s_end - s_start)

    # plot results #############################################################
    plt.plot(lengths, parallel_times, color = 'blue', label='Parallel')
    plt.plot(lengths, serial_times, color='red', label='Serial')
    plt.ylabel('time (seconds)')
    plt.xlabel('n draws')
    plt.title('Parallel vs Serial Calculation Times')
    plt.legend()
    plt.show()


# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    # initialize dview and necessary variables #################################
    dview = initialize()
    dview.block = True
    h = (b-a)/n
    domain = np.arange(a, b, h)

    # scatter domain to have each core work on a different part of integral ####
    dview.scatter('domain', domain)

    # define a helper function that calculates the trapezoid rule area #########
    def get_area(f, h, domain):
        area = 0
        for i in range(len(domain) - 1):
            area = area + (h/2)*(f(domain[i])+f(domain[i+1]))
        return area

    # push necessary info to engines, perform calculation, return result #######
    dview.push({'f':f, 'h':h, 'get_area':get_area})
    dview.execute("area = get_height(f, h, domain)")
    return sum(dview.pull('area'))


# for debugging purposes #######################################################
#if __name__ == "__main__":
    ############################################################################
    # prob 1 ###################################################################
    ############################################################################

    #initialize()

    ############################################################################
    # prob 2 ###################################################################
    ############################################################################
    
    #dx = {'a':10, 'b':5, 'c':1}
    #variables(dx)

    ############################################################################
    # prob 3 ###################################################################
    ############################################################################

    #means, mins, maxs = prob3()
    #print(means)
    #print(mins)
    #print(maxs)

    ############################################################################
    # prob 4 ###################################################################
    ############################################################################

    #prob4()

    ############################################################################
    # prob 5 ###################################################################
    ############################################################################

    #f_1 = lambda x: 1/2
    #print(parallel_trapezoidal_rule(f_1, 0, 1, 200))

    #f_2 = lambda x: 2*x + 5
    #print(parallel_trapezoidal_rule(f_2, 0, 50, 300))
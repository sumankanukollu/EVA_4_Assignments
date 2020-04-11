import numpy as np
import matplotlib.pyplot as plt

def plotCyclicLR(num_cycles,total_iterations,min_lr,max_lr,step_size):
    l_rate=[]
    cycle = 0
    for iteration in range(total_iterations):
        
        if cycle<=num_cycles:
            ti=iteration
            cycle = np.floor(1 + iteration / (2 * step_size))
            x = np.abs((iteration / step_size) - 2 * cycle + 1)
            lr = min_lr + ((max_lr - min_lr) * (1 - x))
            l_rate.append(lr)
        else:
            break
    print('Number of Cycles : {}'.format(cycle))
    #plt.plot(list(range(total_iterations)),l_rate)
    plt.plot(list(range(ti+1)),l_rate)
    
    
plotCyclicLR(num_cycles=6,total_iterations=10000, min_lr=0.1, max_lr=0.5, step_size=1000)

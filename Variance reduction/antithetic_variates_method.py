import matplotlib.pyplot as plt
import numpy as np
from time import time
import pandas as pd

# Underlying informations
S0 = 100
mu = 0.1
sigma = 0.1

# European option informations
T = 1.0
K = 100.0
r = 0.05

# Simulation parameters
nbr_steps = 100
dt = T/nbr_steps
t = np.linspace(0, T, nbr_steps)
min_nbr_sim, max_nbr_sim = 100, 100000
nbr_steps_sims = 100
nbr_sims = np.linspace(min_nbr_sim, max_nbr_sim, nbr_steps_sims)

# Global variables for results storage
prices_standard = []
prices_antithetic = []

# Classic Monte-Carlo simulation
time_begin_classic = time()

for i,nbr_sim in enumerate(nbr_sims):
    print(i)
    nbr_sim = int(nbr_sim)
    price = 0.0
    for _ in range(nbr_sim):
        W = np.random.standard_normal(size = nbr_steps)
        W = np.cumsum(W)*np.sqrt(dt)
        X = (mu-0.5*sigma**2)*t + sigma*W
        S = S0*np.exp(X)

        # Payoff computation of a european call
        if(S[-1]>K):
            price += S[-1]-K
    prices_standard.append((price/nbr_sim)*np.exp(-r*T))

calculation_time_classic = round(time()-time_begin_classic, 1)

# Antithetic variates method
time_begin_antithetic = time()
half_len = int(len(nbr_sims)/2)

for i,nbr_sim in enumerate(nbr_sims[0:half_len]):
    print(i)
    price = 0.0
    nbr_sim = int(nbr_sim)
    for _ in range(nbr_sim):
        W = np.random.standard_normal(size = nbr_steps)
        W_2 = -W
        W = np.cumsum(W)*np.sqrt(dt)
        W_2 = np.cumsum(W_2)*np.sqrt(dt)
        X = (mu-0.5*sigma**2)*t + sigma*W
        X_2 = (mu-0.5*sigma**2)*t + sigma*W_2
        S = S0*np.exp(X)
        S_2 = S0*np.exp(X_2)

        # Computation of both payoffs and then the mean
        if(S[-1]>K):
            price += S[-1]-K
        if(S_2[-1]>K):
            price += S_2[-1]-K
    prices_antithetic.append((price/(2*nbr_sim))*np.exp(-r*T))

calculation_time_antithetic = round(time()-time_begin_antithetic, 1)

# Computing mean and standard deviation
prices = np.array(prices_standard)
mean_val = np.mean(prices)
std_val = round(np.std(prices),4)

# Plotting classical Monte-Carlo simulation
plt.figure(1)
ax1 = plt.subplot(211)
ax1.set_title("Classic method (time of execution : {}s)".format(calculation_time_classic))
ax1.plot(nbr_sims, prices, label='Price')
ax1.plot(nbr_sims, np.linspace(mean_val, mean_val,100), label='Mean')
ax1.plot(nbr_sims, np.linspace(mean_val-std_val, mean_val-std_val,100),'g', label='Mean-std')
ax1.plot(nbr_sims, np.linspace(mean_val+std_val, mean_val+std_val,100),'g', label='Mean+std')
ax1.legend(loc="upper right")
ax1.set_xlabel('number of simulations')
ax1.set_ylabel('price ($)')

# Computing mean and standard deviation
prices = np.array(prices_antithetic)
mean_val = np.mean(prices)
std_val = round(np.std(prices),3)

# Plotting with the antithetic variates method
ax2 = plt.subplot(212, sharex=ax1, sharey=ax1)
ax2.set_title("Antithetic varaites method (time of execution : {}s)".format(calculation_time_antithetic))
ax2.plot(nbr_sims[0:half_len], prices, label='Price')
ax2.plot(nbr_sims[0:half_len], np.linspace(mean_val, mean_val,50), label='Mean')
ax2.plot(nbr_sims[0:half_len], np.linspace(mean_val-std_val, mean_val-std_val,50),'g', label='Mean-std')
ax2.plot(nbr_sims[0:half_len], np.linspace(mean_val+std_val, mean_val+std_val,50),'g', label='Mean+std')
ax2.legend(loc="upper right")
ax2.set_xlabel('number of simulations')
ax2.set_ylabel('price ($)')
plt.tight_layout()
plt.show()

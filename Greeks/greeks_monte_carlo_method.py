import matplotlib.pyplot as plt
import numpy as np
from time import time
import pandas as pd
from scipy.stats import norm

# Underlying informations
S0 = 100.0
sigma = 0.2

# European option informations
T = 1.0
K = 100.0
r = 0.05

# Simulation parameters
nbr_steps = 100
dt = T/nbr_steps
t = np.linspace(0, T, nbr_steps)
nbr_sim = 1000000

# parameters for greek calculation
seed = 2
dS = 1/S0
d_sigma = sigma/100

# European call price and greeks according to Black-Scholes

def d1():
    return (np.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))

def d2():
    return d1() - sigma*np.sqrt(T)

def price_BS(S0):
    return S0*norm.cdf(d1())-K*np.exp(-r*T)*norm.cdf(d2())

def delta_BS():
    return norm.cdf(d1())

def gamma_BS():
    return norm.pdf(d1())/(S0*sigma*np.sqrt(T))

def vega_BS():
    return S0*np.sqrt(T)*norm.pdf(d1())

# Monte-Carlo pricing and greeks

def price_MC(S0, sigma):
    # Setting the seed in order to get the same results
    np.random.seed(seed)

    price = 0.0
    for _ in range(nbr_sim):
        W = np.random.standard_normal(size = nbr_steps)
        W = np.cumsum(W)*np.sqrt(dt)
        X = (r-0.5*sigma**2)*t + sigma*W
        S = S0*np.exp(X)

        # Payoff computation of a european call
        if(S[-1]>K):
            price += S[-1]-K
    return (price/nbr_sim)*np.exp(-r*T)

def delta_MC(dS):
    p_S = price_MC(S0, sigma)
    p_S_dS = price_MC(S0+dS, sigma)
    return (p_S_dS - p_S)/dS

def gamma_MC(dS):
    p_m_dS = price_MC(S0-dS, sigma)
    p_S = price_MC(S0, sigma)
    p_S_dS = price_MC(S0+dS, sigma)
    return (p_m_dS - 2*p_S + p_S_dS)/dS**2

def vega_MC(d_sigma):
    p_sigma = price_MC(S0, sigma)
    p_d_sigma = price_MC(S0, sigma+d_sigma)
    return (p_d_sigma - p_sigma)/d_sigma

# Testing
delta_bs, delta_mc = delta_BS(), delta_MC(dS)
print('Delta : \nTheorical value : {} ; Monte-Carlo value : {} ; Error : {} %'
    .format(delta_bs, delta_mc, 100*np.round(np.abs(delta_mc - delta_bs)/delta_bs, 5)))
gamma_bs, gamma_mc = gamma_BS(), gamma_MC(dS)
print('Gamma : \nTheorical value : {} ; Monte-Carlo value : {} ; Error : {} %'
    .format(gamma_bs, gamma_mc, 100*np.round(np.abs(gamma_mc - gamma_bs)/gamma_bs, 5)))
vega_bs, vega_mc = vega_BS(), vega_MC(dS)
print('Vega : \nTheorical value : {} ; Monte-Carlo value : {} ; Error : {} %'
    .format(vega_bs, vega_mc, 100*np.round(np.abs(vega_mc - vega_bs)/vega_bs, 5)))

input('Press enter to continue...')

import numpy as np
from scipy.stats import norm

# Underlying informations
S0 = 100.0
sigma = 0.2

# European option informations
T = 1.0
K = 100.0
r = 0.05

# Binomial tree parameters
nbr_steps = 1000
dt = T/nbr_steps
u = np.exp(sigma*np.sqrt(dt))
d = 1/u
p = (np.exp(r*dt)-d)/(u-d)

# parameters for greek calculation
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

def theta_BS():
    return -S0*norm.pdf(d1())*sigma/(2*np.sqrt(T))-r*K*np.exp(-r*T)*norm.cdf(d2())

def vega_BS():
    return S0*np.sqrt(T)*norm.pdf(d1())

# European call price and greeks according to the binomial tree

def binom_tree(sigma):
    u = np.exp(sigma*np.sqrt(dt))
    d = 1/u
    p = (np.exp(r*dt)-d)/(u-d)

    steps = []
    steps.append([[S0, 0.0]])

    # Generating prices
    for i in range(2, nbr_steps+2):
        step = []
        for j in reversed(range(i)):
            step.append([S0*(u**(j+1))*d**(i-j), 0.0])
        steps.append(step)

    # Reverse induction payoff
    for j in range(nbr_steps+1):
        steps[nbr_steps][j][1] = max(0.0, steps[nbr_steps][j][0]-K)

    for i in reversed(range(nbr_steps)):
        for j in range(i+1):
            steps[i][j][1] = np.exp(-r*dt)*(p*steps[i+1][j][1]
                                            + (1-p)*steps[i+1][j+1][1])

    return steps

def price_binom():
    return binom_tree(sigma)[0][0][1]

def delta_binom():
    tree = binom_tree(sigma)
    return (tree[1][0][1]-tree[1][1][1])/(S0*(u-d))

def gamma_binom():
    tree = binom_tree(sigma)
    return ((tree[2][0][1] - tree[2][1][1])/(S0*(u**2-1))
            - (tree[2][1][1] - tree[2][2][1])/(S0*(1-d**2)))/(0.5*S0*(u**2-d**2))

def theta_binom():
    tree = binom_tree(sigma)
    return (tree[2][1][1] - tree[0][0][1])/(2*dt)

def vega_binom(d_sigma):
    tree = binom_tree(sigma)
    tree_d_sigma = binom_tree(sigma+d_sigma)
    return (tree_d_sigma[0][0][1] - tree[0][0][1])/d_sigma

# Testing
delta_bs, delta_mc = delta_BS(), delta_binom()
print('Delta : \nTheorical value : {} ; Binomial tree value : {} ; Error : {} %'
    .format(delta_bs, delta_mc, 100*np.round(np.abs((delta_mc - delta_bs)/delta_bs), 5)))
gamma_bs, gamma_mc = gamma_BS(), gamma_binom()
print('Gamma : \nTheorical value : {} ; Binomial tree value : {} ; Error : {} %'
    .format(gamma_bs, gamma_mc, 100*np.round(np.abs((gamma_mc - gamma_bs)/gamma_bs), 5)))
theta_bs, theta_mc = theta_BS(), theta_binom()
print('Theta : \nTheorical value : {} ; Binomial tree value : {} ; Error : {} %'
    .format(theta_bs, theta_mc, 100*np.round(np.abs((theta_mc - theta_bs)/theta_bs), 5)))
vega_bs, vega_mc = vega_BS(), vega_binom(d_sigma)
print('Vega : \nTheorical value : {} ; Binomial tree value : {} ; Error : {} %'
    .format(vega_bs, vega_mc, 100*np.round(np.abs((vega_mc - vega_bs)/vega_bs), 5)))

input('Press enter to continue...')

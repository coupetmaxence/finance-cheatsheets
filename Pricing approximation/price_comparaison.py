import numpy as np
from scipy.stats import norm

# Underlying informations
S0 = 100.0
sigma = 0.2

# European option informations
T = 1.0
K = 110.0
r = 0.1

def d1():
    return (np.log(S0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))

def d2():
    return d1() - sigma*np.sqrt(T)

def price_BS():
    return S0*norm.cdf(d1())-K*np.exp(-r*T)*norm.cdf(d2())

def price_approx():
    return 0.4*S0*sigma*np.sqrt(T)

print(price_BS())
print(price_approx())
print((price_approx()-price_BS())/price_BS())

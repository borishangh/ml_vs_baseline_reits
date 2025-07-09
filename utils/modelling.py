import numpy as np
def gbm(S0, mu, sigma, N, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    dt = 1/N
    z = np.random.normal(0, 1, N)
    returns = np.exp((mu - 0.5 * sigma**2) * dt + (sigma * np.sqrt(dt) * z))
    
    prices = S0 * np.cumprod(returns)
    return prices
import numpy as np
from scipy import stats
def gbm(S0, mu, sigma, N, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    dt = 1/N
    z = np.random.normal(0, 1, N)
    returns = np.exp((mu - 0.5 * sigma**2) * dt + (sigma * np.sqrt(dt) * z))
    
    prices = S0 * np.cumprod(returns)
    return prices

def gbm_nl(S0, mu, sigma,N,  normal_weight = 0.7, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    dt = 1/N
    z = (stats.laplace.rvs(0, 1, N) * (1 - normal_weight) 
         + stats.norm.rvs(0, 1, N) * normal_weight)
    z = z / np.sqrt(normal_weight**2 + 2 * (1 - normal_weight)**2)
    returns = np.exp((mu - 0.5 * sigma**2) * dt + (sigma * np.sqrt(dt) * z))
    
    prices = S0 * np.cumprod(returns)
    return prices

def get_jump_params(returns, threshold = 4):
    mu, sigma = returns.mean(), returns.std()
    z = (returns - mu) / sigma
    
    jumps = np.abs(z) > threshold
    l = jumps.mean()
    jump_data = returns[jumps]
    
    if len(jump_data):
        mu_j, sigma_j = jump_data.mean(), jump_data.std()
    else:
        mu_j, sigma_j = 0, 0.001
    return (l, mu_j, sigma_j) 

def jump_diffusion(S0, mu, sigma, N, jump_params=None, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    dt = 1/N
    prices = np.zeros(N+1)
    prices[0] = S0
    
    z = np.random.normal(0, 1, N)
    cont_returns = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    if jump_params:
        lmbd, mu_j, sigma_j = jump_params
        jumps = stats.poisson.rvs(lmbd * dt, size=N)
        
        jump_sizes = np.ones(N)
        for i in range(N):
            if jumps[i] > 0:
                log_jump = np.random.normal(mu_j, sigma_j, jumps[i])
                jump_sizes[i] = np.exp(np.sum(log_jump))
        
        total_returns = cont_returns * jump_sizes
    else:
        total_returns = cont_returns
    
    prices[1:] = S0 * np.cumprod(total_returns)
    return prices
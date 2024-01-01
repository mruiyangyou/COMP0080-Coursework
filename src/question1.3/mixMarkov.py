import pandas as pd 
import numpy as np

def read_data(path: str) -> np.array:
    data = pd.read_csv(path, header=None, sep = ' ', dtype=np.int32)
    return data.values



def condp(x):
    return x / np.sum(x, axis=0)

def logsumexp(x, axis=None):
    xmax = np.max(x, axis=axis, keepdims=True)
    xmax[xmax == -np.inf] = 0
    return xmax + np.log(np.sum(np.exp(x - xmax), axis=axis))

def mix_markov(v, V, H, opts):
    # Initialization
    ph = condp(np.random.rand(H, 1))
    pv1gh = condp(np.random.rand(V, H))
    pvgvh = condp(np.random.rand(V, V, H))

    llik = []
    ph_old = []

    for emloop in range(opts['maxit']):
        ph_stat = np.zeros((H, 1))
        pv1gh_stat = np.zeros((V, H))
        pvgvh_stat = np.zeros((V, V, H))
        loglik = 0

        for n, seq in enumerate(v):
            T = len(seq)
            lph_old = np.log(ph) + np.log(pv1gh[seq[0], :]).reshape(-1, 1)
            for t in range(1, T):
                lph_old = lph_old + np.log(pvgvh[seq[t], seq[t - 1], :]).reshape(-1,1)
            ph_old.append(condp(np.exp(lph_old)))
            loglik += logsumexp(lph_old)

            # M-step statistics
            ph_stat += ph_old[-1]
            pv1gh_stat[seq[0], :] += ph_old[-1].reshape(3)
            for t in range(1, T):
                pvgvh_stat[seq[t], seq[t - 1], :] += ph_old[-1].reshape(3)

        llik.append(loglik)
   
        # M-step
        ph = condp(ph_stat)
        pv1gh = condp(pv1gh_stat)
        pvgvh = condp(pvgvh_stat)

    loglikelihood = llik[-1]
    return ph, pv1gh, pvgvh, loglikelihood, ph_old

# Example usage
# opts = {'maxit': 50, 'plotprogress': 1}
# v = [[1, 2, 3, 4], [2, 3, 4, 1]]  # Example sequences
# V = 4  # Number of visible states
# H = 2  # Number of mixture components

# ph, pv1gh, pvgvh, loglikelihood, phgv = mix_markov(v, V, H, opts)

if __name__ == '__main__':
    data = read_data('data/meteo1.csv')
    
    ph, pv1gh, pvgvh, loglikelihood, phgv = mix_markov(data, 3, 3, {'maxit': 50})
    
    print(ph)
    print(pv1gh)
    print(pvgvh) 
    print(loglikelihood)
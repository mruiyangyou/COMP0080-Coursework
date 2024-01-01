import pandas as pd
import numpy as np 

def read_data(path: str) -> np.array:
    data = pd.read_csv(path, header=None, sep = ' ', dtype=np.int32)
    return data.values

def compute_sequence_likelihood(data, params):
    h, intial, t_matrix = params
    prob_v = np.zeros((len(data), 3))
    for v, i in enumerate(data.reshape(len(data), -1)):
        for j in range(3): # number of hidden state
            res = intial[j, i[0]]
            for z in range(1, 100):
                if res == 0:
                    prob_v[v, j] = 0
                    continue 
                idx_t, idx_t2 = i[z-1], i[z]
                res *= t_matrix[j, idx_t, idx_t2]
            prob_v[v, j] = res 
    return prob_v
    
def compute_likihood(params, prob_v):
    prob_seq = prob_v @ params[0]
    return np.sum(np.log(prob_seq))

def e_step(params, prob_v):
    prob_posterior = prob_v * params[0]
    return prob_posterior / np.sum(prob_posterior, axis = 1).reshape(len(prob_posterior), -1)

def m_step(data,  posterior):
    
    # update for hidden state
    h = np.mean(posterior, axis = 0)
    
    # update for transition matrix
    t_matrix_cnt = np.zeros((3, 3, 3))
    
    for k in range(3):
        for n in range(len(data)):
            for t in range(99):
                i = data[n, t]
                j = data[n, t+1]
                t_matrix_cnt[k, i,j ] += posterior[n, k]
                
    
    t_matrix = np.zeros((3, 3, 3))
    for k in range(3):
        for i in range(3):
            sum_i = np.sum(t_matrix_cnt[k, i, :])
            if sum_i > 0:
                t_matrix[k, i, :] = t_matrix_cnt[k, i, :] / sum_i 
            # else:
            #     t_matrix[k, i, :] = 1.0 / 3
                
    # update for transition probability
    intial = np.zeros((3,3))
    for k in range(3):
        for v1 in range(3):
            weighted_initial_state = sum(
                r_nk * (data[n, 0] == v1) for n, r_nk in enumerate(posterior[:, k])
            )
            intial[k, v1] = weighted_initial_state / np.sum(posterior[:, k])
            
    return h, intial, t_matrix  
    

def EM_Mixture_Markov(data, num_of_iteration):
    i = 1
    likelihood = [float('inf'), -float('inf')]
    np.random.seed(6)
    
    # Intialize parameters for hidden, inital prob, transition matrix
    h = np.random.rand(3) 
    h /= np.sum(h)
    intial = np.random.rand(3, 3)
    intial /= np.sum(intial, axis = 1).reshape(-1, 1)
    t_matrix = np.zeros((3,3,3))
    for i in range(len(t_matrix)):
        t_matrix_i = np.random.rand(3, 3)
        t_matrix_i /= np.sum(t_matrix_i, axis = 1).reshape(-1, 1)
        t_matrix[i] = t_matrix_i 
        
    # h = np.ones(3) / 3 
    # intial = np.ones((3,3)) / 3
    # t_matrix = np.ones((3,3,3)) / 3

    params = h, intial, t_matrix 
   
    while i <= num_of_iteration:
        prob_v = compute_sequence_likelihood(data, params)
        log_likelihood = compute_likihood(params, prob_v)
        likelihood.append(log_likelihood)
        print(f'Iteration: {i}  The loglikelihood: {log_likelihood: .3f}')
        posetrior = e_step(params, prob_v)
        params = m_step(data, posetrior)
        i += 1
        
    return params 

    

        
if __name__ == '__main__':
    num_of_iteration = 20
    data = read_data('data/meteo1.csv')
    
    mle_params = EM_Mixture_Markov(data, num_of_iteration)
    
    for name, i in zip(['Hidden State Prob:', 'Initial State Prob Given Hidden State:',
                        'Transition Matrix Given Hidden State:'],
                        mle_params):
        print(name,'\n', np.round(i, 3))
    
    # test for first 10 rows
    # prob_v = compute_sequence_likelihood(data, mle_params)
    # posterior = e_step(mle_params, prob_v)
    # posterior_df = pd.DataFrame(np.round(posterior[:10],3), columns=['1', '2', '3'])
    # print('The posterior probability for first 10 samples: \n',
    #       posterior_df)
    
    # print(posterior_df.to_latex())
import numpy as np
from typing import List

def enocding(H: np.ndarray) -> (np.ndarray, np.ndarray):
    m, n = H.shape
    k = n - m
    
    # perforn gaussian elimination
    h_2 = np.copy(H)
    for c in range(k):
        # check lead 1 in each row
        rows = np.where(h_2[:, c])[0]
        row = rows[rows >= c][0]
        
        if row != c:
            h_2[[c, row]] = h_2[[row, c]]
        
        for j in range(m):
            if j != c and h_2[j, c] == 1:
                h_2[j] ^= h_2[c]  
        
    p = h_2[:, m:]
    G = np.concatenate((p, np.eye(k, dtype=int)), axis = 0)
    return h_2, G


def read_y(path: str) -> List[int]:
    
    with open(path, 'r')as f:
        res = [int(line.strip()) for line in f.readlines()] 
    return res
        
def read_h(path: str) -> List[List[int]]:
    
    with open(path, 'r') as f:
        res = [list(map(int, line.strip().split(' '))) for line in f.readlines()]
    return res


def decoding(H: np.ndarray, y: np.ndarray, 
             p: float = 0.1, max_iter: int = 20) -> (int, int, np.ndarray):
    # decide size of check matrix
    m,n = H.shape
    
    # find message nodes for each check node, {v: message node, c: check nodes}
    edges = []
    for i in range(m):
        edges.append(np.where(H[i]==1)[0])
    
    # if x_n = 1, y_n|x_n=1
    P_yx1 = np.zeros(len(y))
    #if x_n = 0, y_n|x_n = 0
    P_yx0 = np.zeros(len(y))
    
    # BSC probablity
    for i in range(len(y)):
        P_yx1[i] = p**((y[i]+1)%2) * (1-p)**((y[i]+1+1)%2)
        P_yx0[i] = p**(y[i]) * (1-p)**((y[i]+1)%2)
        
    # Intialize v to c matrix and c to v matrix
    M_v_to_c = np.array([np.log(P_yx0 / P_yx1) for _ in range(m)])
    M_c_to_v  = np.zeros((m,n))
    
    for iter in range(max_iter):
        
        # update for check nodes to variable nodes
        for i, vs in enumerate(edges):
            new_cv= np.zeros(n)
            cv_val = M_v_to_c[i, (vs)]
            
            for j in range(len(cv_val)):
                # find the value of adjanct variables except it self
                cv_val_edge = np.concatenate((cv_val[:j], cv_val[j+1:len(cv_val)]))
                cv_val_edge_product = np.prod(np.tanh(cv_val_edge/2))
                new_cv[vs[j]] = np.log((1 + cv_val_edge_product)/(1-cv_val_edge_product))
            
            # update each row
            M_c_to_v[i, :] = new_cv
            
        # sum over all check nodes to obtain posterior and likelihood
        posterior = np.sum(M_c_to_v, axis = 0) + np.log(P_yx0/P_yx1)
        
        # update for variable nodes to check nodes
        for i in range(n):
            for j in range(m):
                # posterior include itself, so subtract it 
                M_v_to_c[j, i] = posterior[i] - M_c_to_v[j,i]
    
        # make decision
        res = np.array([0 if p > 0 else 1 for p in posterior])
        check = (H @ res) % 2
        if np.all(check == 0):
            return 0, iter+1, res
        
    return -1, 20, res


            
            
                
    
            
            
    
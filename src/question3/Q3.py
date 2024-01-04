#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# ## Exact Inference

# In[2]:


def neighbour_potential(beta):
    table = np.ones((2,2))
    for x in range(0,2):
        for y in range(0,2):
            table[x,y] = np.exp(beta*1*(x==y)) 
    return table

def col_pot(n, beta):
    if n<2:
        raise Exception('n must be at least 2 (n: number of nodes)')
    pot = neighbour_potential(beta).reshape(-1)
    for _ in range(2,n):
        shape = pot.shape[0]
        pot_top = pot[:int(shape/2)]
        pot_bottom = pot_top[::-1] # reverse the order of pot_top, equivalent to pot[int(shape/2):]
        new_pot_top = pot_top * np.exp(1)
        new_pot_bottom = pot_bottom * 1
        new_pot_half = np.concatenate((new_pot_top,new_pot_bottom))
        pot = np.concatenate((new_pot_half, new_pot_half[::-1]))
    return pot

def var_to_var(n, beta):
    count_list = []
    max_len = len(bin(2**n-1)[2:])
    for col2 in range(2**n):
        col2_bin = bin(col2)[2:]
        l2 = len(col2_bin)
        if l2 < max_len:
            col2_bin = '0'*(max_len-l2) + col2_bin
        for col1 in range(2**n):
            col1_bin = bin(col1)[2:]
            l1 = len(col1_bin)
            if l1 < max_len:
                col1_bin = '0'*(max_len-l1) + col1_bin
            count = sum(bit2 == bit1 for bit2, bit1 in zip(col2_bin, col1_bin))
            count_list.append(np.exp(count))
    count_list = np.array(count_list).reshape(2**n,2**n)
    return count_list

def cal_prob(n, beta):
    '''
    n : length/width of lattice where size = length x width
    beta
    '''
    factor_graph = 1
        
    for _ in range(n-1):
        f = col_pot(n,beta) # column potential i.e. x_prime
        g = var_to_var(n,beta) # variable to variable g(x_i-1, x_i)
        factor_graph = np.multiply(g @ f, factor_graph)
    
    marginal_col = np.multiply(col_pot(n, beta), factor_graph).reshape([2 for i in range(n)])
    marginal_1_10 = marginal_col.sum(tuple([i for i in range(1,n-1)]))
    
    total_prob = marginal_1_10.sum()
    marginal = marginal_1_10 / total_prob
    
    assert marginal.shape == (2,2)
    
    return marginal


# ### $\beta$ = 0.01

# In[5]:


beta = 0.01
ei_small = cal_prob(10,beta)
print(f'P(x_1 = 1, x_10 = 1) = {ei_small[0,0]}')
print(f'P(x_1 = -1, x_10 = 1) = {ei_small[0,1]}')
print(f'P(x_1 = 1, x_10 = -1) = {ei_small[1,0]}')
print(f'P(x_1 = -1, x_10 = -1) = {ei_small[1,1]}')


# ### $\beta$ = 1

# In[6]:


beta = 1
ei_mid = cal_prob(10,beta)
print(f'P(x_1 = 1, x_10 = 1) = {ei_mid[0,0]}')
print(f'P(x_1 = -1, x_10 = 1) = {ei_mid[0,1]}')
print(f'P(x_1 = 1, x_10 = -1) = {ei_mid[1,0]}')
print(f'P(x_1 = -1, x_10 = -1) = {ei_mid[1,1]}')


# ### $\beta$ = 4

# In[7]:


beta = 4
ei_large = cal_prob(10,beta)
print(f'P(x_1 = 1, x_10 = 1) = {ei_large[0,0]}')
print(f'P(x_1 = -1, x_10 = 1) = {ei_large[0,1]}')
print(f'P(x_1 = 1, x_10 = -1) = {ei_large[1,0]}')
print(f'P(x_1 = -1, x_10 = -1) = {ei_large[1,1]}')


# ## Mean Field Approximation

# In[10]:


def mfa(n, beta, init, iteration):
    for _ in range(iteration):
        for col in range(n):
            for row in range(n):
                if col%(n-1)==0 and row%(n-1)==0: # 2 neighbours potential
                    if col < int(n/2) and row < int(n/2):
                        row_axis = (row+1, row)
                        col_axis = (col, col+1)
                    elif col > int(n/2) and row < int(n/2):
                        row_axis = (row+1, row)
                        col_axis = (col, col-1)
                    elif col < int(n/2) and row > int(n/2):
                        row_axis = (row-1, row)
                        col_axis = (col, col+1)
                    elif col > int(n/2) and row > int(n/2):
                        row_axis = (row-1, row)
                        col_axis = (col, col-1)    

                elif (col%(n-1)==0 and row%(n-1)!=0) or (row%(n-1)==0 and col%(n-1)!=0): # 3 neighbours potential
                    if row%(n-1)!=0 and col < int(n/2):
                        row_axis = (row-1, row, row+1)
                        col_axis = (col, col+1, col)
                    elif row%(n-1)!=0 and col > int(n/2):
                        row_axis = (row-1, row, row+1)
                        col_axis = (col, col-1, col)
                    elif col%(n-1)!=0 and row < int(n/2):
                        row_axis = (row, row+1, row)
                        col_axis = (col-1, col, col+1)
                    elif col%(n-1)!=0 and row > int(n/2):
                        row_axis = (row, row-1, row)
                        col_axis = (col-1, col, col+1)

                else: # 4 neighbours potential
                    row_axis = (row, row, row+1, row-1)
                    col_axis = (col-1, col+1, col, col)

                pot_pos = beta * (init[row_axis, col_axis]).sum()
                pot_neg = beta * (1-init[row_axis, col_axis]).sum()

                init[row, col] = np.exp(pot_pos) / (np.exp(pot_pos) + np.exp(pot_neg))
                
    return init


# ### $\beta$ = 0.01

# In[11]:


n = 10
beta = 0.01
init = np.random.rand(n,n)
iteration = 5000

q_star_small = mfa(n, beta, init, iteration)

print(f'P(x_1 = 1, x_10 = 1) = {q_star_small[0,-1] * q_star_small[-1,-1]}')
print(f'P(x_1 = -1, x_10 = 1) = {q_star_small[0,-1] * (1-q_star_small[-1,-1])}')
print(f'P(x_1 = 1, x_10 = -1) = {(1-q_star_small[0,-1]) * q_star_small[-1,-1]}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-q_star_small[0,-1]) * (1-q_star_small[-1,-1])}')


# ### $\beta$ = 1

# In[21]:


n = 10
beta = 1
init = np.random.rand(n,n)
iteration = 5000

q_star_mid = mfa(n, beta, init, iteration)

print('Frist Trial')
print(f'P(x_1 = 1, x_10 = 1) = {q_star_mid[0,-1] * q_star_mid[-1,-1]}')
print(f'P(x_1 = -1, x_10 = 1) = {q_star_mid[0,-1] * (1-q_star_mid[-1,-1])}')
print(f'P(x_1 = 1, x_10 = -1) = {(1-q_star_mid[0,-1]) * q_star_mid[-1,-1]}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-q_star_mid[0,-1]) * (1-q_star_mid[-1,-1])}')


# In[22]:


n = 10
beta = 1
init = np.random.rand(n,n)
iteration = 5000

q_star_mid = mfa(n, beta, init, iteration)

print('Second Trial')
print(f'P(x_1 = 1, x_10 = 1) = {q_star_mid[0,-1] * q_star_mid[-1,-1]}')
print(f'P(x_1 = -1, x_10 = 1) = {q_star_mid[0,-1] * (1-q_star_mid[-1,-1])}')
print(f'P(x_1 = 1, x_10 = -1) = {(1-q_star_mid[0,-1]) * q_star_mid[-1,-1]}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-q_star_mid[0,-1]) * (1-q_star_mid[-1,-1])}')


# In[23]:


n = 10
beta = 1
init = np.random.rand(n,n)
iteration = 5000

q_star_mid = mfa(n, beta, init, iteration)

print('Third Trial')
print(f'P(x_1 = 1, x_10 = 1) = {q_star_mid[0,-1] * q_star_mid[-1,-1]}')
print(f'P(x_1 = -1, x_10 = 1) = {q_star_mid[0,-1] * (1-q_star_mid[-1,-1])}')
print(f'P(x_1 = 1, x_10 = -1) = {(1-q_star_mid[0,-1]) * q_star_mid[-1,-1]}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-q_star_mid[0,-1]) * (1-q_star_mid[-1,-1])}')


# In[29]:


n = 10
beta = 1
init = np.random.rand(n,n)
iteration = 5000

q_star_mid = mfa(n, beta, init, iteration)

print('Fourth Trial')
print(f'P(x_1 = 1, x_10 = 1) = {q_star_mid[0,-1] * q_star_mid[-1,-1]}')
print(f'P(x_1 = -1, x_10 = 1) = {q_star_mid[0,-1] * (1-q_star_mid[-1,-1])}')
print(f'P(x_1 = 1, x_10 = -1) = {(1-q_star_mid[0,-1]) * q_star_mid[-1,-1]}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-q_star_mid[0,-1]) * (1-q_star_mid[-1,-1])}')


# In[30]:


n = 10
beta = 1
init = np.random.rand(n,n)
iteration = 5000

q_star_mid = mfa(n, beta, init, iteration)

print('Fifth Trial')
print(f'P(x_1 = 1, x_10 = 1) = {q_star_mid[0,-1] * q_star_mid[-1,-1]}')
print(f'P(x_1 = -1, x_10 = 1) = {q_star_mid[0,-1] * (1-q_star_mid[-1,-1])}')
print(f'P(x_1 = 1, x_10 = -1) = {(1-q_star_mid[0,-1]) * q_star_mid[-1,-1]}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-q_star_mid[0,-1]) * (1-q_star_mid[-1,-1])}')


# ### $\beta$ = 4

# In[24]:


n = 10
beta = 4
init = np.random.rand(n,n)
iteration = 5000

q_star_large = mfa(n, beta, init, iteration)

print('First Trial')
print(f'P(x_1 = 1, x_10 = 1) = {q_star_large[0,-1] * q_star_large[-1,-1]}')
print(f'P(x_1 = -1, x_10 = 1) = {q_star_large[0,-1] * (1-q_star_large[-1,-1])}')
print(f'P(x_1 = 1, x_10 = -1) = {(1-q_star_large[0,-1]) * q_star_large[-1,-1]}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-q_star_large[0,-1]) * (1-q_star_large[-1,-1])}')


# In[27]:


n = 10
beta = 4
init = np.random.rand(n,n)
iteration = 5000

q_star_large = mfa(n, beta, init, iteration)

print('Second Trial')
print(f'P(x_1 = 1, x_10 = 1) = {q_star_large[0,-1] * q_star_large[-1,-1]}')
print(f'P(x_1 = -1, x_10 = 1) = {q_star_large[0,-1] * (1-q_star_large[-1,-1])}')
print(f'P(x_1 = 1, x_10 = -1) = {(1-q_star_large[0,-1]) * q_star_large[-1,-1]}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-q_star_large[0,-1]) * (1-q_star_large[-1,-1])}')


# In[28]:


n = 10
beta = 4
init = np.random.rand(n,n)
iteration = 5000

q_star_large = mfa(n, beta, init, iteration)

print('Third Trial')
print(f'P(x_1 = 1, x_10 = 1) = {q_star_large[0,-1] * q_star_large[-1,-1]}')
print(f'P(x_1 = -1, x_10 = 1) = {q_star_large[0,-1] * (1-q_star_large[-1,-1])}')
print(f'P(x_1 = 1, x_10 = -1) = {(1-q_star_large[0,-1]) * q_star_large[-1,-1]}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-q_star_large[0,-1]) * (1-q_star_large[-1,-1])}')


# In[31]:


n = 10
beta = 4
init = np.random.rand(n,n)
iteration = 5000

q_star_large = mfa(n, beta, init, iteration)

print('Fourth Trial')
print(f'P(x_1 = 1, x_10 = 1) = {q_star_large[0,-1] * q_star_large[-1,-1]}')
print(f'P(x_1 = -1, x_10 = 1) = {q_star_large[0,-1] * (1-q_star_large[-1,-1])}')
print(f'P(x_1 = 1, x_10 = -1) = {(1-q_star_large[0,-1]) * q_star_large[-1,-1]}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-q_star_large[0,-1]) * (1-q_star_large[-1,-1])}')


# In[32]:


n = 10
beta = 4
init = np.random.rand(n,n)
iteration = 5000

q_star_large = mfa(n, beta, init, iteration)

print('Fifth Trial')
print(f'P(x_1 = 1, x_10 = 1) = {q_star_large[0,-1] * q_star_large[-1,-1]}')
print(f'P(x_1 = -1, x_10 = 1) = {q_star_large[0,-1] * (1-q_star_large[-1,-1])}')
print(f'P(x_1 = 1, x_10 = -1) = {(1-q_star_large[0,-1]) * q_star_large[-1,-1]}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-q_star_large[0,-1]) * (1-q_star_large[-1,-1])}')


# ## Gibbs Sampling

# In[5]:


def get_potential(n, beta, init, target, binary):
    '''
    input:
        beta: beta value, int
        ising: ising model, n x n numpy array
        target: target varibale, [row, col], tuple/list/1d array with len = 2
        binary: target binary value, either 1 or -1, int
    '''
    
    row, col = target[0], target[1]
    if col%(n-1)==0 and row%(n-1)==0: # 2 neighbours potential
        if col < int(n/2) and row < int(n/2):
            row_axis = (row+1, row)
            col_axis = (col, col+1)
        elif col > int(n/2) and row < int(n/2):
            row_axis = (row+1, row)
            col_axis = (col, col-1)
        elif col < int(n/2) and row > int(n/2):
            row_axis = (row-1, row)
            col_axis = (col, col+1)
        elif col > int(n/2) and row > int(n/2):
            row_axis = (row-1, row)
            col_axis = (col, col-1)
    elif (col%(n-1)==0 and row%(n-1)!=0) or (row%(n-1)==0 and col%(n-1)!=0): # 3 neighbours potential
        if row%(n-1)!=0 and col < int(n/2):
            row_axis = (row-1, row, row+1)
            col_axis = (col, col+1, col)
        elif row%(n-1)!=0 and col > int(n/2):
            row_axis = (row-1, row, row+1)
            col_axis = (col, col-1, col)
        elif col%(n-1)!=0 and row < int(n/2):
            row_axis = (row, row+1, row)
            col_axis = (col-1, col, col+1)
        elif col%(n-1)!=0 and row > int(n/2):
            row_axis = (row, row-1, row)
            col_axis = (col-1, col, col+1)
    else: # 4 neighbours potential
        row_axis = (row, row, row+1, row-1)
        col_axis = (col-1, col+1, col, col)
    
    pot = np.prod(np.exp(beta * (init[row_axis,col_axis]==binary)))
    return pot


def gibbs_sampling(n, beta, init):
    '''
    input:
        beta: beta value, int
        n: size of Icing model, n x n lattice, int
        init: initialisation of Icing model, n x n numpy array
    '''
    iteration = 0
    conv = 0
    sample = []
    
    while conv < 20:
        iteration += 1
        init_old = np.copy(init)
        for col in range(n):
            for row in range(n):
                pot_pos = get_potential(n, beta, init, (row,col), 1)
                pot_neg = get_potential(n, beta, init, (row,col), -1)
                prob_pos = pot_pos/(pot_pos + pot_neg)

                if prob_pos < 0 or prob_pos > 1:
                    print('Probability must be between 0 and 1')
                    break

                init[row,col] = np.random.choice([1,-1], p = (prob_pos, 1-prob_pos))

        sample.append([init[0,-1], init[-1,-1]])

        if np.array_equal(init_old, init):
            conv += 1
        else:
            conv = 0

        if iteration == 10000:
            break

    return np.array(sample), iteration


# ### $\beta$ = 0.01

# In[38]:


n = 10
beta = 0.01
init = np.random.choice([1,-1], size=(n,n))

gib_samp_small, iter_small = gibbs_sampling(n, beta, init)

x1 = (gib_samp_small[int(iter_small*0.2):,0] == 1).sum() / len(gib_samp_small[int(iter_small*0.2):,0]) # P(x_1 = 1)
x10 = (gib_samp_small[int(iter_small*0.2):,1] == 1).sum() / len(gib_samp_small[int(iter_small*0.2):,1]) # P(x_10 = 1)

print('First Trial')
print(f'P(x_1 = 1, x_10 = 1) = {x1 * x10}')
print(f'P(x_1 = -1, x_10 = 1) = {(1-x1) * x10}')
print(f'P(x_1 = 1, x_10 = -1) = {x1 * (1-x10)}')
print(f'P(x_1 = -1, x_10 = -1) = {(1-x1) * (1-x10)}')


# In[7]:


n = 10
beta = 0.01
init = np.random.choice([1,-1], size=(n,n))

gib_samp_small, iter_small = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == 1),(gib_samp_small[int(iter_small*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == -1),(gib_samp_small[int(iter_small*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == 1),(gib_samp_small[int(iter_small*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == -1),(gib_samp_small[int(iter_small*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_small[int(iter_small*0.2):,0])

print('First Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[8]:


n = 10
beta = 0.01
init = np.random.choice([1,-1], size=(n,n))

gib_samp_small, iter_small = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == 1),(gib_samp_small[int(iter_small*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == -1),(gib_samp_small[int(iter_small*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == 1),(gib_samp_small[int(iter_small*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == -1),(gib_samp_small[int(iter_small*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_small[int(iter_small*0.2):,0])

print('Second Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[9]:


n = 10
beta = 0.01
init = np.random.choice([1,-1], size=(n,n))

gib_samp_small, iter_small = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == 1),(gib_samp_small[int(iter_small*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == -1),(gib_samp_small[int(iter_small*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == 1),(gib_samp_small[int(iter_small*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == -1),(gib_samp_small[int(iter_small*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_small[int(iter_small*0.2):,0])

print('Third Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[10]:


n = 10
beta = 0.01
init = np.random.choice([1,-1], size=(n,n))

gib_samp_small, iter_small = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == 1),(gib_samp_small[int(iter_small*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == -1),(gib_samp_small[int(iter_small*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == 1),(gib_samp_small[int(iter_small*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == -1),(gib_samp_small[int(iter_small*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_small[int(iter_small*0.2):,0])

print('Fourth Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[11]:


n = 10
beta = 0.01
init = np.random.choice([1,-1], size=(n,n))

gib_samp_small, iter_small = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == 1),(gib_samp_small[int(iter_small*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == -1),(gib_samp_small[int(iter_small*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == 1),(gib_samp_small[int(iter_small*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_small[int(iter_small*0.2):,0] == -1),(gib_samp_small[int(iter_small*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_small[int(iter_small*0.2):,0])

print('Fifth Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# ### $\beta$ = 1

# In[12]:


n = 10
beta = 1
init = np.random.choice([1,-1], size=(n,n))

gib_samp_mid, iter_mid = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == 1),(gib_samp_mid[int(iter_mid*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == -1),(gib_samp_mid[int(iter_mid*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == 1),(gib_samp_mid[int(iter_mid*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == -1),(gib_samp_mid[int(iter_mid*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_mid[int(iter_mid*0.2):,0])

print('First Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[13]:


n = 10
beta = 1
init = np.random.choice([1,-1], size=(n,n))

gib_samp_mid, iter_mid = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == 1),(gib_samp_mid[int(iter_mid*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == -1),(gib_samp_mid[int(iter_mid*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == 1),(gib_samp_mid[int(iter_mid*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == -1),(gib_samp_mid[int(iter_mid*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_mid[int(iter_mid*0.2):,0])

print('Second Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[14]:


n = 10
beta = 1
init = np.random.choice([1,-1], size=(n,n))

gib_samp_mid, iter_mid = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == 1),(gib_samp_mid[int(iter_mid*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == -1),(gib_samp_mid[int(iter_mid*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == 1),(gib_samp_mid[int(iter_mid*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == -1),(gib_samp_mid[int(iter_mid*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_mid[int(iter_mid*0.2):,0])

print('Third Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[15]:


n = 10
beta = 1
init = np.random.choice([1,-1], size=(n,n))

gib_samp_mid, iter_mid = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == 1),(gib_samp_mid[int(iter_mid*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == -1),(gib_samp_mid[int(iter_mid*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == 1),(gib_samp_mid[int(iter_mid*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == -1),(gib_samp_mid[int(iter_mid*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_mid[int(iter_mid*0.2):,0])

print('Fourth Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[16]:


n = 10
beta = 1
init = np.random.choice([1,-1], size=(n,n))

gib_samp_mid, iter_mid = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == 1),(gib_samp_mid[int(iter_mid*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == -1),(gib_samp_mid[int(iter_mid*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == 1),(gib_samp_mid[int(iter_mid*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_mid[int(iter_mid*0.2):,0] == -1),(gib_samp_mid[int(iter_mid*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_mid[int(iter_mid*0.2):,0])

print('Fifth Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# ### $\beta$ = 4

# In[17]:


n = 10
beta = 4
init = np.random.choice([1,-1], size=(n,n))

gib_samp_large, iter_large = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == 1),(gib_samp_large[int(iter_large*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == -1),(gib_samp_large[int(iter_large*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == 1),(gib_samp_large[int(iter_large*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == -1),(gib_samp_large[int(iter_large*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_large[int(iter_large*0.2):,0])

print('First Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[24]:


n = 10
beta = 4
init = np.random.choice([1,-1], size=(n,n))

gib_samp_large, iter_large = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == 1),(gib_samp_large[int(iter_large*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == -1),(gib_samp_large[int(iter_large*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == 1),(gib_samp_large[int(iter_large*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == -1),(gib_samp_large[int(iter_large*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_large[int(iter_large*0.2):,0])

print('Second Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[25]:


n = 10
beta = 4
init = np.random.choice([1,-1], size=(n,n))

gib_samp_large, iter_large = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == 1),(gib_samp_large[int(iter_large*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == -1),(gib_samp_large[int(iter_large*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == 1),(gib_samp_large[int(iter_large*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == -1),(gib_samp_large[int(iter_large*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_large[int(iter_large*0.2):,0])

print('Third Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[27]:


n = 10
beta = 4
init = np.random.choice([1,-1], size=(n,n))

gib_samp_large, iter_large = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == 1),(gib_samp_large[int(iter_large*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == -1),(gib_samp_large[int(iter_large*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == 1),(gib_samp_large[int(iter_large*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == -1),(gib_samp_large[int(iter_large*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_large[int(iter_large*0.2):,0])

print('Fourth Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[35]:


n = 10
beta = 4
init = np.random.choice([1,-1], size=(n,n))

gib_samp_large, iter_large = gibbs_sampling(n, beta, init)

pos_pos = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == 1),(gib_samp_large[int(iter_large*0.2):,1] == 1)).sum()
neg_pos = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == -1),(gib_samp_large[int(iter_large*0.2):,1] == 1)).sum()
pos_neg = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == 1),(gib_samp_large[int(iter_large*0.2):,1] == -1)).sum()
neg_neg = np.logical_and((gib_samp_large[int(iter_large*0.2):,0] == -1),(gib_samp_large[int(iter_large*0.2):,1] == -1)).sum()
total_epoch = len(gib_samp_large[int(iter_large*0.2):,0])

print('Fifth Trial')
print(f'P(x_1 = 1, x_10 = 1) = {pos_pos / total_epoch}')
print(f'P(x_1 = -1, x_10 = 1) = {neg_pos / total_epoch}')
print(f'P(x_1 = 1, x_10 = -1) = {pos_neg / total_epoch}')
print(f'P(x_1 = -1, x_10 = -1) = {neg_neg / total_epoch}')


# In[ ]:





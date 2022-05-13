
"""
In this exercise, you will implement a one-epoch LDA training process
with a simple example. Please complete the functions below.
"""

import numpy as np

def softmax(arr):
    '''
    args:
        arr: an array of shape (m, y)
        m is the number of training examples
        y is the number of classes
        example: arr = [[1,2,3],
                        [0,1,2]]
                means that there are two training examples, 
                and each has 3 differnet classes
    returns:
        pred: an array of shape (m, y)
        but for each training example, 
        a softmax is applied to convert the values into probs
        example: for the first row of arr, softmax is applied to [1,2,3]
                 for the second row of arr, softmax is applied to [0,1,2]
    hint:
        take advantage of the numpy's element-wise calculation
    '''
    exp = np.exp(arr)
    s = np.sum(exp, axis = 1, keepdims = True)

    ### Your Codes Here ###

    return exp / exp.sum(axis=0)



def rnn_cell(xt, a_prev, Wa, Wx, Wy, b1, b2):
    """
    Implements a single forward step of the rnn
    rnn is much simpler than a LSTM
    each step t only generates one hidden state/output a_t. 
    There is no cell/memory state
    each step t takes some new input x_t.
    
    The calculation at step t is described below:
        1. caculate the next hidden state a_next using 
           the previous hidden state a_prev and the current input xt.
           a_next = np.tanh(a_prev @ Wa + xt @ Wx + b1),
           where @ denotes matrix multiplication
           tanh denotes the tanh non-linear transformation, element-wise
           b1 denotes the noise
           trainable parameters are: Wa, Wx, and b1
        2. calculate the prediction at this step
           pred = softmax(a_next @ Wy + b2)
           the trainable parameters are Wy and b2.
        3. the a_next will be used as the a_prev in the next time step.
        
    args:
        xt: new input at timestep t, of shape (m,n_x).
        a_prev: hidden state at timestep t-1, of shape (m,n_a)
        m is the number of training examples
    
    trainable parameters:
        Wa: of shape (n_a, n_a), so that a_prev @ Wa is of shape 
        Wx: of shape (n_x, n_a), so that x_t @ Wx is of shape (m, n_a) as well.
        Wy: of shape (n_a, n_y), so that pred = a_next @ Wy is of shape (m,n_y).
        b1: of shape (1, n_a)
        b2: of shape (1, n_y)
                        
    Returns:
        a_next: next hidden state, of shape (m, n_a)
        pred: prediction at timestep t, of shape (m,n_y).
        cache: tuple (a_next, a_prev, xt, Wa, Wx, Wy, b1, b2)
        cache is used in back propagation.
    """
    # step 1: calculate a_next
    ### Your Codes Here ###
    a_next =a_next = np.tanh(a_prev @ Wa + xt @ Wx + b1)
    # step 2: calculate pred
    ### Your Codes Here ###
    pred = pred = softmax(a_next @ Wy + b2) 
    # step 3: compile the cache
    cache = (a_next, a_prev, xt, Wa, Wx, Wy, b1, b2)
    
    return a_next,pred, cache

def forward_pass(x, n_a, n_y):
    """
    Implement the forward propagation of the rnn.
    if the input x has T_x time steps, we run "rnn_cell" for each time step
    to get the hidden states and predictions of each step

    args:
        x: input series, of shape (m, T_x, n_x), 
           m is the number of training examples, 
           n_x is the dimension of each input
           T_x is the number of time steps.
        n_a: hidden state dimension
        n_y: the number of classes

    returns:
        a: hidden states for every time-step, of shape (m, T_x, n_a)
        predictions: predictions for every time-step, of shape (m, T_x, n_y)
        caches: list of caches in each time step, used for back propagation.
    """
    
    # initialize hidden state, pred, and cache list, and parameters
    m, T_x, n_x = x.shape
    a = np.zeros((m, T_x, n_a))
    predictions = np.zeros((m, T_x, n_y))             
    caches = []
    Wx = np.random.randn(n_x, n_a)
    Wa = np.random.randn(n_a, n_a)
    ### Your Codes Here ###
    Wy = np.random.randn(n_a, n_y)
    b1 = np.random.randn(1, n_a)
    b2 = np.random.randn(1, n_y)
    
    # forward propagation through each time step
    # within each time step, we call "rnn_cell" and save necessary values 
    a_prev = np.zeros((m,n_a))
    for t in range(T_x):
        # calculate a_next and pred for the current time step
        ### Your Codes Here ###
        xt = x[:,t,:]
        a_next, pred, cache = rnn_cell(xt, a_prev, Wa, Wx, Wy, b1, b2)
        # save a_next to a
        a[:,t,:] = a_next
        # save pred to predictions 
        ### Your Codes Here ###
        predictions[:,t,:] = pred
        # Append "cache" to "caches" (≈1 line)
        caches.append(cache)
    
    return a, predictions, caches

'''
After defining these functions, 
please complete a forward pass of rnn where 

1. the hidden state dimension is 4
2. the number of classes is 5
3. the sequence x is randomized using np.random.randn()
4. there are 10 time steps
5. at each time step, xt's dimension is 10
'''

### Your Codes Here ###
np.random.seed(1)
xt_tmp = np.random.randn(3,10,10)
a_next_tmp, yt_tmp, cache_tmp = forward_pass(xt_tmp, 4, 5)
print(a_next_tmp)
print(yt_tmp)
print(cache_tmp)
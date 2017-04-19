#!/usr/bin/env python

import numpy as np
import scipy.sparse as sp
import theano
import theano.sparse as S
import theano.tensor as T
import time

def optimize_func(transform_matrix):
    t0 = time.time()
    M = S.csr_matrix(dtype=theano.config.floatX)
    N = S.csr_matrix(dtype=theano.config.floatX)
    ON = S.csr_matrix(dtype=theano.config.floatX)
    lr = T.scalar('learning rate', dtype=theano.config.floatX)
    # print M, N, ON, lr

    TN = S.dot(transform_matrix,N)
    D = T.sqr(M - TN)
    # PD = S.sqr(N-ON)
    # PD = T.sqrt(S.sp_sum(PD, 1))
    # TPD = T.sqr(TN - ON)
    # TPD = T.sqrt(TPD.sum(1))

    # D2 = T.sqr(PD-TPD)
    cost = T.sum(D) #+ T.sum(D2)

    list_in = [lr, M, N, ON]
    gradient = T.grad(cost, transform_matrix)
    new_transform_matrix = transform_matrix - lr*gradient
    t1 = time.time()
    print 'opt func cost is ' + str(t1 - t0)
    return theano.function(list_in, cost, updates=[(transform_matrix, new_transform_matrix)], on_unused_input='ignore')
def iterations(M, N, curr_feature, vnum):
    t0 = time.time()
    data = np.identity(vnum)
    data = np.asarray(data, dtype=theano.config.floatX)
    transform_matrix = theano.shared(value=data, name='transform matrix', borrow=True)

    func = optimize_func(transform_matrix)
    prev = np.inf
    t1 = time.time()
    print 't1 - t0 is ' is str(t1 - t0)
    for i in range(vnum*20):
        t2 = time.time()
        order = np.random.permutation(curr_feature.shape[0])
        data = np.copy(curr_feature)[order,:]
        data = np.asarray(data, dtype=theano.config.floatX)
        ON = sp.csr_matrix(data, shape=N.shape)
        t4 = time.time()
        out = func(0.5/vnum, M, N, ON)
        print 'opt func is ' + str(time.time() - t4)
        # print abs(out - prev)
        if np.abs(prev-out) < 0.001 or np.isnan(out) or i == 10:
            print np.abs(prev - out)
            break
        prev = out
        t3 = time.time()
        print "t3 - t2 is " + str(t3 - t2)

    return transform_matrix.get_value()

# vnum = 1000
# dim = 64

# cur_feature = np.random.rand(vnum, dim)
# cur_feature = np.asarray(cur_feature, dtype=theano.config.floatX)
# N = sp.csr_matrix(cur_feature, shape=(vnum, dim))

# random_transform = np.random.rand(vnum, vnum)
# prev_feature = np.asarray(np.dot(random_transform,cur_feature), dtype=theano.config.floatX)
# M = sp.csr_matrix(prev_feature, shape=(vnum, dim))


# transform_matrix = iterations(M, N, vnum)

# print random_transform, random_transform - transform_matrix

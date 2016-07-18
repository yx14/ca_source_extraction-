# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 17:32:21 2016

@author: clusteradmin
"""
#%%
import numpy as np 
#the temporal merge program 
#merge components if they have a strong temporal correlation regardless of spatial overlap

#outputs:
#A_comb2 is A with components merged and others removed
#updates temporal components for new C and S  
def temporal_merge(Y, A, b, C, f):
    thr = 0.85

    nr = A.shape[1]
    #produce correlation matrix where each element is the temporal correlation between two components
    print type(nr)
    print C.shape
    test = np.transpose(C[1:nr, :])
    corr_m = np.corrcoef(test)    
    #corr_m = np.corrcoef(np.transpose(C[1:nr, :]))
    #boolean array
    corr_m = np.triu(corr_m) >= thr
    #remove the diagonal
    corr_m = corr_m - np.diag(np.diag(corr_m))
    (merge_x, merge_y) = np.nonzero(corr_m)
    merge_u = merge_x.shape[0]
    A_comb = np.zeros(A.shape)
    #if merge_hu is 0, issues boolean deprecation warning 
    C_comb = np.zeros([A.shape[1] - merge_u, C.shape[1]])
    
    #A_comb has components merged within merge_x elements
    
    for i in range(A.shape[1]):
        if any(i == k for k in merge_x):
            inds = [j for j, x in enumerate(merge_x) if x == i]  
            for j in inds:
                A_comb[:, i] = A[:, i] + A[:, merge_y(j)]
        else:
            A_comb[:, i] = A[:, i]
    
    merge_ct = 0
    A_comb2 = np.zeros([A.shape[0], A.shape[1] - merge_u])
    #A_comb2 drops merge_y components
    for i in range(A.shape[1]):
        if not any(i == k for k in merge_y):
            A_comb2[:, merge_ct] = A_comb[:, i]
            C_comb[merge_ct, :] = C[i, :]
            merge_ct = merge_ct + 1
   
    Yr = np.reshape(Y, [Y.shape[0]*Y.shape[1], Y.shape[2]])
    [C_comb, f, P, S_comb, YrA]  = cse.temporal.update_temporal_components(Yr,A,b,C,f,bl=None,c1=None,sn=None,g=None,**options['temporal_params'])
    return A_comb2, C_comb, S_comb, f, P
'''   
Y = np.random.rand(10000, 1)
Y = np.reshape(map(round, 10*Y), [10, 10, 100])
test1 = np.random.rand(1000, 50)
test = np.random.rand(50, 1000)
b = []
f= []

test = temporal_merge(Y, test1, b, test, f)
 '''           
    
    
    
    

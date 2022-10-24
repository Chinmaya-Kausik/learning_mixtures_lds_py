#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 19:12:03 2022

@author: soominkwon
"""

import numpy as np
from utils import generate_models, compute_autocovariance, generate_mixed_lds, compute_separation
from classification import classification
from subspace_est import subspace_estimation
from clustering import clustering_fast
from model_estimation_new import model_estimation
from helpers import get_clusters, model_errors

# initializing parameters
d   = 30
K   = 4
rho = 0.5

Msubspace        = 30  * d
Mclustering      = 10 * d
Mclassification  = 50 * d
M = Msubspace + Mclustering + Mclassification

Tsubspace        = 20
Tclustering      = 20
Tclassification  = 5

Ntrials = 12
block_num = 15

# initializing lists for errors
all_trials_A = np.zeros((Ntrials, block_num+1))
all_trials_W = np.zeros((Ntrials, block_num+1))

for trial in range(Ntrials):
    print('Trial Number:', trial)
    # generating labels and lengths of trajectories
    true_labels = np.random.randint(1, K, (M, 1))

    Ts = np.concatenate([np.ones((Msubspace, 1))*Tsubspace, np.ones((Mclustering,1))*Tclustering,
                         np.ones((Mclassification, 1))*Tclassification], axis=0)
    
    # generating synthetic data
    As, Whalfs = generate_models(d=d, K=K, rho=rho)

    # squaring Ws
    Ws = []
    for k in range(K):
        Ws.append(Whalfs[k]**2)
    
    # generating synthetic data
    Gammas, Ys = compute_autocovariance(As=As, Whalfs=Whalfs)
    delta_gy = compute_separation(Gammas=Gammas, Ys=Ys)
    data = generate_mixed_lds(As=As, Whalfs=Whalfs, true_labels=true_labels, Ts=Ts)
    
    data_subspace = data[:Msubspace]
    data_clustering = data[Msubspace:(Msubspace+Mclustering)]
    data_classification = data[(Msubspace+Mclustering):]

    print('Synthetic Data Generated')
    
    # coarse subspace estimation
    Vs, Us = subspace_estimation(data_sub_est=data_subspace, K=K)
    print('Coarse Subspace Estimated')
    
    # clustering
    labels_clustering, S_orig, S = clustering_fast(data=data_clustering, Vs=Vs, Us=Us, K=K, tau=delta_gy/4,
                                        no_subspace=0)
    print('Coarse Labels Clustered')

    labels_clustering = labels_clustering + 1

    # getting the data corresponding to clusters
    clusters = get_clusters(data=data_clustering, labels=labels_clustering.squeeze(), K=K)

    # determine permutation
    subtl = true_labels[Msubspace:(Msubspace+Mclustering)]
    perm = np.zeros((K, 1)) # true label -> estimated label
    invperm = np.zeros((K, 1)) # estimated label -> true label
    for k in range(1, K+1):
        idx = (subtl == k)*1
        tmp = []
        for val in range(len(idx)):
            if idx[val] == 1:
                tmp.append(labels_clustering[val])
        tmpint = 0
        if tmp:
            tmpint += round(np.median(tmp))
        perm[k-1] = tmpint
        invperm[tmpint-1] = k-1


    # coarse model estimation
    Ahats, Whats = model_estimation(clusters)
    print('Coarse Models Estimated')

    # classification
    tmpidx = np.linspace(5, np.log(Mclassification), block_num).T
    tmpidx = np.ceil(np.exp(tmpidx))
    tmpidx[len(tmpidx)-1] = Mclassification
    tmpidx = np.insert(tmpidx, 0, 0)
    T_coarse = Tclustering * Mclustering
    T_refined = T_coarse + Tclassification * tmpidx

    # computing initial model errors
    A_error, W_error = model_errors(Ahats=Ahats, As=As, Whats=Whats, Ws=Ws, invperm=invperm)
    print('Initial A Error:', A_error)
    print('Initial W Error:', W_error)

    # initializing error lists
    errors_A = []
    errors_W = []
    
    # coarse errors
    errors_A.append(A_error)
    errors_W.append(W_error)
    
    # going through all block iterations
    for j in range(block_num):
        print("Block Iteration: ", j)
        idx1 = int(tmpidx[j])
        idx2 = int(tmpidx[j+1])

        newdata = data_classification[idx1:idx2] # data for classification

        # coarse model classification
        newlabels = classification(data_classification=newdata, Ahats=Ahats, Whats=Whats)
        newlabels = newlabels + 1

        new_clusters = get_clusters(data=newdata, labels=newlabels.squeeze(), K=K)

        # refining models
        refined_Ahats, refined_Whats = model_estimation(clusters)
        refined_err_Ahats, refined_err_Whats = model_errors(Ahats=refined_Ahats, As=As,
                                                            Whats=refined_Whats, Ws=Ws, invperm=invperm)
        print(refined_err_Ahats)
        # appending errors
        errors_A.append(refined_err_Ahats)
        errors_W.append(refined_err_Whats)
        
    
    all_trials_A[trial, :] = np.asarray(errors_A)
    all_trials_W[trial, :] = np.asarray(errors_W)

np.savez('A_errors.npz', all_trials_A)
np.savez('W_errors.npz', all_trials_W)
np.savez('T_refined.npz', T_refined)

    
    

    
    
    

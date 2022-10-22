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

# initializing parameters
d   = 3
K   = 2
rho = 0.5

Msubspace        = 30  * d
Mclustering      =  d
Mclassification  = 50 * d
M = Msubspace + Mclustering + Mclassification

Tsubspace        = 20
Tclustering      = 20
Tclassification  = 5

# generating labels and lengths of trajectories
true_labels = np.random.randint(1, K, (M, 1))
Ts = np.concatenate([np.ones((Msubspace,1))*Tsubspace, np.ones((Mclustering,1))*Tclustering,
                     np.ones((Mclassification,1))*Tclassification], axis=0)

# generating synthetic data
As, Whalfs = generate_models(d=d, K=K, rho=rho)
Gammas, Ys = compute_autocovariance(As=As, Whalfs=Whalfs)
delta_gy = compute_separation(Gammas=Gammas, Ys=Ys)
data = generate_mixed_lds(As=As, Whalfs=Whalfs, true_labels=true_labels, Ts=Ts)

data_subspace = data[:Msubspace]
data_clustering = data[Msubspace:(Msubspace+Mclustering)]
data_classification = data[(Msubspace+Mclustering):]

# subspace estimation
Vs, Us = subspace_estimation(data_sub_est=data_subspace, K=K)
print('subspace estimated')
# clustering
labels_clustering = clustering_fast(data=data_clustering, Vs=Vs, Us=Us, K=K, tau=delta_gy/4,
                                    no_subspace=0)



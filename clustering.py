#!/usr/bin/env python
# coding: utf-8

# This file computes the misclustering rate with and without dimensionality reduction. Note that misclustering rate is only valid with K=2. 

# In[2]:

from utils import compute_autocovariance, compute_separation
import numpy as np
from scipy import linalg

## Algorithm for fast clustering
def clustering_fast(data, Vs, Us, K, tau, no_subspace):
    """
    Clustering algorithm
    
    Parameters:
        data: mixed LDS data
        Vs, Us: subspace estimation result
        K: Number of clusters
        tau: separation parameter
        no_subspace: 0/1 depending on whether we want to cluster with or without subspace estimation
        
    Returns:
        labels: predicted labels
        S_original: ?
        S: similarity matrix
    """
    M = len(data)
    d = Vs[0].shape[0]
    T = data[0].shape[1] - 1
    N = int(np.floor(T/4))
    Omega1 = list(range(N,2*N-1))
    Omega2 = list(range(3*N,4*N-1))
    
    tmp1_Gammas = []
    tmp2_Gammas = []
    tmp1_Ys = []
    tmp2_Ys = []
    
    for m in range(M):
        X = data[m]
        X1 = X[:, [i-1 for i in Omega1]]
        X2 = X[:, [i-1 for i in Omega2]]
        Xp1 = X[:,Omega1]
        Xp2 = X[:,Omega2]
        
        tmp1_Gammas.append(X1 @ X1.T / N)
        tmp2_Gammas.append(X2 @ X2.T / N)
        tmp1_Ys.append(X1 @ Xp1.T / N)
        tmp2_Ys.append(X2 @ Xp2.T / N)
        
    S_original = np.zeros([M,M])
    S = np.ones([M,M])
    
    for m1 in range(M):
        tmp1_Gamma1 = tmp1_Gammas[m1]
        tmp2_Gamma1 = tmp2_Gammas[m1]
        tmp1_Y1 = tmp1_Ys[m1]
        tmp2_Y1 = tmp2_Ys[m1]
        for m2 in range(m1+1,M):
            tmp1_Gamma2 = tmp1_Gammas[m2]
            tmp2_Gamma2 = tmp2_Gammas[m2]
            tmp1_Y2 = tmp1_Ys[m2]
            tmp2_Y2 = tmp2_Ys[m2]
            
            stat_Gamma = 0
            stat_Y = 0
            
            if no_subspace == 0:
                for i in range(d):
                    stat_Gamma +=  (Vs[i].T @ (tmp1_Gamma1[:,i] - tmp1_Gamma2[:,i])).T @ (Vs[i].T @ (tmp2_Gamma1[:,i] - tmp2_Gamma2[:,i]))
                    stat_Y = (Us[i].T @ (tmp1_Y1[:,i] - tmp1_Y2[:,i])).T @ (Us[i].T @ (tmp2_Y1[:,i] - tmp2_Y2[:,i]))
            else:
                stat_Gamma = ((tmp1_Gamma1[:,i] - tmp1_Gamma2[:,i]).T@((tmp1_Gamma1[:,i] - tmp1_Gamma2[:,i]))).trace()
                stat_Y  = ((tmp1_Y1[:,i] - tmp1_Y2[:,i]).T @ (tmp1_Y1[:,i] - tmp1_Y2[:,i])).trace()
                
            stat = stat_Gamma  + stat_Y
            
            S_original[m1,m2] = stat
            S_original[m2,m1] = stat
            
            
            S[m1,m2] = 0 if stat > tau else 1
            S[m2,m1] = S[m1,m2]
    
    #Cluster using k-means
    _, U = linalg.eigh(S,subset_by_index = [M-K-1,M-1]) ## top k-eigenvectors
    centers = np.zeros([K,K])
    
    for k in range(K):
        col = U[:,k]
        idx = np.argmax(abs(col))
        val = col[idx]
        centers[k][k] = val

    U = U.T
    distances = np.zeros((K,M))
    for t in range(1,50):
        for k in range(K):
            center = centers[:k]
            res = U - center
            distances[k,:] = sum(res**2,axis=0)
        labels = np.argmin(distances,axis=0)
        for k in range(1,K):
            subU = U[:,labels == k]
            centers[:,k] = np.mean(subU, axis=1)
    
    labels = labels.T

    return labels

# In[9]:


### Temp
# A = np.ones([5,5])
# A.trace()


# # In[14]:


# ## Initial setup 
# Ntrial = 30 
# d = 40
# K =  2
# rho = 0.5
# delta_A = 0.12
# Mclustering = 5*d
# Tclusterings = np.array([10*i for i in range(1,7)])


# # In[16]:


# # To store the parameters of the linear models
# As = [[] for _ in range(K)]
# Whalfs = [[] for _ in range(K)]
# Ws = [[] for _ in range(K)]

# R =  linalg.orth(np.random.rand(d,d)) #Random Orthogonal matrix

# # Generate linear models using a random orthogonal matrix R
# for k in range(K):
#     rho_A = rho + ((-1)**k)*delta_A
#     As[k] = rho_A * R
#     Whalfs[k] = np.identity(d)
#     Ws[k] = Whalfs[k]@Whalfs[k]


# # In[15]:


# # To store errors without/with subspace estimation and dimension reduction
# error_list_without = np.zeros([len(Tclusterings), Ntrial])
# error_list_with = np.zeros([len(Tclusterings), Ntrial])


# In[20]:


# Compute Gamma's, Y's, and delta_{Gamma,Y}'s
# Gammas, Ys = compute_autocovariance(As,Whalfs)
# delta_gy  = compute_separation(Gammas, Ys)
# tau = delta_gy/4

# for k_T in range(len(Tclusterings)):
#     Tclustering = Tclusterings[k_T]
#     for k_trial in range(Ntrial): 
#         # Generate a mixed LDS
#         true_labels = np.random.randint(K,size=[Mclusterings,1])
#         Ts = np.ones([Mclustersing,1])*Tclustering
#         data = generate_mixed_lds(As, Whalfs,true_labels,Ts)
        
#         #Subspace estimation
#         Vs, Us = subspace_estimation(data,K)
        
#         #clustering without subspace estimation
#         no_subspace = 1 #0/1 clustering with/without dim reduction
#         labels_without = clustering_fast(data,Vs,Us, K, tau, no_subspace)
#         no_subspace = 0
#         labels_without =  clustering_fast(data,Vs,Us, K, tau, no_subspace)
        
#         ### What exactly is happening this lines???
#         mis_without = min(np.mean(abs(labels_without - true_labels), np.mean(abs(1-labels_without - true_labels))))
#         mis_with = min(np.mean(abs(labels_with - true_labels), np.mean(abs(1-labels_with - true_labels))))
        
#         error_list_without[k_T,k_trial] = mis_without 
#         error_list_with[k_T,k_trial] = mis_with
        
# errors_without = np.mean(error_list_without,axis=1)
# errors_with = mean(error_list_with,axis=1)


# ##Plot the errors
# plt.plot(Tclusterings,errors_without, 'b--o')
# pt.plot(Tclusterings,errors_with,'r-o')
# legend(["without subspace", "with subspace"])
# xlabel('$T_{\rm clustering}$')
# ylabel("clustering error")
# plt.show()


# In[ ]:





# ## Appendix

# In[4]:





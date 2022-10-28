import numpy as np
#input:
    #clusters:
        #for k = 1,...,K
        #clusters[k] is an array of matrices X_m of dimension d x (T_m+1)
        #where d = number of features, T_m = trajectory length 
#output:
    #Ahats
        #for k = 1,...,K
        #Ahats[k] is matrix of dim dxd 
    #Whats
        #for k = 1,...,K
        #Whats[k] is matrix of dim dxd 
def model_estimation(clusters):
    K = len(clusters)
    Ahats = []
    Whats = []
    d = len(clusters[0][0])
    
    for k in range(K):
        traj_lengths = [len(X[0])-1 for X in clusters[k]]
        tmpT = sum(traj_lengths)
        T = max(traj_lengths)
        
        tp = np.array([np.pad(X[:,:len(X[0])-1], ((0,0),(0,T+1-len(X[0]))), 'constant', constant_values=(0)) for X in clusters[k]])
        tpp = np.array([np.pad(X[:,1:], ((0,0),(0,T+1-len(X[0]))), 'constant', constant_values=(0)) for X in clusters[k]])
        xxt = np.sum(np.matmul(tp, np.transpose(tp,(0,2,1))), axis=0)
        xpxt = np.sum(np.matmul(tpp, np.transpose(tp,(0,2,1))), axis=0)
        Ahat = np.matmul(xpxt, np.linalg.inv(xxt))
        Ahats.append(Ahat)
        
        whats = tpp - np.matmul(Ahat, tp)
        What = 1/tmpT*np.sum(np.matmul(whats, np.transpose(whats, (0,2,1))), axis=0)
        Whats.append(What)

    return Ahats,Whats
#input:
    #clusters:
        #for k = 1,...,K
        #clusters[k] is an array of matrices X_m each of dimension d x T
        #where d = number of features, T = length of trajectory
#output:
    #Ahats
        #for k = 1,...,K
        #Ahats[k] is matrix of dxd 
    #Whats
        #for k = 1,...,K
        #Whats[k] is matrix of dxd 
def model_estimation(clusters):
    K = len(clusters)
    Ahats = []
    Whats = []
    d = len(clusters[0][0])
    T = len(clusters[0][0][0])
    
    for k in range(K):
        tp = clusters[k][:,:,:T-1]
        tpp = clusters[k][:,:,1:]
        xxt = np.sum(np.matmul(tp, np.transpose(tp,(0,2,1))), axis=0)
        xpxt = np.sum(np.matmul(tpp, np.transpose(tp,(0,2,1))), axis=0)
        Ahat = np.matmul(xpxt, np.linalg.inv(xxt))
        Ahats.append(Ahat)
        
        tmpT = len(clusters[k])*T
        whats = tpp - np.matmul(Ahat, tp)
        What = 1/tmpT*np.sum(np.matmul(whats, np.transpose(whats, (0,2,1))), axis=0)
        Whats.append(What)

    return Ahats,Whats
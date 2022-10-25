#input:
    #listOfClusters: a list of clusters, each of same size K
    #for each clusters:
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
def model_estimation(listOfClusters):
    K = len(listOfClusters[0])
    Ahats = []
    Whats = []
    d = len(listOfClusters[0][0][0])
    
    for k in range(K):
        xxt = np.zeros((d,d))
        xpxt = np.zeros((d,d))
        for clusters in listOfClusters: 
            T = len(clusters[0][0][0])
            tp = clusters[k][:,:,:T-1]
            tpp = clusters[k][:,:,1:]
            xxt = xxt + np.sum(np.matmul(tp, np.transpose(tp,(0,2,1))), axis=0)
            xpxt = xpxt + np.sum(np.matmul(tpp, np.transpose(tp,(0,2,1))), axis=0)
        Ahat = np.matmul(xpxt, np.linalg.inv(xxt))
        Ahats.append(Ahat)
        
        tmpT = 0
        What = np.zeros((d,d))
        for clusters in listOfClusters:
            T = len(clusters[0][0][0])
            tmpT += len(clusters[k])*(T-1)
            tp = clusters[k][:,:,:T-1]
            tpp = clusters[k][:,:,1:]
            whats = tpp - np.matmul(Ahat, tp)
            What = What + np.sum(np.matmul(whats, np.transpose(whats, (0,2,1))), axis=0)
        What = 1/tmpT*What
        Whats.append(What)

    return Ahats,Whats

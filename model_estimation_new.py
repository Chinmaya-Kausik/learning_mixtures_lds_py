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
import numpy as np

def model_estimation(clusters):
    # initializing
    K = len(clusters)
    d = len(clusters[0][0])
    T = len(clusters[0][0][0])

    Ahats = []
    Whats = []
    
    for k in range(K):
        # computing Ahat
        xxt = np.zeros((d, d))
        xpxt = np.zeros((d, d))

        for m in range(clusters[k].shape[0]):
            tmp = clusters[k][m, :, :T-1]
            tmpp = clusters[k][m, :, 1:]
            xxt = xxt + tmp @ tmp.T
            xpxt = xpxt + tmpp @ tmp.T

        Ahat = xpxt @ np.linalg.inv(xxt)
        Ahats.append(Ahat)

        # computing What
        tmpT = 0
        What = np.zeros((d, d))

        for m in range(clusters[k].shape[0]):
            tmp = clusters[k][m, :, :T-1]
            tmpp = clusters[k][m, :, 1:]
            whats = tmpp - Ahat @ tmp
            What = What + whats @ whats.T
            tmpT = tmpT + tmp.shape[1]

        What = What / tmpT
        Whats.append(What)

    return Ahats, Whats

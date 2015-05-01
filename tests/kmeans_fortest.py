
import numpy as np
import numpy.random as nrd
#centroid
# def Centroid(Y):
#     """Y is a subset of the dataset"""
#     return np.average(Y,axis=0)

def Cost(C, Y):
    """C is a subset of the dataset, Y can be a point or a subset"""
    if  len(Y.shape)==1 or Y.shape[0]==1:
        #Y is a point
        MinIndex = np.argmin(np.sum((Y-C)**2,axis=1))
        return np.sum((Y-C[MinIndex,])**2)
    else:
        return np.sum([Cost(C,Y_i) for Y_i in Y])

def weight(C, data):
    """C is the centroid set and data is the target data set"""
    if len(C.shape)==1 or C.shape[0]==1:
        #C only have one point
        if len(data.shape)==1 or data.shape[0]==1:
            return np.array([1])
        else:
            return np.array([len(data)])
    else:
        #the cloest center for each point in data
        Index_min = [np.argmin(np.sum((x-C)**2,axis=1)) for x in data]
        #frequency for each center
        return np.array([Index_min.count(i) for i in range(len(C))]).astype(float)

def kmeanspar(k,l,r,data):
    """k is the number of centers, l is the expected number of intermediate points
    in each iteration, r is the number of iterations, data is the target data set"""
    #l*r should be larger than k in case k-means|| select too few points
    if l*r < k:
        raise ValueError('r or l must be bigger, ')
    #if k is too large
    if k >= len(data):
        raise ValueError('k is too large')
    #Step 1: choose one point randomly
    C = data[nrd.choice(range(len(data)),1),]
    #for loop
    for i in range(r):
        prob = [l*Cost(C,x) for x in data]/Cost(C,data)
        flag = nrd.uniform(size=len(data))
        C = np.concatenate((C,data[prob>=flag,]))
    #step 7
    weights = weight(C,data)
    #step 8: k-means++ to choose weighted points
    c = C[nrd.choice(range(len(C)),1),]
    while len(c) < k:
        p = np.array([Cost(c,x) for x in C])
        Prob = p*weights/np.sum(p*weights)
        x = nrd.choice(range(len(C)),1,p=Prob)
        c = np.concatenate((c,C[x,]))
    #change the return to only test initialization part
    return c

def Random(k,data):
    """k is the number of centers, data is target data"""
    if k >= len(data):
        raise ValueError('k is too large')
    return data[np.random.choice(len(data),k,replace=False),:]

def kmeansplus(k,data):
    if k >= len(data):
        raise ValueError('k is too large')
    #Step 1
    C = data[nrd.choice(range(len(data)),1),]
    #while loop
    while len(C) < k:
        prob = ([Cost(C,x) for x in data]/Cost(C,data)).reshape(len(data))
        x = nrd.choice(range(len(data)),1,p=prob)
        C = np.concatenate((C,data[x,]))
    return C
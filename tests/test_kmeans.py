
import numpy as np
import numpy.random as nrd
from numpy.testing import assert_equal
from kmeans import Cost,weight

# def test_Centroid_dimension():
   
# def test_Centroid_when_same():
    
# def test_Centroid_when_different():
    
# def test_Centroid_known():

def test_Cost_integer1():
    Y=np.array([1,2,3])
    C=np.array([[1,2,3],[3,4,2],[5,2,1]])
    assert Cost(C,Y) == 0
    
def test_Cost_integer1():
    Y=C=np.array([[1,2,3],[3,4,2],[5,2,1]])
    assert Cost(C,Y) == 0
    

def test_Cost_non_negative1():
    for i in range(10):
        Y=nrd.multivariate_normal([0,0,0],5*np.identity(3),4)
        C=nrd.multivariate_normal([0,0,0],5*np.identity(3),3)
        assert Cost(C,Y) >= 0
        
def test_Cost_non_negative2():
    for i in range(10):
        Y=nrd.multivariate_normal([0,0,0],5*np.identity(3),1)
        C=nrd.multivariate_normal([0,0,0],5*np.identity(3),3)
        assert Cost(C,Y) >= 0
    
def test_Cost_zero1():
    Y=C=nrd.multivariate_normal([0,0,0],5*np.identity(3),3)
    assert Cost(C,Y) == 0
    
def test_Cost_zero2():
    C=nrd.multivariate_normal([0,0,0],5*np.identity(3),3)
    Y=C[0,]
    assert Cost(C,Y) == 0
    
def test_Cost_known1():
    C1=nrd.multivariate_normal([0,0,0],5*np.identity(3),3)
    C2=C1[0:1,]
    Y=nrd.multivariate_normal([0,0,0],5*np.identity(3),4)
    assert Cost(C1,Y) <= Cost(C2,Y)
    
def test_weight_known1():
    C=nrd.multivariate_normal([0,0,0],5*np.identity(3),1)
    assert_equal(weight(C,C),np.array([1]))

def test_weight_known2():
    C=nrd.multivariate_normal([0,0,0],5*np.identity(3),3)
    assert_equal(weight(C,C),np.array([1,1,1]))

def test_weight_zero1():
    C=nrd.multivariate_normal([0,0,0],5*np.identity(3),5)
    X=C[0:4,]
    assert_equal(weight(C,X),np.array([1,1,1,1,0]))
    
def test_weight_zero1():
    C=nrd.multivariate_normal([0,0,0],5*np.identity(3),5)
    X=C[1:5,]
    assert_equal(weight(C,X),np.array([0,1,1,1,1]))
    
def test_weight_integer1():
    X=np.array([[1,1,1],[2,3,4],[1,1,1]])
    C=X[0:2,]
    assert_equal(weight(C,X),np.array([2,1]))
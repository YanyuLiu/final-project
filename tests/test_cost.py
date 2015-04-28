
import numpy as np
import numpy.random as nrd
from numpy.testing import assert_equal
from kmeans import Cost

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
    
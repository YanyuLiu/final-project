
import numpy as np
import numpy.random as nrd
from numpy.testing import assert_equal
from kmeans_fortest import Cost,weight

def test_weight_one_point():
    C=nrd.multivariate_normal([0,0,0],5*np.identity(3),3)
    assert_equal(weight(C[0,],C),np.array([3]))
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
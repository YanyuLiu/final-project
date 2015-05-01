
import numpy as np
import numpy.random as nrd
from numpy.testing import assert_equal,assert_raises
from kmeans_fortest import Cost,weight,kmeanspar

k = 5
centers = nrd.multivariate_normal([0,0,0],5*np.identity(3),k)
data = [nrd.multivariate_normal(center, np.identity(3),10) for center in centers]
data = np.vstack(data)

def test_kmeanspar_len1():
    assert_equal(len(kmeanspar(2,2,5,data)),2)

def test_kmeanspar_len2():
    assert_equal(len(kmeanspar(2,1,5,data)),2)

def test_kmeanspar_len3():
    assert_equal(len(kmeanspar(5,3,5,data)),5)

def test_kmeanspar_inside1():
    assert_equal(all([d in data for d in kmeanspar(5,1,10,data)]),True)
    
def test_kmeanspar_inside2():
    assert_equal(all([d in data for d in kmeanspar(10,3,5,data)]),True)
    
def test_kmeanspar_throws_exception():
    assert_raises(ValueError,kmeanspar,500,100,100,data)
    
def test_kmeanspar_throws_exception2():
    assert_raises(ValueError,kmeanspar,5,1,3,data)
    
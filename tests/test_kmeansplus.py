import numpy as np
import numpy.random as nrd
from numpy.testing import assert_equal,assert_raises
from kmeans_fortest import Cost,kmeansplus

k = 5
centers = nrd.multivariate_normal([0,0,0],5*np.identity(3),k)
data = [nrd.multivariate_normal(center, np.identity(3),10) for center in centers]
data = np.vstack(data)

def test_kmeansplus_len1():
    assert_equal(len(kmeansplus(10,data)),10)

def test_kmeansplus_len2():
    assert_equal(len(kmeansplus(7,data)),7)

def test_kmeansplus_inside1():
    assert_equal(all([d in data for d in kmeansplus(5,data)]),True)
    
def test_kmeansplus_inside2():
    assert_equal(all([d in data for d in kmeansplus(1,data)]),True)
    
def test_kmeansplus_throws_exception():
    assert_raises(ValueError,kmeansplus,100,data)
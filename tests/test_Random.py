
import numpy as np
import numpy.random as nrd
from numpy.testing import assert_equal,assert_raises
from kmeans_fortest import Random

k = 5
centers = nrd.multivariate_normal([0,0,0],5*np.identity(3),k)
data = [nrd.multivariate_normal(center, np.identity(3),10) for center in centers]
data = np.vstack(data)

def test_Random_len():
    assert_equal(len(Random(10,data)),10)

def test_Random_inside():
    assert_equal(all([d in data for d in Random(5,data)]),True)
    
def test_Random_throws_exception():
    assert_raises(ValueError,Random,500,data)
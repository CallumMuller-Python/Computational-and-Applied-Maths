# Initialisation Cell
from matplotlib import pyplot as plt
from IPython.display import display, HTML, Javascript
from math import *
import numpy as np
import numpy.testing as nt
import math


def bisection_method(f, g, interval, n, *eps):
    a=interval[0]
    b=interval[1]
    for i in range(1,n+1):
        if(abs(a-b)<=eps):
            return [a,b]
        t=(a+b)/2
        f_t=f(t)
        if((f(a)-f_t)>(f(b)-f_t)):
            a=t
            b=b
        else:
            a=a
            b=t
    if(a==b):
        return f(a)
    return [a,b]


f = lambda x: 3*x**2 - 4*x+1
g = lambda x: 6*x-4
interval = np.array([0, 2])
n = 10
eps = 0.001
print(bisection_method(f, g, interval, n, eps))

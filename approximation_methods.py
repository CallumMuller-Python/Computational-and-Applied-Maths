import math
from math import log, exp,sin,cos,e
from numpy import arange, zeros, log, exp, pi



def trapezoidal(f, n, a, b):
    h = (b - a) / n
    xvals = a + arange(n + 1) * h
    summ = 0.5 * h * (f(xvals[0]) + f(xvals[n]))
    for i in arange(1, n):
        summ = summ + h * f(xvals[i])
    return summ


def simpsons(f, n, a, b):
    h = (b - a) / n
    xvals = a + arange(n + 1) * h
    summ = (1/3) * h * (f(xvals[0]) + f(xvals[n]))
    for i in arange(1, n):
        if (i%2==0):
            summ = summ + (2/3)*h * f(xvals[i])
        else:
            summ = summ + (4/3)*h * f(xvals[i])
    return summ



def smallestN(f,a,b):
    smaller=0.00001
    n=1
    while(abs(trapezoidal(f,n,a,b)-simpsons(f,n,a,b))>smaller):
        n=n+1
    return n


def erf(x):
    p = lambda x: (e ** ((-x ** 2)))
    return (2/math.sqrt(pi))*midpoint(p,smallestNQ6(p,0,x),0,x)


def midpoint(f, n, a, b):
    h = (b - a) / n
    xvals = a + arange(n + 1) * h
    summ = h * (f((xvals[0]+xvals[1])/2))
    for i in arange(1, n):
        summ = summ + h * (f((xvals[i]+xvals[i+1])/2))
    return summ



def intervalLength():
    p = lambda x: (e ** ((-x ** 2)))
    return (0.5-0)/smallestNQ6(p,0,0.5)


def smallestNQ6(f,a,b):
    smaller=0.000001
    n=1
    while(abs(trapezoidal(f,n,a,b)-simpsons(f,n,a,b))>smaller):
        n=n+1
    return n


f = lambda x: 1 / (1 + x ** 2)
g = lambda x: (cos(x)*log(sin(x)))/((sin(x)**2)+1)
h = lambda x: 1/math.sqrt(1+pow(e,x)-x)
p = lambda x: (e ** ((-x ** 2)))
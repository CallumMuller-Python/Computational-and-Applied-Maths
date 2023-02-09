def golden_search(f, a, b, tol):
    p = (3- sqrt(5)) / 2
    if (b - a < tol):
        return ([a + p * (b - a), f(a + p * (b - a))])
    a1 = a + p * (b - a)
    b1 = a + (1 - p) * (b - a)
    if (f(a1) < f(b1)):
        return golden_search(f, a, b1, tol)
    else:
        return golden_search(f, a1, b, tol)

def exam(x):
    y=0
    for i in range(1,x+1):
        y=y+237/pow(10,3*i);
    return y
f = lambda x: x**4 - 14*x**3 + 60*x**2 - 70*x
a = 0
b = 2
tol = 10**(-5)

f = lambda x: x**4 - 14*x**3 + 60*x**2 - 70*x
a = 0
b = 2
tol = 10**(-5)
nt.assert_array_almost_equal(np.array([0.78088186, -24.36960157]), golden_search(f, a, b, tol), err_msg='incorrect function')
print('All Tests Passed!!!')
print((1+exam(20)).as_integer_ratio())
print(79/333)
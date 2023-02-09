
import numpy
import numpy as np
import matplotlib.pylab as plt


def Hilbert(n):
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i][j] = 1 / (i + j + 1)
    return H

def Hilbert_main(A):
    sum=0
    for i in range(1000):
        sum=sum+A[i][i]
    return sum

def Hilbert_up(A):
    sum = 0
    for i in range(999):
        sum = sum + A[i][i+1]
    return sum

def Hilbert_down(A):
    sum = 0
    for i in range(1,1000):
        sum = sum + A[i][i-1]
    return sum

def Power_method(A,x0,n):
    u0=get_u0(x0)
    x1=np.dot(A,u0)
    for i in range (n-1):
        u0=get_u0(x1)
        x1=np.dot(A,u0)
    return np.dot(x1,u0), x1

def get_u0(x0):
    x=[]
    sum=0
    for j in range(len(x0)):
        sum=sum+pow(x0[j],2)
    for i in range(len(x0)):
        x.append(1/np.sqrt(sum))
    return x


def Dom_Eigen(A):
    x=[]
    for i in range(123):
        x.append(1)
    return Power_method(A,x,10)[0]


def Newton(F,J,x0,tol):
    x=len(J[0])
    y=len(J)
    n=0
    x1=x0
    while (tol>get_tol(x0,x1)):
        Jx0=numpy.zeros(shape=(x,y))
        Fx0 = []
        for i in range(len(F)):
            Fx0.append(F[i](x0[0],x0[1]))
        for i in range(x):
            for j in range(y):
                Jx0[i][j]=J[i][j](x0[0],x0[1])
        invJ = np.linalg.inv(Jx0)
        X = x0 - np.dot(invJ, Fx0)
        x1=x0
        x0=X
    return x0

def get_tol(x1,x0):
    return (normalize(np.subtract(x1,x0))/normalize(x0))

def normalize(x0):
    x=[]
    sum=0
    for j in range(len(x0)):
        sum=sum+pow(x0[j],2)
    y=np.sqrt(sum)
    return y

def generate_x(a,h,n):
    x=[]
    for i in range(n-1):
        x.append(a+i*h)
    return x
def Newton_Q6(F,J,x0,tol):
    x = len(J[0])
    y = len(J)
    n = 0
    x1 = x0
    while (tol > get_tol(x0, x1)):
        Jx0 = numpy.zeros(shape=(x, y))
        Fx0 = []
        for i in range(len(F)):
            Fx0.append(F[i](x0[i], x0[i+1],x0[i+2],generate_x(1,1/3,6)[i]))
            print(F[i](x0[i], x0[i+1],x0[i+2],generate_x(1,1/3,6)[i]))
        for i in range(x):
            for j in range(y):
                Jx0[i][j] = J[i][j](x0[0], x0[1])
        invJ = np.linalg.inv(Jx0)
        X = x0 - np.dot(invJ, Fx0)
        x1 = x0
        x0 = X
    return x0



def plot(x,y):
    plt.plot(x,y)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    plt.title('Yi vs Xi')
    plt.show()
def det(X):
    return (X[0][0]*X[1][1])-(X[0][1]*X[1][0])
def trace(X):
    return X[0][0]+X[1][1]
def discriminant(X):
    return (trace(X)**2)-4*det(X)




x=[1,1]
generate_x(1,1/3,6)
A = np.array([[1,2],[3,4]])
Hil123=Hilbert(123)
x0=[-0.6,0.6]
H=lambda x: 2*x
F=lambda x,y: (x**3)-3*(x*y**2)-1
G=lambda x,y: 3*(x**2)*y-y**3
fdx=lambda x,y: 3*(x**2)-3*y**2
fdy=lambda x,y: -6*x*y
gdx=lambda x,y: 6*x*y
gdy=lambda x,y: 3*(x**2)-3*y**2
N1=lambda y0,y1,y2,x1: -9*y2 -18*y1+9*17+4+0.25*(x1**3)-y1*(((2*y2)-17)/3)
N2=lambda y1,y2,y3,x2: -9*y3 -18*y2+9*y1+4+0.25*(x2**3)-y2*(((2*y3)-y1)/3)
N3=lambda y2,y3,y4,x3: -9*y4 -18*y3+9*y2+4+0.25*(x3**3)-y3*(((2*y4)-y2)/3)
N4=lambda y3,y4,y5,x4: -9*y5 -18*y4+9*y3+4+0.25*(x4**3)-y4*(((2*y5)-y3)/3)
N5=lambda y4,y5,y6,x5: -9*(43/3) -18*y5+9*y4+4+0.25*(x5**3)-y5*(((2*(43/3))-y4)/3)

dxN1=lambda x,y2: ((17-y2)/12)+2/9
dyN1=lambda x,y1: -(y1/12)-1/9
dxN2=lambda x,y2: (y2/12)-1/9
dyN2=lambda y1,y3: ((y1-y3)/12)+2/9
dzN2=lambda x,y2: (-y2/12)-1/9
dyN3=lambda x,y3: (y3/12) -1/9
dzN3=lambda y2,y4: ((y2-y4)/12)+2/9
daN3=lambda x,y3: (-y3/12)-1/9
dzN4=lambda x,y4: (y4/12)-1/9
daN4=lambda y3,y5: ((y3-y5)/12)+2/9
dbN4=lambda x,y4: (-y4/12)-1/9
dbN5=lambda x,y5: (y5/12)-1/9
dcN5=lambda x,y4: ((y4-43/3)/12)+2/9
J=np.array([[fdx,fdy],[gdx,gdy]])
N=np.array([N1,N2,N3,N4,N5])
Njac=np.array([[dxN1,dyN1,0,0,0],
              [dxN2,dyN2,dzN2,0,0],
              [0,dyN3,dzN3,daN3,0],
              [0,0,dzN4,daN4,dbN4],
              [0,0,0,dbN5,dcN5]])
X=[17,17,17,17,17,17,17]
z = [1,2,3]
y = [2,4]
B = np.array([[0,1],
              [-1,3]])
print("Det:",det(B))
print("Trace:",trace(B))
print("Dis:",discriminant(B))

Funcvec=[]
Funcvec.append(F)
Funcvec.append(G)
Y=[]


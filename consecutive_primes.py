def test_prime(n):
    if (n==1):
        return False
    elif (n==2):
        return True
    else:
        for x in range(2,n):
            if(n % x==0):
                return False
        return True
prime_array=[]
input=int(input())
for i in range(2,input):
    if (test_prime(i)):
        prime_array.append(i)
print(prime_array)
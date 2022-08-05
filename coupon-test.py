#!/usr/bin/python3

import random
import numpy as np
import math
import matplotlib.pyplot as plt

N_UNIQUE=100
M=N_UNIQUE*50
J=N_UNIQUE*50
res=[]


# for i in range(M):
#     got=[False]*N_UNIQUE
#     to_get=N_UNIQUE
#     visits = 0
#     while (to_get > 0):
#         get=random.randrange(0,N_UNIQUE)
#         if got[get] == False:
#             got[get] = True
#             to_get -= 1
#         visits += 1
#     res.append(visits)

res = []
x = {}
y = [0] * M
for i in range(M):
    x[i] = [[0] * N_UNIQUE, 0]
for j in range(J):
    y = [0] * M
    for i in range(M):
        if x[i][1] == N_UNIQUE:
            y[i] += 1
            continue
        r = random.randrange(0, N_UNIQUE)
        v = x[i][0][r]
        if (v == 0):
            x[i][0][r] = 1
            x[i][1] += 1
            if x[i][1] == N_UNIQUE:
                y[i] += 1
    res.append(sum(y))
    if (j % 100 == 0):
        print(j, end="", flush=True)
    else:
        print(".",end="", flush=True)
        
p = np.array(np.array(res)/M)

# def f(n, w):
#     if (n > w * np.log(w)):
#         return 1
#     else:
#         return 0

def g(j, n):
    res = []
    for i in j:
        if (i < n):
            res.append(0)
        else:
            newval = 1
            for m in range(1,n):
                newval -= math.comb(n, m)*(1-m/n)**i
            res.append(newval)
    return res

xs = np.arange(J)
gs = g(xs, N_UNIQUE)
plt.plot(xs, p, xs, xs > N_UNIQUE * np.log(N_UNIQUE), xs, gs)
plt.savefig('coupon-test.png')
print(f"Max of g: {np.max(gs)}, Min of g: {np.min(gs)}")
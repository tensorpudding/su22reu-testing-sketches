#!/usr/bin/python3

import random
import numpy as np
import matplotlib.pyplot as plt

X=100
M=100
x = range(1,X)
res=[]

for j in x:
    run = []
    for i in range(M):
        empty=[True]*j
        for i in range(j):
            bin = random.randrange(0,j)
            empty[bin] = False
        run.append(np.sum(empty))
    res.append(np.mean(run))
    if (j % 100 == 0):
        print(j, end="", flush=True)
    else:
        print(".",end="", flush=True)

def f(n):
    return n * (1.0 - np.reciprocal(n)) ** n

np.vectorize(f)

plt.plot(x, res, x, f(x), x, x/np.exp(1))
plt.savefig("bin-test.png")

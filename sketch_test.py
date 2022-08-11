#!/usr/bin/env python3

import hashing_families
import numpy as np

def main():
    a1 = [0,0,1,0,0,-1,1,0]
    a2 = [0,0,0,1,0,1,-1,0]
    a12 = [0,0,1,1,0,0,0,0]
    n = len(a1)
    delta = 0.1
    h = hashing_families.HashFunction(2, n, n**3)
    seed = np.random.default_rng().integers(0,100)
    print(seed)
    M1 = [hashing_families.SSampler(n, h, 333667, j, err=0.05, seed=seed) for j in range(int(np.log(n)))]
    M2 = [hashing_families.SSampler(n, h, 333667, j, err=0.05, seed=seed) for j in range(int(np.log(n)))]
    M12 = [hashing_families.SSampler(n, h, 333667, j, err=0.05, seed=seed) for j in range(int(np.log(n)))]
    for i in range(n):
        for j in range(len(M1)):
            M1[j].update((i, a1[i]))
            M2[j].update((i, a2[i]))
            M12[j].update((i, a12[i]))
    res1 = None
    res2 = None
    res12 = None
    for j in range(len(M1)):
        if res1 == None:
            res1 = M1[j].complete()
        if res2 == None:
            res2 = M2[j].complete()
        if res12 == None:
            res12 = M12[j].complete()
    print(f"b1: {res1}, b2: {res2}, b12: {res12}")

if __name__=='__main__':
    main()


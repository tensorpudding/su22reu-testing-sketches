#!/usr/bin/env python3

import hashing_families
import numpy as np
import sys

def main():
    if (len(sys.argv) > 1):
        seed = int(sys.argv[1])
    else:
        seed = 172
    # s1 = hashing_families.OneSparseRecover(333667, 125)
    # s2 = hashing_families.OneSparseRecover(333667, 125)
    # s1.update(5, 1)
    # s2.update(5, -1)
    # s1.update(2, 1)
    # s2.update(5, 1)
    # s1.update(2, -1)
    # s2.update(4, 1)
    # s3 = s1 + s2
    # print(s1.recover())
    # print(s2.recover())
    # print(s3.recover())

    # ss1 = hashing_families.SSparseRecover(2, 5, 333667, err=0.05, seed=125)
    # ss2 = hashing_families.SSparseRecover(2, 5, 333667, err=0.05, seed=125)
    # print(ss1 == ss2)
    # ss1.update(0, 5)
    # ss2.update(0, -5)
    # ss1.update(1, 3)
    # ss2.update(1, -3)
    # ss3 = ss1 + ss2
    # print(ss1.recover())
    # print(ss2.recover())
    # print(ss3.recover())
    rng = np.random.default_rng(seed)
    seed = rng.integers(2**32 - 1)
    a1 = [0,3,2,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    a2 = [0,0,0,-1,9,2,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]
    n = len(a1)
    a_gen = hashing_families.gen_a(20, n, 0.5, seed=seed)
    LZs = []
    NUM_SKETCH=1000
    for sketch in range(NUM_SKETCH):
        LZs.append(hashing_families.L0Sketch(n, 333667))
    for i, delta in a_gen:
        for sketch in range(NUM_SKETCH):
            LZs[sketch].update(i, delta)
    print(f"Samples from random a: ", end="", flush=True)
    hist = np.zeros((n,))
    miss = 0
    for sketch in range(NUM_SKETCH):
        s = LZs[sketch].sample()
        if s == None:
            miss += 1
        else:
            hist[s[0]] += 1
    print(hist)
    print("")
    print(f"Failures: {miss} (failure rate = {100*miss/NUM_SKETCH}%)")
    return
    ROUNDS=30
    hist_1 = np.zeros((n+1,))
    hist_2 = np.zeros((n+1,))
    hist_3 = np.zeros((n+1,))
    hist_12 = np.zeros((n+1,))
    for i in range(ROUNDS):
        #print(f"Round {i}: seed = {seed}")
        LZ1 = hashing_families.L0Sketch(n, 333667, seed=seed)
        LZ2 = hashing_families.L0Sketch(n, 333667, seed=seed)
        LZ12 = hashing_families.L0Sketch(n, 333667, seed=seed)
        for i in range(len(a1)):
            LZ1.update(i, a1[i])
            LZ2.update(i, a2[i])
            LZ12.update(i, a1[i])
            LZ12.update(i, a2[i])
            LZ3 = LZ1 + LZ2
            # print(LZ1.debug_print())
        if LZ1.sample():
            hist_1[LZ1.sample()[0]] += 1
        else:
            hist_1[n] +=1
        if LZ2.sample():
            hist_2[LZ2.sample()[0]] += 1
        else:
            hist_2[n] +=1
        if LZ3.sample():
            hist_3[LZ3.sample()[0]] += 1
        else:
            hist_3[n] +=1
        if LZ12.sample():
            hist_12[LZ12.sample()[0]] += 1
        else:
            hist_12[n] +=1
        seed = rng.integers(2**32 - 1)
    print(hist_1)
    print(hist_2)
    print(hist_3)
    print(hist_12)
    # a1 = [0,0,1,0,0,-1,1,0]
    # a2 = [0,0,0,1,0,1,-1,0]
    # a12 = [0,0,1,1,0,0,0,0]
    # n = len(a1)
    # delta = 0.1
    # h = hashing_families.HashFunction(2, n, n**3)
    # seed = np.random.default_rng().integers(0,100)
    # print(seed)
    # M1 = [hashing_families.SSampler(n, h, 333667, j, err=0.05, seed=seed) for j in range(int(np.log(n)))]
    # M2 = [hashing_families.SSampler(n, h, 333667, j, err=0.05, seed=seed) for j in range(int(np.log(n)))]
    # M12 = [hashing_families.SSampler(n, h, 333667, j, err=0.05, seed=seed) for j in range(int(np.log(n)))]
    # for i in range(n):
    #     for j in range(len(M1)):
    #         M1[j].update((i, a1[i]))
    #         M2[j].update((i, a2[i]))
    #         M12[j].update((i, a12[i]))
    # res1 = None
    # res2 = None
    # res12 = None
    # for j in range(len(M1)):
    #     if res1 == None:
    #         res1 = M1[j].complete()
    #     if res2 == None:
    #         res2 = M2[j].complete()
    #     if res12 == None:
    #         res12 = M12[j].complete()
    # print(f"b1: {res1}, b2: {res2}, b12: {res12}")

if __name__=='__main__':
    main()


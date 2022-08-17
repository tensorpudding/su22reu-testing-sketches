from sparse_recover import OneSparseRecover, SSparseRecover
from hash import HashFunction
import numpy as np
import sys

class L0Sketch:
    def __init__(self, n, p=None, err=0.05, seed=None):
        if seed == None:
            self.seed = np.random.default_rng().integers(2000)
        else:
            self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        self.n = n
        x = self.n ** 3
        if x < 147:
            self.p = 147
        elif x < 333667:
            self.p = 333667
        elif x < 9999991:
            self.p = 9999991
        elif x < 2147483647:
            self.p = 2147483647
        elif x < 272914271260019:
            self.p = 272914271260019
        else:
            sys.exit(1)
        self.err = err
        self.s = int(np.log(1/err))+1
        self.k = 2*self.s
        self.j_max = int(np.log(self.n**2))
        self.h = HashFunction(self.k, self.n**2, self.n**6, seed=self.rng.integers(3000))
        self.levels = []
        for j in range(self.j_max):
            self.levels.append(SSparseRecover(self.s, self.n**2, self.p, self.err, seed=self.rng.integers(3000)))
    
    def update(self, i, j, delta):
        """
        Update the sketch by adding an edge (i,j) if delta = 1 or removing an edge (i,j) if delta = -1
        """
        assert(i != j)
        index1 = i*self.n + j
        index2 = j*self.n + i
        hi = self.h.eval(index1)
        j = 0
        while hi < int((self.n ** 6) / (2**j)) and j < self.j_max:
            if delta == 1:
                self.levels[j].update(index1, 1)
                self.levels[j].update(index2, -1)
            else:
                self.levels[j].update(index1, -1)
                self.levels[j].update(index2, 1)
            j+=1

    def recover(self):
        j = 0
        while j < self.j_max:
            got = self.levels[j].recover()
            if got:
                return got
            else:
                j += 1
        return None
    
    def sample(self):
        got = self.recover()
        if got:
            lowest_index = got[0][0]
            lowest_hash = self.h.eval(got[0][0])
            lowest_delta = got[0][1]
            for j in range(1,len(got)):
                index, delta = got[j]
                hash = self.h.eval(index)
                if hash < lowest_hash:
                    lowest_hash = hash
                    lowest_delta = delta
                    lowest_index = index
            if lowest_delta >= 1:
                j = lowest_index % self.n
                i = int(lowest_index / self.n)
                return (i,j,lowest_delta)
            else:
                i = lowest_index % self.n
                j = int(lowest_index / self.n)
                return (i,j,-lowest_delta)
        else:
            return None
    
    def debug_print(self):
        print(f"L0 sampler: n = {self.n}, s={self.s}, k={self.k}, j_max={self.j_max}, seed={self.seed}")
        print(f"h: p = {self.h.p}, m = {self.h.m}, cs={self.h.cs} -> {list(self.h.get_hash_vals())}")
        for j in range(self.j_max):
            print(f"j = {j}: ")
            r = 0
            for h in self.levels[j].hs:
                print(f"r = {r}, p = {h.p}, cs= {h.cs} -> {list(h.get_hash_vals())}")
                r += 1

    
    def __eq__(self, other):
        return self.seed == other.seed and self.n == other.n and self.p == other.p and self.err == other.err

    def __add__(self, other):
        assert(self == other)
        ret = L0Sketch(self.n, self.p, self.err, self.seed)
        for j in range(len(self.levels)):
            ret.levels[j] = self.levels[j] + other.levels[j]
        return ret

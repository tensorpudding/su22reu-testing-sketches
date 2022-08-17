import numpy as np
import sys

class HashFunction:
    """
    Class to represent k-wise independent hash functions.
    Implemented by analyzing polynomials over a finite field, with coefficients uniformly sampled from the field.
    By Carter and Wegman (1981), these produce strong universal_k hash functions.
    This means that for a random hash function onto the set [m], and any set of up to k distinct keys, the probability of a collision is bounded by 1/m^k
    We assume here that k << m, this does not work well at all otherwise.
    """
    def __init__(self, k, n, m, seed):
        """
        Constructor for hash function.
        k = k for k-wise independence
        n = size of input set
        m = size of output set
        seed = a seed for the RNG, to allow for deterministic generation (None = random)
        """
        # We generate the large modulus that produces our finite field of coefficients.
        # A finite field requires order of a prime, or power of a prime
        # We set p initially to be an arbitrary prime larger than both n and m
        self.p = 147
        # We consider the possibility that the hash space is either larger or smaller than the space of keys
        # The size of the base field must be larger than either
        x = max(n, m)
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
            print(f"Error: cannot generate prime large enough (larger than {x})", flush=True, file=sys.stderr)
            sys.exit(1)
        self.k = k
        self.n = n
        self.m = m
        if seed == None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(seed=seed)
        # Here, we generate the random coefficients of our degree k-1 polynomial
        self.cs = rng.integers(0, self.p, size=self.k)

    def eval(self, x):
        # Evaluate the hash function at x
        # We assume that x has already been normalized modulo n
        res = self.cs[0]
        xi = 1
        for i in range(1,self.k):
            xi = (xi * x) % self.p
            res += (self.cs[i] * xi) % self.p
            #res = res % self.p
        return res % (self.m)
    def get_hash_vals(self):
        for i in range(self.n):
            yield self.eval(i)
    def get_cs(self):
        return self.cs
    def set_cs(self, *args):
        if len(args) != len(self.cs):
            #print("Error: too few arguments for set_cs", file=sys.stderr)
            sys.exit(2)
        for i in range(len(self.cs)):
            self.cs[i] = args[i]
    def __eq__(self, other):
        return self.p == other.p and self.k == other.k and self.m == other.m and self.n == other.n

def test_independence(k, n, m, rounds, seed=None):
    if seed == None:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()
    hist = np.zeros((m,))
    seeds = rng.integers(0, 2**32, size=rounds)
    for i in range(rounds):
        h = HashFunction(k, n, m, seed=seeds[i])
        hist[h.eval(0)] += 1
    return ((hist / rounds) - (1 / m))/(1 / m)
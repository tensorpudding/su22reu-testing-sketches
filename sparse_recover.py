import numpy as np
from hash import HashFunction

class OneSparseRecover:
    def __init__(self, p, seed=None):
        self.seed = seed
        if seed == None:
            self.seed = np.random.default_rng().integers(3000)
        self.phi = 0
        self.tau = 0
        self.iota = 0
        self.p = p
        self.z = np.random.default_rng(self.seed).integers(0, p)
        self.is_one_sparse = False

    def update(self, i, delta):
        self.phi += delta
        self.iota += i*delta
        if i != 0:
            self.tau += delta * field_power(self.z, i, self.p)
        else:
            self.tau += delta
        self.tau %= self.p
        self.set_sparsity()
    
    def set_sparsity(self):
        if (self.phi != 0):
            r = (self.iota / self.phi) % (self.p - 1)
            zr = field_power(self.z, r, self.p) % self.p
            x = (self.phi * zr) % self.p
            self.is_one_sparse = x == self.tau
        else:
            self.is_one_sparse = False
    def recover(self):
        if self.is_one_sparse:
            #print(f"OneSparseRecover: recovered {(int(self.iota / self.phi), self.phi)}")
            return (int(self.iota / self.phi), self.phi)
        else:
            return None

    def __eq__(self, other):
        return self.p == other.p and self.z == other.z

    def __add__(self, other):
        assert(self == other)
        ret = OneSparseRecover(self.p, self.seed)
        ret.tau = self.tau + other.tau
        ret.phi = self.phi + other.phi
        ret.iota = self.iota + other.iota
        ret.set_sparsity()
        return ret

class SSparseRecover:
    def __init__(self, s, n, p, err=0.05, seed=None):
        self.s = s
        self.n = n
        self.p = p
        self.r_max = int(np.log(s/err))
        self.err = err
        self.hs = []
        if seed == None:
            self.seed = np.random.default_rng().integers(3000)
        else:
            self.seed = seed
        rng = np.random.default_rng(self.seed)
        h_seeds = rng.integers(3000, size=self.r_max)
        for i in range(self.r_max):
            self.hs.append(HashFunction(2, n, 2*s, seed=h_seeds[i]))
        self.ps = []
        for i in range(self.r_max):
            self.ps.append([])
            seeds = rng.integers(3000, size=2*s)
            for j in range(2*s):
                self.ps[i].append(OneSparseRecover(p, seeds[j]))
    
    def update(self, i, delta):
        for j in range(self.r_max):
            self.ps[j][self.hs[j].eval(i)].update(i, delta)
    
    def recover(self):
        a_prime = []
        for j in range(self.r_max):
            for k in range(2*self.s):
                res = self.ps[j][k].recover()
                if res != None and res not in a_prime:
                    #print(f"Recovered {res}")
                    a_prime.append(res)
        self.is_s_sparse = (len(a_prime) <= self.s) and (len(a_prime) >= 1)
        if self.is_s_sparse:
            return a_prime
        else:
            return None

    def __eq__(self, other):
        if self.s != other.s or self.r_max != other.r_max:
            return False
        for i in range(len(self.hs)):
            if self.hs[i] != other.hs[i]:
                return False
        for i in range(self.r_max):
            for j in range(2*self.s):
                if self.ps[i][j] != other.ps[i][j]:
                    return False
        return True

    def __add__(self, other):
        assert(self == other)
        ret = SSparseRecover(self.s, self.n, self.p, self.err, self.seed)
        for i in range(self.r_max):
            for j in range(2*self.s):
                ret.ps[i][j] = self.ps[i][j] + other.ps[i][j]
        return ret

def field_power(z, e, p):
    """
    Returns z^e (mod p) through efficient repeated squaring
    This assumes that z is an element of Z/pZ and e has been reduced (mod p-1) as per Fermat's Little Theorem
    """
    # First save any fractional part of e
    r = e - int(e)
    # We only care about the integral part
    q = int(e)
    if q == 0:
        return (z ** r) % p
    zes = z % p
    zq = 1
    while q != 0:
        if q % 2 == 1:
            zq *= zes
            zq %= p
        q >>= 1
        zes = (zes ** 2) % p
    return ((z ** r) * zq) % p

import numpy as np
import sys

class HashFunction:
    def __init__(self, p, k, n, m=None, seed=None):
        self.p = p
        self.k = k
        self.n = n
        if m:
            self.m = m
        else:
            self.m = n ** 3
        assert(self.p > self.m)
        if seed:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()
        self.cs = rng.integers(0, p, size=self.k)

    def eval(self, x):
        res = self.cs[0]
        pk = 1
        for i in range(1,self.k):
            res += self.cs[i] * x ** pk
            pk += 1
            res = res % self.p
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

def get_sample_indices(n, h, j, m):
    """
    Return a list of a(j) for j from 0 to j_max. Each is a sparse vector, for which a(j)[i] = a[i] iff h(i) > n^3 * 2^-j
    """
    h_gen = list(h.get_hash_vals())
    aj = []
    for i in range(len(h_gen)):
        if m / (2 ** (j)) >= h_gen[i]:
            aj.append(i)
    #print(f"Length of j={j}: {len(aj)}")
    #print(f"Hashvals: {h_gen} Indices: {aj}")
    return aj

class OneSparseRecover:
    def __init__(self, p, seed=None):
        if seed:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        self.phi = 0
        self.tau = 0
        self.iota = 0
        self.p = p
        self.z = rng.integers(0, p)
        self.is_one_sparse = False

    def update(self, i, delta):
        self.phi += delta
        self.iota += i*delta
        if i != 0:
            self.tau += delta * field_power(self.z, i, self.p)
        else:
            self.tau += delta
        self.tau %= self.p
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

class SSparseRecover:
    def __init__(self, s, n, p, err=0.05, seed=None):
        self.s = s
        self.r_max = int(np.log(s/err))
        self.hs = []
        if seed:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng(seed)
        for i in range(self.r_max):
            self.hs.append(HashFunction(p, 2, n, m=2*s, seed=rng.integers(0,20000)))
        self.ps = []
        for i in range(self.r_max):
            self.ps.append([])
            for j in range(2*s):
                self.ps[i].append(OneSparseRecover(p))
    
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
    # We need the binary expansion of q
    # ds stores this
    d = int(np.log2(q))
    ds = [0] * (d+1)
    while q > 0:
        d = int(np.log2(q))
        ds[d] = 1
        q -= 2**d
    # zq is z^q, which is determined by repeated squaring z, and multiplying powers of z to multiples of 2 present in q's binary expansion
    zq = z ** ds[0]
    # zes keeps track of z^(2^i), zq is multiplied by zes if the ith digit is 1 in q's binary expansion
    zes = z
    for i in range(1,len(ds)):
        zes = (zes ** 2) % p
        if ds[i] == 1:
            zq *= zes
            zq %= p
    return ((z ** r) * zq) % p

class SimpleSampler:
    def __init__(self, n, h, p, j, seed=None):
        self.n = n
        self.j = j
        if seed:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()
        self.osr = OneSparseRecover(p, seed=seed)
        indices = get_sample_indices(n, h, j, n ** 3)
        self.sample = {}
        for index in indices:
            self.sample[index] = 0

    def update(self, input):
        #print(f"INPUT IS {input}!!!!!")
        i = input[0]
        if i in self.sample:
            self.sample[i] += input[1]
            self.osr.update(i, input[1])
        self.is_one_sparse = self.osr.is_one_sparse

    def complete(self):
        #print(f"Sampler = {self.sample}")
        s = sum(self.sample.values())
        #print(f"This sampler on j = {self.j} tracks {len(self.sample.keys())} indices")
        #print(f"This sampler on j = {self.j} is one-sparse: {self.is_one_sparse}")
        try:
            assert((s == 1) == self.is_one_sparse)
        except(AssertionError):
            print(f"ASSERTION FAILED: {self.sample} but claims that \"is 1-sparse\" = {self.is_one_sparse}", file=sys.stderr)
        #print(f"z = {self.osr.z}, phi = {self.osr.phi}, tau = {self.osr.tau}, iota = {self.osr.iota}")

class SSampler:
    def __init__(self, n, h, p, j, err=0.05, seed=None):
        self.n = n
        self.j = j
        self.s = int(np.log(1/err))
        if seed:
            rng = np.random.default_rng(seed=seed)
        else:
            rng = np.random.default_rng()
        self.ssr = SSparseRecover(self.s, n, p, err=err, seed=seed)
        indices = get_sample_indices(n, h, j, n ** 3)
        self.sample = {}
        for index in indices:
            self.sample[index] = 0
    
    def update(self,input):
        i = input[0]
        if i in self.sample:
            self.sample[i] += input[1]
            self.ssr.update(i, input[1])

    def complete(self):
        #print(f"Sampler {self.j} = {self.sample}")
        a_prime = self.ssr.recover()
        #print(f"Sampler {self.j} returns a\' = {a_prime}")
        s_true = sum(self.sample.values())
        #print(f"This sampler on j = {self.j} tracks {len(self.sample.keys())} indices")
        #print(f"This sampler on j = {self.j} is {self.s}-sparse: {self.ssr.is_s_sparse}")
        try:
            assert((s_true <= self.s and s_true > 0) == self.ssr.is_s_sparse)
        except(AssertionError):
            print(f"ASSERTION FAILED: {self.sample} but claims that \"is {self.s}-sparse\" = {self.ssr.is_s_sparse}", file=sys.stderr)
        #print(f"z = {self.osr.z}, phi = {self.osr.phi}, tau = {self.osr.tau}, iota = {self.osr.iota}")
        if self.ssr.is_s_sparse:
            #print(f"Sampler that is {self.s}-sparse found!")  
            return a_prime
        else:
            return None 


def gen_a(ss, n, prob, seed=None):
    """
    Generate a stream of length ss, of pairs (i, delta), where i in [0,n] is an identifier, and delta is a modification
    prob represents the probability that a given i will have total sum 0
    """
    index = 0
    cs = [0] * n
    ws = [0] * n
    if seed:
        rng = np.random.default_rng(seed=seed)
    else:
        rng = np.random.default_rng()
    means = rng.choice([0,1], size=n, p=[prob, 1-prob])
    while index < (ss * n):
        i = rng.integers(0, n)
        if cs[i] < ss-1:
            delta = int(100 * (rng.normal()-means[i]))
            yield (i, delta)
            ws[i] += delta
            cs[i] += 1
            index += 1
        elif cs[i] == ss-1:
            delta = means[i]-ws[i]
            yield (i, delta)
            cs[i] += 1
            ws[i] += delta
            index += 1
    #print(f"ws = {ws}, cs = {cs}, means = {means}")
            
def main():
    n = 30
    ss = 30
    prob = 0.8
    if (len(sys.argv) > 1):
        seed = int(sys.argv[1])
    else:
        seed = None
    if (len(sys.argv) > 2):
        n = int(sys.argv[2])
    if (len(sys.argv) > 3):
        ss = int(sys.argv[3])
    # print(f"Trying 10 hash functions in the family of 3-independent hash functions on p=333667, n=20, m=20**3")
    # hs = []
    # for i in range(10):
    #     hs.append(HashFunction(333667, 3, 20**3, seed))
    #     ress = [hs[i].eval(x) for x in range(20)]
    #     print(f"{list(range(20))} -> {ress}")
    #print(get_samples(a, n, HashFunction(333667, 3, n), 4)[4])
    seed = None
    a_gen = gen_a(ss, n, prob)
    s = []
    if (n ** 3 < 333667):
        p = 333667
    elif (n ** 3 < 9999991):
        p = 9999991
    elif (n ** 3 < 2147483647):
        p = 2147483647
    else:
        print("n is too large!")
        sys.exit(2)
    j_max = int(np.log(n))+1
    err = 0.01
    k = 2*int(np.log(1/err))
    rng = np.random.default_rng(seed=seed)
    
    
    # for i in range(j_max):
    #     s.append(SimpleSampler(n, h, p, i+1, seed=seeds[i]))
    #     #print(f"SimpleSampler {i} has indices {s[i].indices}")
    # for a in a_gen:
    #     #print(a)
    #     for sampler in s:
    #         sampler.update(a)
    # for sample in s:
    #     sample.complete()
    ROUNDS=10
    hits = {}
    hits[-1] = 0
    print(f"s = {int(np.log(1/err))}, k = {k}, j_max={j_max}", flush=True)
    for r in range(ROUNDS):
        a_gen = gen_a(ss, n, prob, seed=177)
        h = HashFunction(p, k, n)
        #print(f"TESTING HASH FUNCTION round {r}: coefficients: {h.cs}, hashvals: {[h.eval(z) for z in range(n)]}")
        seeds = rng.integers(0, 1000, size=j_max)
        samplers = []
        for i in range(j_max):
            #samplers.append(SSampler(n, h, p, i+1, err=err, seed=seeds[i]))
            samplers.append(SSampler(n, h, p, i+1, err=err))
        for a in a_gen:
            for sampler in samplers:
                sampler.update(a)
        had_hit = False
        for sample in samplers:
            res = sample.complete()
            if res != None:
                low_hash = p
                low_index = -10
                for l in range(0,len(res)):
                    if low_hash > h.eval(res[l][0]):
                        low_hash = h.eval(res[l][0])
                        low_index = res[l][0]
                if low_index in hits:
                    hits[low_index] += 1
                else:
                    hits[low_index] = 1
                #print(f"Hit in round {r}", flush=True)
                had_hit = True
                break
        if had_hit == False:
            hits[-1] += 1
    miss_rate = hits[-1] / ROUNDS
    print(f"Observed distribution: {hits} (miss rate = {miss_rate:.2f})")
    real_dist = {}
    for i, delta in gen_a(ss, n, prob, seed=177):
        if i not in real_dist:
            real_dist[i] = 0
        real_dist[i] += delta
    s = 0
    for i in real_dist.keys():
        if real_dist[i] == 1:
            s += 1
    for i in list(real_dist.keys()):
        if real_dist[i] == 0:
            del (real_dist[i])
        else:
            real_dist[i] = ROUNDS * real_dist[i] / s
    print(f"Expected distribution assuming no misses: {real_dist}")



if __name__=='__main__':
    main()
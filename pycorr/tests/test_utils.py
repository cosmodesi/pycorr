import numpy as np

from pycorr import utils


def test_sky_cartesian():
    rng = np.random.RandomState(seed=42)
    positions = [rng.uniform(0., 2., 100) for i in range(3)]
    rdd = utils.cartesian_to_sky(positions)
    assert np.allclose(utils.sky_to_cartesian(rdd), positions)


def test_packbit():
    n = 63
    a = np.array([[0,1]*(n//2),[1,0]*(n//2)], dtype='?')
    b = np.packbits(a, axis=-1)
    assert b.shape == (2, (n-1)//8+1)
    b = np.packbits([0,1,0,0,0,1,1,1] + [1] + [0]*3, axis=0, bitorder='little')
    c = np.packbits([0,1,0,0,0,1,1,1] + [1] + [0]*4, axis=0, bitorder='little')
    assert np.all(b == c)


def test_popcount():
    a = np.array(256, dtype=np.uint32)
    assert np.all(utils.popcount(a) == 1)
    for dtype in [np.uint8, np.int8]:
        a = np.array(255, dtype=dtype)
        assert np.all(utils.popcount(a) == 8)

    num = 8564071463
    a = np.array(num, dtype=np.uint64)
    assert np.allclose(utils.popcount(a), bin(num).count('1'))

    a = utils.pack_bitarrays(*[np.array([1], dtype='?') for i in range(33)], dtype=np.uint64)
    num = a[0][0]
    assert np.allclose(utils.popcount(*a), bin(num).count('1'))


def test_reformatbit():
    rng = np.random.RandomState(42)
    size = (2,1)
    a = [np.array(rng.randint(0, int(2**63 - 1), size), dtype=np.uint64)]
    b = utils.reformat_bitarrays(*a, dtype=np.uint16)
    assert all(x_.shape == size for x_ in b)
    assert sum(x_.dtype.itemsize for x_ in b) == sum(x_.dtype.itemsize for x_ in a)
    assert np.all(sum(utils.popcount(x_) for x_ in b) == sum(utils.popcount(x_) for x_ in a))
    assert np.all(sum(utils.popcount(x_[:-1] & x_[1:]) for x_ in b) == sum(utils.popcount(x_[:-1] & x_[1:]) for x_ in a))
    a = [np.array([12890],dtype=np.uint16), np.array([10],dtype=np.uint16)]
    b = utils.reformat_bitarrays(*a, dtype=np.uint8)
    c = utils.reformat_bitarrays(*a[:1], dtype=np.uint8)
    for b_, c_ in zip(b, c):
        assert np.all(b_ == c_)


def pack_bitweights(array):
    """
    Creates an array of bitwise weights stored as 64-bit signed integers
    Input: a 2D boolean array of shape (Ngal, Nreal), where Ngal is the total number
           of target galaxies, and Nreal is the number of fibre assignment realizations.
    Output: returns a 2D array of 64-bit signed integers.
    """
    Nbits = 64
    dtype = np.int64
    Ngal, Nreal = array.shape           # total number of realizations and number of target galaxies
    Nout = (Nreal + Nbits - 1) // Nbits # number of output columns
    # intermediate arrays
    bitw8 = np.zeros((Ngal, 8), dtype="i")   # array of individual bits of 8 realizations
    bitweights = np.zeros(Ngal, dtype=dtype) # array of 64-bit integers
    # array to store final output
    output_array = np.zeros((Ngal, Nout), dtype=dtype)
    idx_out = 0 # initial column in output_array
    # loop through realizations to build bitwise weights
    for i in range(Nreal):
        bitw8[array[:,i], i%8] = 1
        arr = np.array(np.packbits(bitw8[:,::-1]), dtype=dtype)
        bitweights = np.bitwise_or(bitweights, np.left_shift(arr, 8*((i%Nbits)//8)))
        if (i+1)%Nbits == 0 or i+1 == Nreal:
            output_array[:,idx_out] = bitweights
            bitweights[:] = 0
            idx_out += 1
        if (i+1)%8 == 0:
            bitw8[:] = 0
    return output_array


def unpack_bitweights(we):
    Nwe= 1
    Nbits = 64
    Ngal = np.shape(we)[0]
    Nreal = Nbits*Nwe
    print('Nbits, Nwe = ',Nbits,Nwe)
    print('Nreal = ',Nreal)
    print('Ngal = ',Ngal)
    true8=[np.uint8(255) for n in range(0, Ngal)]
    array_bool = np.zeros((Ngal,Nreal), dtype=bool)
    for j in range(Nwe):
        lg = np.zeros((Ngal, Nbits), dtype=bool)
        for i in range(Nbits//8):
            chunk8 = np.uint8(np.bitwise_and(np.right_shift(we,8*i), true8))
            lg[:,Nbits-8*(i+1):Nbits-i*8] = np.reshape(np.unpackbits(chunk8), (Ngal, 8))
        array_bool[:,j*Nbits:(j+1)*Nbits] = lg[:,::-1]
    return array_bool


def test_pack_unpack():
    rng = np.random.RandomState(42)
    size = (8,2)
    bits = [rng.randint(0, 2, size) for i in range(12)]

    for dtype in [np.uint32, np.uint64, 'i4', 'i8']:
        test = utils.unpack_bitarrays(*utils.pack_bitarrays(*bits, dtype=np.uint64))
        for test_, ref_ in zip(test, bits):
            assert test_.shape == ref_.shape
            assert np.all(test_ == ref_)

    """
    size = 8
    bits = [rng.randint(0, 2, size) for i in range(64)]
    ref = pack_bitweights(np.array(bits).T)
    test = utils.pack_bitarrays(*bits, dtype=np.uint64)
    print(ref)
    print(ref.view(dtype=np.uint64))
    print(test)
    """

def get_bitweight_normalization(weights1, weights2=None):
    w1 = np.bincount(utils.popcount(*bitweights1) + 1, weights=indweights1, minlength=nrealizations+1)
    c1 = np.flatnonzero(w1)
    auto = weights2 is None
    if auto:
        w2 = w1
        c2 = c1
    else:
        w2 = np.bincount(utils.popcount(*bitweights2) + 1, weights=indweights2, minlength=nrealizations+1)
        c2 = np.flatnonzero(w2)
    joint = utils.joint_occurences(nrealizations, max_occurences=max(c1.max(), c2.max()))
    sumw_cross = 0
    for c1_, w1_ in zip(c1, w1):
        for c2_, w2_ in zip(c2, w2):
            sumw_cross += w1_ * w2_ * (joint[c1_][c2_] if c2_ <= c1_ else joint[c2_][c1_])
    sumw_auto = 0
    if not auto:
        w1sq = np.bincount(utils.popcount(*bitweights1), weights=indweights1**2, minlength=nrealizations+1)
        for c1_, w1sq_ in zip(c1, w1sq):
            sumw_auto += joint[c1_, c1_] * w1sq_
    return sumw_cross - sumw_auto


def test_normalization():
    #print(utils.pascal_triangle(4))
    size = 42
    tri = utils.pascal_triangle(size)
    a, b = 7, 32
    from scipy import special
    assert tri[b][a] == special.comb(b, a, exact=True)
    tmp = utils.joint_occurences(nrealizations=64)
    print(tmp[30][30])


def test_rebin():
    array = np.ones((4,10), dtype='i4')
    shape = (2, 2)
    rarray = utils.rebin(array, shape, statistic=np.sum)
    assert rarray.shape == shape
    assert np.all(rarray == 10)


if __name__ == '__main__':

    test_sky_cartesian()
    test_normalization()
    test_packbit()
    test_popcount()
    test_reformatbit()
    test_pack_unpack()
    test_normalization()
    test_rebin()

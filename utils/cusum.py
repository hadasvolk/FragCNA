import pandas as pd
import numpy as np

from numba import jit

from sklearn.utils import shuffle
from scipy.special import gammaln, jv
from typing import Tuple


class CUSUM:

    def __init__(self, data: np.ndarray, test: str = 'bessel', zerosv: np.ndarray = None) -> None:

        self.data = data
        self.n, self.m = data.shape
        self.test = test

        self.zerosv = zerosv


    def run(self) -> Tuple[float, int, float, np.ndarray]:
        if self.test == 'bessel':
            return self.cusum()


    def cusum_perm(self, num_permutations: int = 1000) -> Tuple[float, int, float, np.ndarray]:

        max_cusum, arg_max_cusum, cusum_stats = cusum_ma(self.data, self.n, self.m)

        p_val = 0
        for _ in range(num_permutations):
            perm_data = shuffle(self.data)
            max_cusum_perm, _, _ = cusum_ma(perm_data)
            if max_cusum_perm >= max_cusum:
                p_val += 1
        p_val /= num_permutations
        return max_cusum, arg_max_cusum, p_val, cusum_stats


    def cusum(self) -> Tuple[float, int, float, np.ndarray]:

        max_cusum, arg_max_cusum, cusum_stats = cusum_ma(self.data, self.n, self.m)

        tn = np.sqrt(max_cusum)

        z = np.outer(self.zerosv, 1 / tn)
        fuval = np.exp((self.m - 2) * np.log(z) - 0.5 * z**2 - 
                    gammaln(self.m / 2) - self.m / 2 * np.log(2)) / \
                np.resize((jv(self.m / 2, self.zerosv)**2), (50, 1))
        
        fuval = 1 - (4 / tn**2) * np.sum(fuval)

        return max_cusum, arg_max_cusum, fuval, cusum_stats



@jit(nopython=True)
def cusum_ma(data: np.ndarray, n: int = None, m: int = None) -> Tuple[float, int, np.ndarray]:

    # Precomputed chi2.ppf(0.8, df = df)
    # k = np.sqrt(chi2.ppf(0.8, df = data.shape[1]))
    chi2 = {
        1: 1.642374415149818,
        2: 3.218875824868201,
        3: 4.64162767608745,
        4: 5.9886166940042465,
        5: 7.289276126648961,
        6: 8.558059720250668,
        7: 9.803249900240838,
        8: 11.03009143030311,
        9: 12.24214546984707
    }


    def _medians(data):
        medians = np.empty(data.shape[1])
        for i in range(data.shape[1]):
            medians[i] = np.median(data[:, i])
        return medians


    bandwidth = np.log(n / 50) / np.log(1.8 + m / 40)

    med = _medians(data)
    MAD = _medians(np.abs(data - med)) / 1.4826
    data = (data - med) / MAD

    norms = np.array([np.linalg.norm(row) for row in data])

    k = np.sqrt(chi2[m])
    for i in range(n):
        if norms[i] > k:
            data[i, :] = data[i, :] * k / norms[i]
    data = data - data.sum(axis=0) / n

    cov_mat = lrv(data, int(bandwidth), kBartlett)
    cov_mat_inv = np.linalg.pinv(cov_mat)

    # data_cumsum = np.cumsum(data, axis=0)
    data_cumsum = np.empty((n, m))
    for i in range(m):
        data_cumsum[:, i] = np.cumsum(data[:, i])

    cusum_stats = np.zeros(n)
    for i in range(n):
        temp = np.zeros(m)
        for j in range(m):
            temp[j] = data_cumsum[i, j] - ((i - 1) * data_cumsum[-1, j % m]) / n
        
        for j in range(m):
            for k in range(j, m):
                if j == k:
                    cusum_stats[i] += temp[j]**2 * cov_mat_inv[j, j]
                else:
                    cusum_stats[i] += 2 * temp[j] * temp[k] * cov_mat_inv[k, j]
        
        cusum_stats[i] /= n
    
    # # fpc correction
    # max_cusum = (np.sqrt(max_cusum) + 1.46035 / np.sqrt(2 * np.pi) / np.sqrt(n))**2
    return np.max(cusum_stats), np.argmax(cusum_stats), cusum_stats


@jit(nopython=True)
def kParzen(x: float) -> float:
    
    abs_x = np.abs(x)
    if abs_x >= 0 and abs_x <= 0.5:
        return 1 - 6 * abs_x**2 + 6 * abs_x**3
    elif abs_x > 0.5 and abs_x <= 1:
        return 2 * (1 - abs_x)**3
    else:
        return 0


@jit(nopython=True)
def kBartlett(x: float) -> float:
    
    abs_x = np.abs(x)
    if abs_x < 1:
        return 1 - abs_x
    else:
        return 0


@jit(nopython=True)
def lrv1(x: np.ndarray, n: int, bandwidth: int, kernel) -> float:
    
    var = np.sum(np.square(x))
    
    accu = 0
    for h in range(1, bandwidth):
        temp = 0
        for i in range(n - h):
            temp += x[i] * x[i + h]
        accu += temp * kernel(h / bandwidth)
    
    return (var + 2 * accu) / n


@jit(nopython=True)
def lrv2(x1: np.ndarray, x2: np.ndarray, n: int, bandwidth: int, kernel) -> float:
    
    accu = 0
    for i in range(n):
        accu += x1[i] * x2[i]
    for h in range(1, bandwidth):
        temp = 0
        for i in range(n - h):
            temp += x1[i] * x2[i + h] + x1[i + h] * x2[i]
        accu += temp * kernel(h / bandwidth)
    
    return accu / n


@jit(nopython=True)
def lrv(x: np.ndarray, bandwidth: int, kernel) -> np.ndarray:
    
    n, m = x.shape
    cov_mat = np.zeros((m, m))
    for j in range(m):
        for i in range(j, m):
            if i == j:
                cov_mat[i, j] = lrv1(x[:, i], n, bandwidth, kernel)
            else:
                cov_mat[i, j] = cov_mat[j, i] = lrv2(x[:, i], x[:, j], n, bandwidth, kernel)
    
    return cov_mat


if __name__ == '__main__':

    data = pd.read_csv('/mnt/c/Users/hadas/Documents/Repos/FragCNA/utils/test/luad34.regions.entropies.csv').values
    zerosv = pd.read_csv('test/zeros.csv', index_col=0).iloc[:, data.shape[1] - 2]
    t1, pos1, p1, first = CUSUM(data, zerosv=zerosv).run()
    t2, pos2, p2, second = CUSUM(data[:pos1], zerosv=zerosv).run()
    t3, pos3, p3, third = CUSUM(data[pos1:], zerosv=zerosv).run()

    print(f'Tstats: {t1} {t2} {t3}')
    print(f'pos: {pos1} {pos2} {pos3}')
    print(f'pval: {p1} {p2} {p3}')

                
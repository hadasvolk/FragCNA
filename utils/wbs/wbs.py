import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt

from cusum import CUSUM


class WBS:

    def __init__(self, x: np.array, M: int = 5000, rand_interval: bool = True) -> None:
        
        self.x = x
        self.M = M
        self.rand_interval = rand_interval
        
        self.n = len(x)

        assert self.n > 1, "x must have at least 2 elements"
        assert np.isnan(x).any() == False, "x must not contain NaN values"
        assert np.var(x) != 0, "x must not be constant"

        assert np.isnan(M) == False, "M must not be NaN"
        assert M > 0, "M must be positive"

        self.intervals = self.intervals_init()

        self.results = []
    

    def __str__(self):
        
        return f'WBS(x={self.x}, M={self.M}, rand_interval={self.rand_interval}, n={self.n}, \nintervals=\n{self.intervals}, \nresults=\n{self.results})'

    
    def intervals_init(self) -> pd.DataFrame:
        
        if self.rand_interval:
            self.intervals = np.zeros((self.M, 2), dtype=int)

            for i in range(self.M):
                start = np.random.randint(0, self.n - 2)
                end = np.random.randint(start + 2, self.n)
                self.intervals[i, 0] = start
                self.intervals[i, 1] = end

        else:
            m = math.ceil(0.5 * (math.sqrt(8 * self.M + 1) + 1))
            m = min(self.n, m)
            self.M = int(m * (m - 1) / 2)
            end_points = np.round(np.concatenate(([1], np.linspace(2, self.n - 1, m - 2), [self.n])))
            self.intervals = np.zeros((int(self.M), 2))
            
            k = 0
            for i in range(1,m):
                tmp = (m - i)
                self.intervals[k:(k + tmp), 0] = np.repeat(end_points[i-1], tmp)
                self.intervals[k:(k + tmp), 1] = end_points[i:m]
                k = k + tmp
            
        return pd.DataFrame(self.intervals, columns=['s', 'e'], dtype=int)


    @staticmethod
    def ipi_arg_max(res: np.array, n: int) -> tuple:
        
        max_count = 0
        max_fabs = -1
        ipargmax = 0
        for i in range(n-1):
            abs_res = abs(res[i])
            if abs_res > max_fabs:
                ipargmax = i
                max_fabs = abs_res
                max_count = 1
            elif abs_res == max_fabs:
                max_count += 1

        if max_count > 1:
            max_count = max_count // 2 + (max_count % 2)
            k = 0
            i = 0
            while ((i < (n - 1)) and (k < max_count)):
                i += 1
                if abs(res[i]) == max_fabs:
                    k += 1
            ipargmax = i
        
        return ipargmax, res[ipargmax]


    @staticmethod
    def wbs_ipi(x: np.array, n: int) -> tuple:
        
        one_over_n = 1.0 / n
        n_squared = n * n

        iminus = [1.0 / math.sqrt(n_squared - n) * sum(x[1:])]
        iplus = [math.sqrt(1.0 - one_over_n) * x[0]]
        res = [iplus[0] - iminus[0]]

        for i in range(1, n - 1):
            iplusone_inv = 1.0 / (i + 1.0)
            factor = math.sqrt((n - i - 1.0) * i * iplusone_inv / (n - i))
            iplus.append(iplus[i - 1] * factor + x[i] * math.sqrt(iplusone_inv - one_over_n))
            iminus.append(iminus[i - 1] / factor - x[i] / math.sqrt(n_squared * iplusone_inv - n))
            res.append(iplus[i] - iminus[i])

        return WBS.ipi_arg_max(res, n)
        

    def bs_rec(self, x: np.array, s: int, e: int, minth: float = -1., scale: int = 0) -> None:
        
        n = e - s + 1
        if n > 1:
            ipargmax, ipmax = WBS.wbs_ipi(x[s-1:e], n)
            cptcand = ipargmax + s

            if minth > abs(ipmax) or minth < 0:
                minth = abs(ipmax)
            
            self.results.append([s, e, cptcand, ipmax, minth, scale])
            
            self.bs_rec(x, s, cptcand, minth, scale + 1)
            self.bs_rec(x, cptcand + 1, e, minth, scale + 1)


    def wbs_rec(self, s: int, e: int, index: list, indexn: int, minth: float = -1, scale: int = 1) -> None:
        
        n = e - s + 1

        if n > 1:
            if indexn > 0:
                ipargmax, ipmax = WBS.wbs_ipi(self.x[s-1:e], n)
 
                if np.abs(ipmax) < self.wbs_res.loc[index[0], 'abs.CUSUM']:
                    cptcand = self.wbs_res.loc[index[0], 'cpt']
                    if minth > self.wbs_res.loc[index[0], 'abs.CUSUM'] or minth < 0:
                        minth = self.wbs_res.loc[index[0], 'abs.CUSUM']
                    self.results.append([s, e, cptcand, ipmax, minth, scale])
                else:
                    cptcand = ipargmax + s
                    if minth > np.abs(ipmax) or minth < 0:
                        minth = np.abs(ipmax)
                    self.results.append([s, e, cptcand, ipmax, minth, scale])
                
                indexnl, indexnr = [], []
                for i in range(indexn):
                    if self.wbs_res.loc[index[i], 's'] >= s and self.wbs_res.loc[index[i], 'e'] <= cptcand:
                        indexnl.append(index[i])
                    elif self.wbs_res.loc[index[i], 's'] >= cptcand + 1 and self.wbs_res.loc[index[i], 'e'] <= e:
                        indexnr.append(index[i])
                
                if len(indexnl) > 0:
                    self.wbs_rec(s, cptcand, indexnl, len(indexnl), minth, scale + 1)
                else:
                    self.bs_rec(self.x, s, cptcand, minth, scale + 1)
                
                if len(indexnr) > 0:
                    self.wbs_rec(cptcand + 1, e, indexnr, len(indexnr), minth, scale + 1)
                else:
                    self.bs_rec(self.x, cptcand + 1, e, minth, scale + 1)
            
            else:
                self.bs_rec(self.x, s, e, minth, scale)

    
    def wbs_cusum(self) -> None:
        
        wbs_res = []
        for i in range(self.M):
            s = self.intervals.loc[i, 's'] + 1
            e = self.intervals.loc[i, 'e'] + 1
            ipargmax, ipmax = WBS.wbs_ipi(self.x[s-1:e], e - s + 1)
            cptcand = ipargmax + s
            wbs_res.append([s, e, cptcand, ipmax, abs(ipmax)])
    
        self.wbs_res = pd.DataFrame(wbs_res, columns=['s', 'e', 'cpt', 'CUSUM', 'abs.CUSUM'])
        largest_cusum_index = self.wbs_res.sort_values(by='abs.CUSUM', ascending=False).index.to_list()
        
        self.wbs_rec(1, self.n, largest_cusum_index, self.M)

        self.results = pd.DataFrame(self.results, columns=['s', 'e', 'cpt', 'CUSUM', 'min.th', 'scale'])
        self.results.sort_values(by=['cpt'], inplace=True)
    

    def bs_rec_mulcusum(self, s: int, e: int, minth: float = -1, scale: int = 0) -> None:

        n = e - s
        if n > 1:
            t, pos, p, arr = CUSUM(self.x[s:e+1], test='bessel').run()
            pos = pos + s

            if minth > t or minth < 0:
                minth = t
                
            self.results.append([s, e, pos + s, t, minth, scale])

            self.bs_rec_mulcusum(s, pos, minth, scale + 1)
            self.bs_rec_mulcusum(pos + 1, e, minth, scale + 1)


    def wbs_rec_mulcusum(self, s: int, e: int, index: list, indexn: int, minth: float = -1, scale: int = 0) -> None:
        
        n = e - s

        if n > 1:
            if indexn > 0:
                t, pos, p, arr = CUSUM(self.x[s:e+1], test='bessel').run()

                if t < self.wbs_res.loc[index[0], 'CUSUM']:
                    cptcand = self.wbs_res.loc[index[0], 'cpt']
                    if minth > self.wbs_res.loc[index[0], 'CUSUM'] or minth < 0:
                        minth = self.wbs_res.loc[index[0], 'CUSUM']
                    self.results.append([s, e, cptcand, t, p, scale])
                else:
                    cptcand = pos + s
                    if minth > t or minth < 0:
                        minth = t
                    self.results.append([s, e, cptcand, t, p, scale])
                
                indexnl, indexnr = [], []
                for i in range(indexn):
                    if self.wbs_res.loc[index[i], 's'] >= s and self.wbs_res.loc[index[i], 'e'] <= cptcand:
                        indexnl.append(index[i])
                    elif self.wbs_res.loc[index[i], 's'] >= cptcand + 1 and self.wbs_res.loc[index[i], 'e'] <= e:
                        indexnr.append(index[i])
                
                if len(indexnl) > 0:
                    self.wbs_rec_mulcusum(s, cptcand, indexnl, len(indexnl), minth, scale + 1)
                else:
                    self.bs_rec_mulcusum(s, cptcand, minth, scale + 1)

                if len(indexnr) > 0:
                    self.wbs_rec_mulcusum(cptcand + 1, e, indexnr, len(indexnr), minth, scale + 1)
                else:
                    self.bs_rec_mulcusum(cptcand + 1, e, minth, scale + 1)
                
            else:
                self.bs_rec_mulcusum(s, e, minth, scale)

    
    def wbs_mulcusum(self) -> None:

        wbs_res = []
        for i in range(self.M):
            s = self.intervals.loc[i, 's']
            e = self.intervals.loc[i, 'e']
            t, pos, p, arr = CUSUM(self.x[s:e+1], test='bessel').run()
            pos = pos + s
            wbs_res.append([s, e, pos, t, p])
        
        self.wbs_res = pd.DataFrame(wbs_res, columns=['s', 'e', 'cpt', 'CUSUM', 'p-value'])
        largest_cusum_index = self.wbs_res.sort_values(by='CUSUM', ascending=False).index.to_list()

        self.wbs_rec_mulcusum(0, self.n - 1, largest_cusum_index, self.M)

        self.results = pd.DataFrame(self.results, columns=['s', 'e', 'cpt', 'CUSUM', 'pval', 'scale'])
        self.results.sort_values(by=['cpt'], inplace=True)

    
    def cost_rank(self, s: int, e: int) -> float:
        if e - s == 0:
            raise ValueError("end - start must be greater than 1")
        
        mean = np.reshape(np.mean(self.ranks[s:e], axis=0), (-1, 1))
        gain = -(e - s) * mean.T @ self.inv_cov @ mean
        return gain[0][0]
    

    def rank_cptcand(self, s: int, e: int) -> tuple:

        segment_cost = self.cost_rank(s, e)
        gains = list()
        for i in range(s + 1, e - 1):
            gain = segment_cost - self.cost_rank(s, i) - self.cost_rank(i + 1, e)
            gains.append((i, gain))
        cptcand, i = max(gains, key=lambda x: x[1])
        return gain, cptcand
    

    def bs_rec_ml(self, s: int, e: int, scale: int = 0) -> None:

        n = e - s

        if n > 1:
            gain, cptcand = self.ml_cptcand(s, e + 1)
            self.results.append([s, e, cptcand, gain, scale])
            self.bs_rec_ml(s, cptcand, scale + 1)
            self.bs_rec_ml(cptcand + 1, e, scale + 1)
    

    def wbs_rec_ml(self, s: int, e: int, index: list, indexn: int, scale: int = 0) -> None:

        n = e - s

        if n > 1:
            if indexn > 0:
                gain, cptcand = self.ml_cptcand(s, e + 1)
                if gain < self.wbs_res.loc[index[0], 'gain']:
                    cptcand = self.wbs_res.loc[index[0], 'cpt']
                    self.results.append([s, e, cptcand, gain, scale])
                else:
                    self.results.append([s, e, cptcand, gain, scale])
            
                indexnl, indexnr = [], []
                for i in range(indexn):
                    if self.wbs_res.loc[index[i], 's'] >= s and self.wbs_res.loc[index[i], 'e'] <= cptcand:
                        indexnl.append(index[i])
                    elif self.wbs_res.loc[index[i], 's'] >= cptcand + 1 and self.wbs_res.loc[index[i], 'e'] <= e:
                        indexnr.append(index[i])
                
                if len(indexnl) > 0:
                    self.wbs_rec_ml(s, cptcand, indexnl, len(indexnl), scale + 1)
                else:
                    self.bs_rec_ml(s, cptcand, scale + 1)
                
                if len(indexnr) > 0:
                    self.wbs_rec_ml(cptcand + 1, e, indexnr, len(indexnr), scale + 1)
                else:
                    self.bs_rec_ml(cptcand + 1, e, scale + 1)

            else:
                self.bs_rec_ml(s, e, scale)
    

    def cost_ml(self, s: int, e: int) -> float:
    
        sub_gram = self.gram[s:e, s:e]
        val = np.diagonal(sub_gram).sum()
        val -= sub_gram.sum() / (e - s)
        return val
    

    def ml_cptcand(self, s: int, e: int) -> tuple:

        segment_cost = self.cost_ml(s, e)
        gains = list()
        for i in range(s + 1, e - 1):
            gain = segment_cost - self.cost_ml(s, i) - self.cost_ml(i + 1, e)
            gains.append((i, gain))
        cptcand, i = max(gains, key=lambda x: x[1])
        return gain, cptcand


    def wbs_ml(self) -> None:

        s_ = self.x.reshape(-1, 1) if self.x.ndim == 1 else self.x

        covar = np.cov(s_.T)
        self.metric = np.linalg.inv(covar.reshape(1, 1) if covar.size == 1 else covar)

        self.gram = s_.dot(self.metric).dot(s_.T)
        self.signal = s_

        wbs_res = []
        for i in range(self.M):
            s = self.intervals.loc[i, 's']
            e = self.intervals.loc[i, 'e']
            gain, cptcand = self.ml_cptcand(s, e + 1)
            wbs_res.append([s, e, cptcand, gain])
        
        self.wbs_res = pd.DataFrame(wbs_res, columns=['s', 'e', 'cpt', 'gain'])
        largest_gain_index = self.wbs_res.sort_values(by='gain', ascending=False).index.to_list()

        self.wbs_rec_ml(0, self.n - 1, largest_gain_index, self.M)

        self.results = pd.DataFrame(self.results, columns=['s', 'e', 'cpt', 'gain', 'scale'])
        self.results.sort_values(by=['cpt'], inplace=True)

    
    @staticmethod
    def means_between_changepoints(x: np.array, changepoints: list) -> np.array:
        
        changepoints = sorted(changepoints)
        len_cpt = len(changepoints)
        s = np.zeros(len_cpt + 1, dtype=int)
        e = np.zeros(len_cpt + 1, dtype=int)
        e[-1] = len(x) - 1
        if len_cpt:
            s[1:] = np.array(changepoints) + 1
            e[:-1] = np.array(changepoints)

        means = np.zeros(len_cpt + 1)
        for i in range(len_cpt + 1):
            means[i] = np.mean(x[s[i]:e[i]+1])
        
        return np.repeat(means, e - s + 1)
    

    @staticmethod
    def ssic_penalty(n: int, cpt: list, alpha: float, ssic_type: str) -> float:
        
        if ssic_type == "log": pen = np.log(n) ** alpha
        elif ssic_type == "power": pen = n ** alpha
        else : raise ValueError("ssic_type must be 'log' or 'power'")
        
        return pen * len(cpt)
    

    def changepoint_cusum(self, threshold: float = None, threshold_const: float = 1.3, Kmax: int = 50, alpha: float = 1.01, ssic_type: str = "log") -> None:

        sigma = np.median(np.abs(np.diff(self.x) - np.median(np.diff(self.x)))) * 1.4826 / np.sqrt(2)

        if threshold is not None:
            th = threshold
        else:
            th = sigma * threshold_const * np.sqrt(2 * np.log(self.n))

        self.results.sort_values(by=['min.th'], ascending=False, inplace=True)
        changepoints = self.results['cpt'].tolist()[0:Kmax]
        changepoints = [x - 1 for x in changepoints]

        ic_curve = np.zeros(len(changepoints) + 1)
        for i in range(len(changepoints), -1, -1):
            means = WBS.means_between_changepoints(self.x, changepoints[:i])
            min_log_likelihood = self.n / 2 * np.log(np.sum((self.x - means) ** 2) / self.n)
            ic_curve[i] = min_log_likelihood + WBS.ssic_penalty(self.n, changepoints[:i], alpha=alpha, ssic_type=ssic_type)
        min_ic_index = np.argmin(ic_curve)
        if min_ic_index == 0:
            cpt_ic = None
        else:
            cpt_ic = changepoints[:min_ic_index]
        self.changepoints = cpt_ic

    
    def changepoint_ml(self, threshold: float = None, threshold_const: float = 1.3, Kmax: int = 50, alpha: float = 1.01, ssic_type: str = "log") -> None:

        sigma = np.median(np.abs(np.diff(self.x) - np.median(np.diff(self.x)))) * 1.4826 / np.sqrt(2)

        if threshold is not None:
            th = threshold
        else:
            th = sigma * threshold_const * np.sqrt(2 * np.log(self.n))

        self.results.sort_values(by=['gain'], ascending=False, inplace=True)
        changepoints = self.results['cpt'].tolist()[0:Kmax]
        changepoints = [x - 1 for x in changepoints]

        ic_curve = np.zeros(len(changepoints) + 1)
        for i in range(len(changepoints), -1, -1):
            means = WBS.means_between_changepoints(self.x, changepoints[:i])
            min_log_likelihood = self.n / 2 * np.log(np.sum((self.x - means) ** 2) / self.n)
            ic_curve[i] = min_log_likelihood + WBS.ssic_penalty(self.n, changepoints[:i], alpha=alpha, ssic_type=ssic_type)
        min_ic_index = np.argmin(ic_curve)
        if min_ic_index == 0:
            cpt_ic = None
        else:
            cpt_ic = changepoints[:min_ic_index]
        self.changepoints = cpt_ic


if __name__ == "__main__":
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    # np.random.seed(42)
    # x1 = np.concatenate((np.random.normal(0, 1, 50), np.random.normal(1, 1, 50), np.random.normal(0, 1, 50)))
    # x2 = np.concatenate((np.random.normal(0, 1, 50), np.random.normal(1, 1, 50), np.random.normal(0, 1, 50)))
    # x = np.vstack((x1, x2)).T

    # wbs = WBS(x)
    # wbs.wbs_ml()
    # print(wbs.results.sort_values(by=['gain'], ascending=False))
    # print(wbs.changepoint_ml(Kmax=100, alpha=0.1, ssic_type="log"))
    # print(wbs.changepoint(Kmax=100, alpha=0.8, ssic_type="log"))

    # print(wbs.results.loc[wbs.results['min.th'].max() == wbs.results['min.th'], :])
    # plt.figsize=(20, 10)
    # plt.plot(wbs.results['cpt'], wbs.results['CUSUM'])
    # plt.show()


    data = pd.read_csv('/mnt/c/Users/hadas/Documents/Repos/FragCNA/utils/test/luad34.regions.entropies.csv')

    wbs = WBS(data.values)
    wbs.wbs_mulcusum()
    print(wbs.results.sort_values(by=['pval']))
    wbs.results.sort_values(by=['pval'], inplace=True).to_csv('/mnt/c/Users/hadas/Documents/Repos/FragCNA/utils/test/wbs.mulcusum.csv', index=False)

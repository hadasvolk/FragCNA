import math
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib.pyplot as plt

from cusum import CUSUM


class WBS:

    def __init__(self, x: np.array, M: int = 5000, rand_interval: bool = True, zeros_path: str = None) -> None:
        
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

        self.zerosv = pd.read_csv(zeros_path, index_col=0).iloc[:, self.x.shape[1] - 2]
    

    def __str__(self):
        
        return f'WBS(\nx=\n{self.x} \n\nM = {self.M} \n\nrand_interval = {self.rand_interval} \n\nn = {self.n} \n\nintervals = \n{self.intervals} \n\nresults = \n{self.results})'

    
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


    def bs_rec_mulcusum(self, s: int, e: int, minth: float = -1, scale: int = 0) -> None:

        n = e - s + 1
        
        if n > 1:
            t, pos, p, arr = CUSUM(self.x[s:e], test='bessel', zerosv=self.zerosv).run()
            cptcand = pos + s

            if minth > t or minth < 0:
                minth = t
                
            self.results.append([s, e, cptcand, t, minth, scale])

            self.bs_rec_mulcusum(s, cptcand, minth, scale + 1)
            self.bs_rec_mulcusum(cptcand, e, minth, scale + 1)


    def wbs_rec_mulcusum(self, s: int, e: int, index: list, indexn: int, minth: float = -1, scale: int = 0) -> None:
        
        n = e - s + 1

        if n > 1:
            if indexn > 0:
                t, pos, p, arr = CUSUM(self.x[s:e], test='bessel', zerosv=self.zerosv).run()

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
                    self.wbs_rec_mulcusum(cptcand, e, indexnr, len(indexnr), minth, scale + 1)
                else:
                    self.bs_rec_mulcusum(cptcand, e, minth, scale + 1)
                
            else:
                self.bs_rec_mulcusum(s, e, minth, scale)

    
    def wbs_mulcusum(self) -> None:

        wbs_res = []
        for i in range(self.M):
            s = self.intervals.loc[i, 's']
            e = self.intervals.loc[i, 'e']
            t, pos, p, arr = CUSUM(self.x[s:e+1], test='bessel', zerosv=self.zerosv).run()
            cptcand = pos + s
            wbs_res.append([s, e, cptcand, t, p])
        
        self.wbs_res = pd.DataFrame(wbs_res, columns=['s', 'e', 'cpt', 'CUSUM', 'p-value'])
        largest_cusum_index = self.wbs_res.sort_values(by='CUSUM', ascending=False).index.to_list()

        self.wbs_rec_mulcusum(0, self.n, largest_cusum_index, self.M)

        self.results = pd.DataFrame(self.results, columns=['s', 'e', 'cpt', 'CUSUM', 'pval', 'scale'])
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
    

    def changepoint(self, threshold: float = None, threshold_const: float = 1.3, Kmax: int = 50, alpha: float = 1.01, ssic_type: str = "log") -> None:

        self.results.sort_values(by=['CUSUM'], ascending=False, inplace=True)
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


    data = pd.read_csv('/mnt/c/Users/hadas/Documents/Repos/FragCNA/utils/test/luad34.regions.entropies.csv')

    wbs = WBS(data.values)
    wbs.wbs_mulcusum()
    wbs.results.sort_values(by=['pval']).to_csv('/mnt/c/Users/hadas/Documents/Repos/FragCNA/utils/test/wbs.mulcusum.csv', index=False)
    print(wbs.changepoint())

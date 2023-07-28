import numpy as np
import random

class CBSSegmenter:
    def __init__(self, shuffles, p):
        self.shuffles = shuffles
        self.p = p

    def calculate_cbs_stat(self, x):
        mean = np.median(x)
        sum_value = 0
        i_max = (-1, -np.inf)
        i_min = (-1, np.inf)
        for i in range(len(x)):
            x0 = x[i] - mean
            sum_value += x0
            e0 = (i, sum_value)
            if e0[1] > i_max[1]:
                i_max = e0
            if e0[1] < i_min[1]:
                i_min = e0
        i0 = min(i_max, i_min, key=lambda pair: pair[0])
        i1 = max(i_max, i_min, key=lambda pair: pair[0])
        max_t = 0 if i1[0] == i0[0] else ((i1[1] - i0[1])**2 * len(x)) / ((i1[0] - i0[0]) * (len(x) - i1[0] + i0[0]))
        return i0[0], i1[0] + 1, max_t, False

    def calculate_cbs(self, x):
        max_start, max_end, max_t, threshold = self.calculate_cbs_stat(x)
        if max_end - max_start == len(x):
            return max_start, max_end, max_t, False
        if max_start < 1:
            max_start = 0
        if len(x) - max_end < 1:
            max_end = len(x)
        thresh_count = 0
        alpha = self.shuffles * self.p
        xt = x.copy()
        for _ in range(self.shuffles):
            random.shuffle(xt)
            max_start_s, max_end_s, max_t_s, _ = self.calculate_cbs_stat(xt)
            if max_t_s >= max_t:
                thresh_count += 1
            if thresh_count > alpha:
                return max_start, max_end, max_t, False
        if thresh_count <= alpha:
            return max_start, max_end, max_t, True
        return max_start, max_end, max_t, False

    def recursive_binary_segmentation(self, x, start, end):
        if end <= start + 1:
            return []
        slice_x = x[start:end]
        max_start, max_end, max_t, threshold = self.calculate_cbs(slice_x)
        cbs_length = max_end - max_start
        if not threshold or cbs_length < 2 or cbs_length > (end - start - 2):
            return [(start, end, max_t, np.median(slice_x))]
        return (self.recursive_binary_segmentation(x, start, start + max_start) +
                [(start + max_start, start + max_end, max_t, np.median(x[start + max_start : start + max_end]))] +
                self.recursive_binary_segmentation(x, start + max_end, end))


    def cbs_segment(self, values):
        return self.recursive_binary_segmentation(values, 0, len(values))


def plot_segments(segments, values, bkps = None):
    plt.figure(figsize=(10, 8))
    for s in segments:
        plt.scatter(range(s[0], s[1]), values[s[0]:s[1]], s=1)
        plt.hlines(s[3], s[0], s[1], color='r')
    if bkps is not None:
        for b in bkps:
            plt.axvline(x=b, color='r', linestyle='--')
    plt.hlines(0, 0, len(values), color='k')
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    np.random.seed(10)
    x = np.random.random(100000)
    x[10000:20000] = x[10000:20000] + 0.1
    x[25000:27000] = x[25000:27000] - 0.1
    x[28000:29000] = x[28000:29000] + 0.2

    bkps = [10000, 20000, 25000, 27000, 28000, 29000]

    cbs = CBSSegmenter(100, 0.05)
    segments = cbs.cbs_segment(x)
    for s in segments:
        print(s)
    plot_segments(segments, x, bkps)


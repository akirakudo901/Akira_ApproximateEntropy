
import random

import matplotlib.pyplot as plt
import numpy as np

import timeSeries

class ApproximateEntropy:

    def approximate_entropy_from_file(file_path : str, m : int=2, r : float=None,
                                      use_memoization : bool = True):
        X = timeSeries.return_time_series(file_path)
        return ApproximateEntropy.approximate_entropy(X, m, r, use_memoization)

    def approximate_entropy(time_series, m : int=2, r : float=None,
                            use_memoization : bool = True):
        """
        Calculates the approximate entropy of the given time series with
        given values of m and r.

        :param List time_series: A time series holding data.
        :param int m: The window size for ApEn.
        :param float r: The noise filter size for ApEn.
        :param bool use_memoization: Boolean indicating whether to use memoization.
        Defaults to True.
        """
        # first extract the number of items N from time_series
        N = len(time_series)
        X = np.array(time_series)
        # ensure ApEn is runnable - N is not smaller than m + 1, for example
        if m + 1 > N: 
            raise Exception("The chosen m value is too big to be applied for the time series.\
                            Choose a time series of length greater than m + 1. m defaults to 2.")
        
        # if r was not given, intialize it to 0.2 * std of time series, as given in paper
        if r is None: r = 0.2 * np.std(X)
        # ensure r is positive real
        if r <= 0: raise Exception("r has to be set to be a positive real number!")

        if use_memoization:
            phi_m = ApproximateEntropy._get_phi_m_r_memoization(X[:-1], m, r)
            phi_m1 = ApproximateEntropy._get_phi_m_r_memoization(X, m + 1, r)
        else:
            phi_m = ApproximateEntropy._get_phi_m_r(X[:-1], m, r)
            phi_m1 = ApproximateEntropy._get_phi_m_r(X, m + 1, r)
        return phi_m - phi_m1

    def _get_block(time_series, start_idx, m):
        return time_series[start_idx : start_idx + m]

    def _get_C_m_i_r(time_series, m, i, r):
        count = 0
        N = len(time_series)
        x_i = ApproximateEntropy._get_block(time_series, i, m)

        for j in range(N - m + 1):
            x_j = ApproximateEntropy._get_block(time_series, j, m)
            if ApproximateEntropy._get_dist_between_blocks(x_i, x_j) <= r:
                count += 1
        
        return count / (N - m + 1)
    
    def _get_C_m_i_r_memoization(time_series, m, i, r, memo):
        """
        Calculate C_m_i_r using memoization by passing and returning a 
        memo which stores comparisons already done.
        memo is an np.ndarray of length (N-m+1) which ith position 
        holds the number of matches between (x[i], ..., x[i+m-1]) and 
        (x[j], ..., x[j+m-1]) for all j <= i.

        For example, if we have a perfectly regular time series with all entries
        being matches of each other, then the memo for when i=3 starts will be something like:
        memo: np.array([1, 2, 3, 4, 1, 1, 1])
        """
        count = memo[i]
        N = len(time_series)
        x_i = ApproximateEntropy._get_block(time_series, i, m)

        for j in range(i + 1, N - m + 1):
            x_j = ApproximateEntropy._get_block(time_series, j, m)
            if ApproximateEntropy._get_dist_between_blocks(x_i, x_j) <= r:
                count += 1
                memo[j] += 1
        
        return count / (N - m + 1)

    def _get_dist_between_blocks(block1, block2):
        return np.max(np.abs(block1 - block2))

    def _get_phi_m_r(time_series, m, r):
        sum_log_cmir = 0
        N = len(time_series)
        for i in range(N - m + 1):
            log_cmir = np.log(ApproximateEntropy._get_C_m_i_r(time_series, m, i, r))
            sum_log_cmir += log_cmir
        
        return sum_log_cmir / (N - m + 1)
    
    def _get_phi_m_r_memoization(time_series, m, r):
        """
        Implements memoization for calculation.
        memo is an np.ndarray of length (N-m+1) which 1-indexed ith position 
        holds the number of matches between (x[i], ..., x[i+m-1]) and 
        (x[j], ..., x[j+m-1]) for all j <= i.

        For example, if we have a perfectly regular time series with all entries
        being matches of each other, then the memo for when i=3 will be something like:
        memo: np.array([1, 2, 3, 1, 1, 1, 1])
        """
        sum_log_cmir = 0
        N = len(time_series)
        memo = np.ones((N - m + 1, )) #memo initialized with all 1 entries

        for i in range(N - m + 1):
            log_cmir = np.log(ApproximateEntropy._get_C_m_i_r_memoization(time_series, m, i, r, memo))
            sum_log_cmir += log_cmir
        
        return sum_log_cmir / (N - m + 1)

    
    def test():
        l1 = [0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.4,-0.3,-0.2,-0.1,0.0]
        l2 = l1*20
        time_series1 = np.array(l2)

        random.seed(99) #10
        random.shuffle(l2)
        time_series2 = np.array(l2)

        plt.plot(time_series1)
        plt.title = "Time series 1"
        # plt.show()
        print(f"time_series1[0:20]: {time_series1[0:20]} with length {len(time_series1)}.\n")

        plt.plot(time_series2)
        plt.title = "Time series 2"
        # plt.show()
        print(f"time_series2[0:20]: {time_series2[0:20]} with length {len(time_series2)}.\n")

        t1_apen = ApproximateEntropy.approximate_entropy(time_series1, m=2)
        t2_apen = ApproximateEntropy.approximate_entropy(time_series2, m=2)
        print(f"ApEn scores are {t1_apen} for series 1 and {t2_apen} for series 2! Expected?")
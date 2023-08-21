
import random
import unittest

import numpy as np

from approximate_entropy import ApproximateEntropy as AE

class TestApproximateEntropy(unittest.TestCase):

    def test_return_block(self):
        ts = np.array(range(4))

        actual1 = AE._get_block(ts, start_idx=0, m=2)
        expected1 = np.array([0, 1])
        self.assertTrue(np.equal(actual1.all(), expected1.all()))
        
        actual2 = AE._get_block(ts, start_idx=2, m=2)
        expected2 = np.array([2, 3])
        self.assertTrue(np.equal(actual2.all(), expected2.all()))
        
        actual3 = AE._get_block(ts, start_idx=1, m=2)
        expected3 = np.array([1, 2, 3])
        self.assertTrue(np.equal(actual3.all(), expected3.all()))

        actual4 = AE._get_block(ts, start_idx=1, m=2)
        expected4 = np.array([0, 1, 2])
        self.assertFalse(np.equal(actual4.all(), expected4.all()))
    
    def test_get_dist_between_blocks(self):
        b1 = np.array([0, 1, 2, 3])
        b2 = np.array([9, 8, 7, 6])
        self.assertEqual(AE._get_dist_between_blocks(b1, b2), 9)
        
        b1 = np.array([0, 1, 2, 3])
        b2 = np.array([0, 3, 6, 2])
        self.assertEqual(AE._get_dist_between_blocks(b1, b2), 4)

        b1 = np.array([0, 1, 2, 3])
        b2 = np.array([0, 1, 2, 3])
        self.assertEqual(AE._get_dist_between_blocks(b1, b2), 0)

        b1 = np.array([9, 51, 6, 13])
        b2 = np.array([10, 2, 22, 3])
        self.assertEqual(AE._get_dist_between_blocks(b1, b2), 49)

    def test_get_C_m_i_r(self):
        ts = np.array([0., 1., 2., 3.])
        b1 = np.array([0., 1.])
        r = 1.5
        actual = AE._get_C_m_i_r(ts, m=2, i=0, r=r)
        expected = (int(AE._get_dist_between_blocks(b1, np.array([0., 1.])) <= r) +
                    int(AE._get_dist_between_blocks(b1, np.array([1., 2.])) <= r) +
                    int(AE._get_dist_between_blocks(b1, np.array([2., 3.])) <= r)) / 3
        self.assertEqual(actual, expected)

    def test_memoization_equal_results_to_non_memo(self):
        l1 = [0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2,0.1,0.0,-0.1,-0.2,-0.3,-0.4,-0.5,-0.4,-0.3,-0.2,-0.1,0.0]
        l2 = l1*20
        time_series1 = np.array(l2)

        random.seed(99) #10
        random.shuffle(l2)
        time_series2 = np.array(l2)

        t1_apen = AE.approximate_entropy(time_series1, m=2, use_memoization=False)
        t2_apen = AE.approximate_entropy(time_series2, m=2, use_memoization=False)

        t1_apen_memo = AE.approximate_entropy(time_series1, m=2, use_memoization=True)
        t2_apen_memo = AE.approximate_entropy(time_series2, m=2, use_memoization=True)

        self.assertAlmostEqual(t1_apen, t1_apen_memo)
        self.assertAlmostEqual(t2_apen, t2_apen_memo)

if __name__ == "__main__":
    unittest.main()
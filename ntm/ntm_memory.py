import numpy as np

"""Description of this file."""

__author__ = "lfievet"
__copyright__ = "Copyright 2016, NTM"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "28/09/2016"
__status__ = "Production"


class NtmMemory:
    def __init__(self, n, m):
        """
        Initialize a blurry memory matrix of size NxM.
        :param n: Number of memory locations.
        :param m: Vector size at each location.
        """

        self.n = n
        self.m = m
        self.memory = np.zeros((n, m))

    def read(self, w):
        """
        Read the memory based on weights w.
        :param w: Location weights of length N.
        :return: Vector of size M.
        """

        self.assert_valid_weights(w)

        return w.dot(self.memory)

    def write(self, w, e, a):
        """
        Write the memory based on weights w.
        :param w: Location weights of length N.
        :param e: Erase vector of length M.
        :param a: Add vector of length M.
        """

        self.assert_valid_weights(w)
        self.assert_vector(e, "erase")
        self.assert_vector(a, "add")

        # Erase step
        erase_matrix = np.ones((self.n, self.m)) - np.array([wi * e for wi in w])
        self.memory *= erase_matrix

        # Add step
        add_matrix = np.array([wi * a for wi in w])
        self.memory += add_matrix

    def assert_vector(self, v, name=""):
        if len(v) != self.m:
            raise AssertionError(
                "Expected {0} vector of length {1}, received "
                "length {2} instead: {3}.".format(
                    name,
                    self.m,
                    len(v),
                    v
                ))

        if np.sum([0 if abs(vi) <= 1 else 1 for vi in v]) > 0:
            raise AssertionError(
                "Expected {0} elements to be in [0, 1]. Instead got {0}.".format(
                    name,
                    v
                ))

    def assert_valid_weights(self, w):
        """
        Check that the given weights are valid.
        Weights need to have length N and sum to one.
        :param w: Vector of length N.
        """

        if len(w) != self.n:
            raise AssertionError(
                "Expected vector of {0} weights, received "
                "{1} weights instead: {2}.".format(
                    self.n,
                    len(w),
                    w
                ))

        if abs(np.sum(w) - 1.0) > 1E-6:
            raise AssertionError(
                "Expected weights to sum to one. Instead got {0}.".format(
                    np.sum(w)
                ))

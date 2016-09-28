import unittest

import numpy as np

from ntm.ntm_memory import NtmMemory

"""Description of this file."""

__author__ = "lfievet"
__copyright__ = "Copyright 2016, code"
__credits__ = ["lfievet"]
__license__ = "No License"
__version__ = "1.0"
__maintainer__ = "lfievet"
__email__ = "lfievet@ethz.ch"
__date__ = "28/09/2016"
__status__ = "Production"


from numpy.testing import assert_equal


class NtmMemoryTests(unittest.TestCase):
    def test_read_wrong_number_weights(self):
        memory = NtmMemory(2, 4)

        self.assertRaises(
            AssertionError,
            memory.read,
            np.array([1])
        )

    def test_read_empty_memory(self):
        memory = NtmMemory(2, 4)
        value = memory.read(np.array([1, 0]))

        self.assertEqual(
            len(value),
            4,
            "Value should be of length 4"
        )

        assert_equal(
            value,
            np.array([0, 0, 0, 0]),
            "Values should all be zero"
        )

    def test_write_onto_empty_memory(self):
        memory = NtmMemory(2, 4)
        memory.write(
            np.array([1, 0]),
            np.array([0, 0, 0, 0]),
            np.array([1, 0, 1, 0])
        )

        expected = np.ndarray((2, 4))
        expected[0, 0] = 1
        expected[0, 2] = 1

        assert_equal(
            memory.memory,
            expected,
            "Values are not as expected."
        )

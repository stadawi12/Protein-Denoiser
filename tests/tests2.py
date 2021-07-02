import sys
import os
sys.path.insert(1, '../lib')
sys.path.insert(1, '../lib/utils')

# Unit test import
import unittest

from Inputs import Read_Input

input_data = Read_Input('inputs.yaml')

class Test_functions(unittest.TestCase):

    def test_train(self):
        pass

if __name__ == '__main__':
    unittest.main()

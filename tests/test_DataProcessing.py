import unittest
import pandas as pd
from pandas.testing import assert_frame_equal
from Utils.DataUtils.DataImputation import impute_missing_values
import sys
import os

# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

class TestData(unittest.TestCase):
    
    def setUp(self) -> None:
        super().setUp()
        self.dataframe = pd.DataFrame({'A': [1, 2, None, 4, 5],
                                       'B': [None, 4, 3, None, 2],
                                       'C': [3, 1, 3, 6, 4]})
    
    def test_impute_missing_values_mean(self):
        expected_result = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                                        'B': [3, 4, 3, 3, 2],
                                        'C': [3, 1, 3, 6, 4]})

        # Call the function being tested
        imputed_dataframe = impute_missing_values(self.dataframe, method='mean')
        # Perform assertion using assert_frame_equal
        assert_frame_equal(expected_result.astype(float), imputed_dataframe.astype(float))
        
        
    def test_impute_missing_values_forward_backward(self):
        expected_result = pd.DataFrame({'A': [1, 2, 2, 4, 5],
                                        'B': [4, 4, 3, 3, 2],
                                        'C': [3, 1, 3, 6, 4]})

        # Call the function being tested
        imputed_dataframe = impute_missing_values(self.dataframe, method='forward')
        imputed_dataframe = impute_missing_values(imputed_dataframe, method='backward')
        # Perform assertion using assert_frame_equal
        assert_frame_equal(expected_result.astype(float), imputed_dataframe.astype(float))
        
         
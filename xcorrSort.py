import sys
sys.path.insert(1, 'lib/utils')

from xCorr3D import xcorr_sort
from Inputs import Read_Input
from sort import Maps

"""This should automatically send all maps below 
cross correlation threshold to bad maps.
Remember to use ccpem-python for this module
"""


if __name__ == '__main__':

    input_data = Read_Input('inputs.yaml')
    threshold = input_data["threshold"]

    # Ensure all maps are moved out of badMaps
    m1 = Maps('data/1.0/')
    m2 = Maps('data/2.0/')
    m1.move_all_from_bad()
    m2.move_all_from_bad()

    xcorr_sort('data', threshold)
    print("All maps below threshold have been moved to badMaps")

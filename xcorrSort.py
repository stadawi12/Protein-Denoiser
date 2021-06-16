import sys
sys.path.insert(1, 'lib/utils')

from xCorr3D import xcorr_sort
from Inputs import Read_Input

"""This should automatically send all maps below 
cross correlation threshold to bad maps.
Remember to use ccpem-python for this module
"""

input_data = Read_Input('inputs.yaml')
threshold = input_data["threshold"]

xcorr_sort('data', threshold)
print("All maps below threshold have been moved")

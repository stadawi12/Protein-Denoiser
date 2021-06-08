import sys
sys.path.insert(1, 'lib')
sys.path.insert(1, 'lib/utils')
sys.path.insert(1, 'lib/utils/ml_toolbox/src')
sys.path.insert(1, 'lib/net')

from Train import Train
from Inputs import Read_Input
import unet

input_data = Read_Input('inputs.yaml')

Train(unet, input_data, 'data')

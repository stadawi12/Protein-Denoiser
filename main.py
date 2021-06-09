import sys
sys.path.insert(1, 'lib')
sys.path.insert(1, 'lib/utils')
sys.path.insert(1, 'lib/utils/ml_toolbox/src')
sys.path.insert(1, 'lib/net')

from Train import Train
from Process import Process
from Inputs import Read_Input, Create_Parser
import unet


if __name__ == '__main__':

    args = Create_Parser()

    input_data = Read_Input('inputs.yaml')

    if args.action == 'train':

        Train(unet, input_data, 'data')

    if args.action == 'test':

        print("Sorry, come back when this has been developed!")

    if args.action == 'proc':

        p1 = Process(unet, input_data)
        p1.process()




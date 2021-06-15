import sys
sys.path.insert(1, 'lib')
sys.path.insert(1, 'lib/utils')
sys.path.insert(1, 'lib/utils/ml_toolbox/src')
sys.path.insert(1, 'lib/net')

from Train import Train
from Process import Process
from Inputs import Read_Input, Create_Parser
import unet
from sort import Maps


if __name__ == '__main__':

    args = Create_Parser()

    if args.action == 'train':

        Train(unet)

    if args.action == 'test':

        print("Sorry, come back when this has been developed!")

    if args.action == 'proc':

        p1 = Process(unet, input_data)
        p1.process()

    if args.action == 'sortback':
        """moves data from badMaps back to good maps"""
        m1 = Maps('data/1.0/')
        m2 = Maps('data/2.0/')
        m1.move_all_from_bad()
        m2.move_all_from_bad()

    if args.action == 'xcorr':
        """This should automatically send all data below 
        cross correlation threshold to bad maps"""
        print("Sorry, come back when this has been developed!")

    if args.action == 'download':
        """This function should download maps into the directories
        inside data"""
        print("Sorry, come back when this has been developed!")







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

    input_data = Read_Input('inputs.yaml')

    if args.action == 'train':

        Train(unet)

    if args.action == 'test':

        print("Sorry, come back when this has been developed!")

    if args.action == 'proc':

        p1 = Process(unet, input_data)
        p1.process(input_data)

    if args.action == 'moveback':
        """moves data from badMaps back to good maps"""
        m1 = Maps('data/1.0/')
        m2 = Maps('data/2.0/')
        m1.move_all_from_bad()
        m2.move_all_from_bad()
        print("All maps have been moved out of badMaps")
        

    if args.action == 'download':
        import download_data as dd
        """This function should download maps into the directories
        inside data"""

        entries, tails = dd.get_names(input_data, 
                'lib/utils/halfMaps.csv')

        answer = input("Do you want to download these maps? y/n: ")
        if answer == 'y':

            # Ensure badMaps are empty
            m1 = Maps('data/1.0/')
            m2 = Maps('data/2.0/')
            m1.move_all_from_bad()
            m2.move_all_from_bad()

            # Download maps
            for i in range(len(entries)):
                dd.download(entries[i], tails[i])

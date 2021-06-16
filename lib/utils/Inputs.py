import yaml
import argparse

def Read_Input(input_path : str):

    with open(input_path, 'r') as input_file:
        input_data = yaml.load(input_file, Loader=yaml.FullLoader)

    return input_data

def Create_Parser():

    MSG_ACTION = "Select an action to perform"

    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--action', type = str,
            choices = ['train', 'test', 'proc', 'moveback'
                , 'download', 'xcorr'],
            help = MSG_ACTION)

    return parser.parse_args()

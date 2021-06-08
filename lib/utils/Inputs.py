import yaml

def Read_Input(input_path : str):

    with open(input_path, 'r') as input_file:
        input_data = yaml.load(input_file, Loader=yaml.FullLoader)

    return input_data

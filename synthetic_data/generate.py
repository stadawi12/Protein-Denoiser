import sys
sys.path.insert(1, '../lib/utils')

import os
from Inputs import Read_Input

def generate(input_data):
    pdb = input_data['map_name'] + '.ent'
    r   = input_data['res']

    # Check if map already generated
    mrc = pdb[:-3] + 'mrc'
    check_path = os.path.join('maps', str(r), mrc)
    path_exists = os.path.exists(check_path)
    if not path_exists:
        print("Generating density map from model...")

        pdb_path = 'pdbs'
        p = os.path.join('maps', str(r))


        s = "chimera --nogui --script 'chimera_molmap.py %s %d %s.mrc' > /dev/null" % (
            os.path.join(pdb_path, pdb), float(r),
            os.path.join(p, pdb[:-4]))
        os.system(s)
    else: 
        print("Map has already been generated")

if __name__ == '__main__':

    # Read inputs
    input_data = Read_Input('inputs.yaml')

    generate(input_data)


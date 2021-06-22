import sys
sys.path.insert(1, '../lib/utils')

import os
from Inputs import Read_Input

if __name__ == '__main__':

    input_data = Read_Input('inputs.yaml')

    pdb = input_data['pdb']
    r   = input_data['r']

    pdb_path = 'pdbs'
    p = os.path.join('maps', str(r))


    s = "chimera --nogui --script 'chimera_molmap.py %s %d %s.mrc' > /dev/null" % (
        os.path.join(pdb_path, pdb), float(r),
        os.path.join(p, pdb[:-4]))
    os.system(s)

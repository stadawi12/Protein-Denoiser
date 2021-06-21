import os

pdb_path = 'pdbs'
pdb = 'pdb6nyy.ent'
r = '3.0'
p = os.path.join('maps',r)


s = "chimera --nogui --script 'chimera_molmap.py %s %d %s.mrc' > /dev/null" % (
    os.path.join(pdb_path, pdb), float(r),
    os.path.join(p, pdb[:-4]))
os.system(s)

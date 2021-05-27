import chimera
from sys import argv

print(argv)

chimera.runCommand("open %s" % (argv[1]))
chimera.runCommand("molmap #0 %s sigmaFactor %f gridSpacing %f modelId 1" % (argv[2], 0.225, 0.33*float(argv[2])))
chimera.runCommand("volume #1 save %s" % (argv[3]))
chimera.runCommand("stop now")

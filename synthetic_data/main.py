import os

os.system("python3 generate.py")
os.system("python3 add_noise.py")
os.system("python3 denoise.py")
os.system("python3 normalise.py")
os.system("ccpem-python cross_correlate.py")

import sys
sys.path.insert(1, '../lib/utils/ml_toolbox/src')

from proteins import Sample
import numpy as np

denoised = Sample(3.0, 'denoised/3.0/e_20_pdb6nyy.mrc')
noisy = Sample(3.0, 'noisy/3.0/pdb6nyy.mrc')
clean = Sample(3.0, 'maps/3.0/pdb6nyy.mrc')

a, b = np.min(clean.map), np.max(clean.map)
Min_d, Max_d = np.min(denoised.map), np.max(denoised.map)
Min_n, Max_n = np.min(noisy.map), np.max(noisy.map)

denoised.map = (((b-a) * (denoised.map-Min_d)) / \
        (Max_d - Min_d)) + a
denoised.save_map('denoised/3.0/e_20_pdb6nyy.mrc')

noisy.map = (((b-a) * (noisy.map-Min_n)) / \
        (Max_n - Min_n)) + a
noisy.save_map('noisy/3.0/pdb6nyy.mrc')

denoised = Sample(3.0, 'denoised/3.0/e_20_pdb6nyy.mrc')
noisy = Sample(3.0, 'noisy/3.0/pdb6nyy.mrc')


if __name__ == '__main__':

    print(denoised.map.shape)
    print(np.min(denoised.map), np.max(denoised.map))
    print(noisy.map.shape)
    print(np.min(noisy.map), np.max(noisy.map))
    print(clean.map.shape)
    print(np.min(clean.map), np.max(clean.map))

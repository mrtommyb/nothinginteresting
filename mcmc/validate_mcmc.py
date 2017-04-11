import numpy as np
import h5py


def main(fn):
    f = h5py.File(fn)
    g = f['mcmc']['chain'][:]

    print(np.any(g.flatten()) == np.nan)



if __name__ == '__main__':
    filename = 'koi1_np4_priorTrue_dil0.hdf5'
    main(filename)
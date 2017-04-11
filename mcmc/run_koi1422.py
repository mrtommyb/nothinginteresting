from __future__ import division, print_function

#from clean_and_search import Clean
import numpy as np
from scipy.stats import scoreatpercentile as scpc
import h5py
from scipy.stats import nanmedian, nanstd
#import pyfits
#import kplr

import transitemcee_koi1422 as tmod
import emcee
import time as thetime
#import os
from emcee.utils import MPIPool
import sys


def get_lc():
    time,flux,ferr = np.genfromtxt(
        '../data/lc.dat',unpack=True)
    return time,flux,ferr


def main(runmpi=True,nw=100,th=6,bi=10,fr=10,
    starnum=1):

    if runmpi:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool=None

    ldfileloc = '/Users/tom/Projects/koi1422/code/'
    codedir = '/Users/tom/Projects/koi1422/code'


    koi = 1422
    cadence=1625.3
    nplanets = 5
    feh = -0.0821937
    feh_unc = 0.144467

    # KOI-1422A:
    # SPTYPE_STRING   STRING    'M2V '
    # TEFF_ROJAS      FLOAT           3522.09
    # E_TEFF_ROJAS    FLOAT           69.5549
    # EP_TEFF_ROJAS   FLOAT           71.7698
    # EN_TEFF_ROJAS   FLOAT           74.2302
    # FEH_ROJAS       FLOAT        -0.0821937
    # E_FEH_ROJAS     FLOAT          0.144467
    # MH_ROJAS        FLOAT        -0.0615816
    # E_MH_ROJAS      FLOAT          0.138630
    # MSTAR           FLOAT          0.385578
    # E_MSTAR         FLOAT         0.0537308
    # RSTAR           FLOAT          0.374393
    # E_RSTAR         FLOAT         0.0475887
    # ISOCHRONE       STRING    'Dartmouth

    # KOI-1422B:
    # Teff = 3309
    # MSTAR = 0.211
    # logg = 5.05
    # rsun = 0.227
    # rho = 25.5
    # rho_unc = 11.

    if starnum == 1:
        teff = 3474
        teff_unc = 86
        logg = 4.932
        logg_unc = 0.031
        rho = 13.5
        rho_unc = 3.5

        dil = 0.222
        dil_unc = 0.1

    elif starnum == 2:
        teff = 3280
        teff_unc=66
        logg = 5.074
        logg_unc = 0.069
        rho = 29.
        rho_unc = 7.

        dil = 0.778
        dil_unc = 0.10


    period = np.array([
        5.84163568,
        1.98502893e+01,
        1.08643773e+01,
        6.33363220e+01,
        3.41420793e+01])
    impact = np.array([0.18, 0.24, 0.4, 0.4, 0.02])
    T0 = np.array([
        1.35923091e+02,
        1.33649807e+02,
        1.31129034e+02,
        1.62606378e+02,
        1.36036384e+02])
    rprs = (1./np.sqrt(dil)) * np.array([
        0.01311, 0.01348, 0.01048, 0.0110, 0.00959])
    ecosw = np.zeros_like(rprs)
    esinw = np.zeros_like(rprs)
    planet_guess = np.array([
        T0,period,impact,rprs,ecosw,esinw])


    time,flux,ferr = get_lc()

    rho_prior = True
    ldp_prior = False

    nwalkers = nw
    threads = th
    burnin = bi
    fullrun = fr
    thin = 1

    n_ldparams = 2

    toffset_lc = 54832.5
    toffset_rv = 0

    zpt_0 = 1.E-10

    M = tmod.transitemcee_koi1422(
        nplanets,cadence,
        ldfileloc=ldfileloc,codedir=codedir)

    M.get_stellar(teff,
        logg,
        feh,
        n_ldparams,ldp_prior=ldp_prior)

    M.already_open(time,
        flux,ferr,
        timeoffset=toffset_lc,
        normalize=False)

    rho_vals = np.array([rho,rho_unc])

    M.get_rho(rho_vals,rho_prior)
    M.get_zpt(zpt_0)



    if dil is not None:
        M.get_sol(planet_guess,dil=dil)
    else:
        M.get_sol(planet_guess)

    M.cut_non_transit(8)


    outfile = 'koi{0}_np{1}_prior{2}_dil{3}.hdf5'.format(
            koi,nplanets,rho_prior,dil)

    p0 = M.get_guess(nwalkers)

    l_var = np.shape(p0)[1]

    N = len([indval for indval in range(fullrun)
            if indval%thin == 0])
    with h5py.File(outfile, u"w") as f:
        f.create_dataset("time", data=M.time)
        f.create_dataset("flux", data=M.flux)
        f.create_dataset("err", data=M.err)
        f.create_dataset("itime", data=M._itime)
        f.create_dataset("ntt", data = M._ntt)
        f.create_dataset("tobs", data = M._tobs)
        f.create_dataset("omc",data = M._omc)
        f.create_dataset("datatype",data = M._datatype)
        f.attrs["rho_0"] = M.rho_0
        f.attrs["rho_0_unc"] = M.rho_0_unc
        f.attrs["nplanets"] = M.nplanets
        f.attrs["ld1"] = M.ld1
        f.attrs["ld2"] = M.ld2
        f.attrs["koi"] = koi
        f.attrs["dil"] = dil
        g = f.create_group("mcmc")
        g.attrs["nwalkers"] = nwalkers
        g.attrs["burnin"] = burnin
        g.attrs["iterations"] = fullrun
        g.attrs["thin"] = thin
        g.attrs["rho_prior"] = rho_prior
        g.attrs["ldp_prior"] = ldp_prior
        g.attrs["onlytransits"] = M.onlytransits
        g.attrs["tregion"] = M.tregion
        g.attrs["ldfileloc"] = M.ldfileloc
        g.attrs["n_ldparams"] = M.n_ldparams
        g.create_dataset("fixed_sol", data= M.fixed_sol)
        g.create_dataset("fit_sol_0", data= M.fit_sol_0)


        c_ds = g.create_dataset("chain",
            (nwalkers, N, l_var),
            dtype=np.float64)
        lp_ds = g.create_dataset("lnprob",
            (nwalkers, N),
            dtype=np.float64)


        args = [M.nplanets,M.rho_0,M.rho_0_unc,M.rho_prior,
            M.ld1,M.ld1_unc,M.ld2,M.ld2_unc,M.ldp_prior,
            M.flux,M.err,M.fixed_sol,M.time,M._itime,M._ntt,
            M._tobs,M._omc,M._datatype,
            M.n_ldparams,M.ldfileloc,
            M.onlytransits,M.tregion]


        tom = tmod.logchi2


        if runmpi:
            sampler = emcee.EnsembleSampler(nwalkers, l_var, tom,
                args=args,pool=pool)
        else:
            sampler = emcee.EnsembleSampler(nwalkers, l_var, tom,
                args=args,threads=th)


        time1 = thetime.time()
        p2, prob, state = sampler.run_mcmc(p0, burnin,
            storechain=False)
        sampler.reset()
        with h5py.File(outfile, u"a") as f:
            g = f["mcmc"]
            g.create_dataset("burnin_pos", data=p2)
            g.create_dataset("burnin_prob", data=prob)

        time2 = thetime.time()
        print('burn-in took ' + str((time2 - time1)/60.) + ' min')
        time1 = thetime.time()
        for i, (pos, lnprob, state) in enumerate(sampler.sample(p2,
            iterations=fullrun, rstate0=state,
            storechain=False)):

            #do the thinning in the loop here
            if i % thin == 0:
                ind = i / thin
                with h5py.File(outfile, u"a") as f:
                    g = f["mcmc"]
                    c_ds = g["chain"]
                    lp_ds = g["lnprob"]
                    c_ds[:, ind, :] = pos
                    lp_ds[:, ind] = lnprob

        time2 = thetime.time()
        print('MCMC run took ' + str((time2 - time1)/60.) + ' min')
        print('')
        print("Mean acceptance: "
            + str(np.mean(sampler.acceptance_fraction)))
        print('')

        if runmpi:
            pool.close()
        else:
            sampler.pool.close()

        return sampler

if __name__ == '__main__':
    sampler = main(runmpi=True,nw=70,th=1,bi=1,fr=1,
        starnum=2)



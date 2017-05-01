from __future__ import division, print_function

import numpy as np
import h5py

import transitemcee_koi1422 as tmod

import emcee
import time as thetime

from emcee.utils import MPIPool
import sys

def get_lc():
    time,flux = np.genfromtxt(
        '../data/lc-v2.txt',unpack=True)
    flux -= 1.0
    ferr = np.zeros_like(time) + 0.0001
    return time,flux,ferr

def main(runmpi=True,nw=100,th=6,bi=10,fr=10, circ=False):

    print('run in python3 env')

    if runmpi:
        pool = MPIPool()
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
    else:
        pool=None


    ldfileloc = './'
    codedir = '.'


    koi = 1
    cadence=1625.3
    nplanets = 4
    feh = 0.01
    feh_unc = 0.3

    teff = 5189
    teff_unc = 250
    logg = 4.557
    logg_unc = 0.2
    rho = 2.2
    rho_unc = 0.5

    dil = 0
    dil_unc = 0.0

    period = np.array([
        3.5599989749333116,
        5.404802096830052,
        8.26135525389768,
        12.757074436863912])
    impact = np.array([0.4, 0.11, 0.38, 0.58])
    T0 = np.array([
        1.9440373393339179,
        4.780461299317543,
        2.2702362924313317,
        0.3278855261790516])
    rprs = np.array([
        0.0203972780893194, 
        0.02307855413116277, 
        0.03112935384777642, 
        0.025237445669347654])
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

    toffset_lc = 2457738.376807
    toffset_rv = 0

    zpt_0 = 1.E-10

    M = tmod.transitemcee_koi1422(
        nplanets,cadence,
        ldfileloc=ldfileloc,codedir=codedir)


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

    outfile = 'koi{0}_np{1}_prior{2}_dil{3}_circ{4}.hdf5'.format(
        koi,nplanets,rho_prior,dil,circ)

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


        tom = tmod.logchi2_circ

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
        if th > 1:
            sampler.pool.close()

    return sampler

if __name__ == '__main__':
    sampler = main(runmpi=True,nw=270,th=4,bi=1,fr=6000)

    sys.exit()




from __future__ import division, print_function
import transitemcee
from transitemcee import get_ar
import numpy as np
from copy import deepcopy
from scipy.stats import truncnorm
from claretquadpy import claretquad
from claret4ppy import claretlimb4p
from numpy import random
import time as thetime
import emcee
import tmodtom as tmod
import sys

from scipy.special import beta

class transitemcee_koi1422(transitemcee.transitemcee_fitldp):

    def __init__(self,nplanets,cadence=1625.3,
        ldfileloc='/Users/tom/svn_code/tom_code/',
        codedir='/Users/tom/svn_code/tom_code/'):
        super(transitemcee_koi1422,self).__init__(nplanets,
            cadence=cadence,
            ldfileloc=ldfileloc,
            codedir=codedir)
        sys.path.append(codedir)

    def get_sol(self,args,**kwargs):
        tom = args
        assert np.shape(tom) == (6,self.nplanets)

        if 'dil' in kwargs.keys():
            dil = kwargs['dil']
            print(' running with dil = %s' %(dil))
        else:
            dil = 0.0
        if 'veloffset' in kwargs.keys():
            veloffset = kwargs['veloffset']
        else:
            veloffset = 0.0
        if 'rvamp' in kwargs.keys():
            rvamp = kwargs['rvamp']
        else:
            rvamp = 0.0
        if 'occ' in kwargs.keys():
            occ = kwargs['occ']
        else:
            occ = 0.0
        if 'ell' in kwargs.keys():
            ell = kwargs['ell']
        else:
            ell = 0.0
        if 'alb' in kwargs.keys():
            alb = kwargs['alb']
        else:
            alb = 0.0

        try:
            if self.zpt_0 == 0.:
                self.zpt_0 = 1.E-10
        except AttributeError:
            self.zpt_0 = 1.E-10
            self.zpt_0_unc = 1.E-6

        fit_sol = np.array([self.rho_0,self.zpt_0,
            self.ld1,self.ld2])

        for i in range(self.nplanets):
            T0_0 = args[0,i]
            per_0 = args[1,i]
            b_0 = args[2,i]
            rprs_0 = args[3,i]
            ecosw_0 = args[4,i]
            esinw_0 = args[5,i]

            new_params = np.array([T0_0,per_0,
                b_0,rprs_0,ecosw_0,esinw_0])
            fit_sol = np.r_[fit_sol,new_params]

        self.fit_sol = fit_sol
        self.fit_sol_0 = deepcopy(self.fit_sol)

        self.fixed_sol = np.array([dil,veloffset,rvamp,
            occ,ell,alb])


    def get_guess(self,nwalkers):
        """
        pick sensible starting ranges for the guess parameters
        T0, period, impact paramter, rp/rs, ecosw and esinw
        """
        rho_unc = 0.001
        zpt_unc = 1.E-9
        ld1_unc = 0.02
        ld2_unc = 0.02
        T0_unc = 0.0002
        per_unc = 0.00001
        b_unc = 0.001
        rprs_unc = 0.0001
        ecosw_unc = 0.001
        esinw_unc = 0.001

        p0 = np.zeros([nwalkers,4+self.nplanets*6+1])

        rho = self.fit_sol[0]
        zpt = self.fit_sol[1]
        ld1 = self.fit_sol[2]
        ld2 = self.fit_sol[3]

        start,stop = ((0.0001 - rho) / rho_unc,
            (50.0 - rho) / rho_unc)
        p0[...,0] = truncnorm.rvs(start,stop
                ,loc=rho,scale=rho_unc,size=nwalkers)

        p0[...,1] = np.random.normal(loc=zpt,scale=zpt,
            size=nwalkers)

        start,stop = ((0.0 - ld1) / ld1_unc,
            (1.0 - ld1) / ld1_unc)
        p0[...,2] = truncnorm.rvs(start,stop
                ,loc=ld1,scale=ld1_unc,size=nwalkers)

        start,stop = ((0.0 - ld2) / ld2_unc,
            (1.0 - ld2) / ld2_unc)
        p0[...,3] = truncnorm.rvs(start,stop
                ,loc=ld2,scale=ld2_unc,size=nwalkers)

        for i in range(self.nplanets):
            (T0,per,b,rprs,ecosw,
                esinw) = self.fit_sol[i*6+4:i*6+10]

            b = 0.2
            ecosw = 0.0
            esinw = 0.0
            p0[...,i*6+4] = np.random.normal(
                T0,T0_unc,size=nwalkers)
            p0[...,i*6+4+1] = np.random.normal(
                per,per_unc,size=nwalkers)
            start,stop = (0.0 - b) / b_unc, (1.0 - b) / b_unc
            p0[...,i*6+4+2] = truncnorm.rvs(
                start,stop,loc=b,scale=b_unc,size=nwalkers)
            start,stop = (0.0 - rprs) / rprs_unc, (0.5 - rprs) / rprs_unc
            p0[...,i*6+4+3] = truncnorm.rvs(
                start,stop,loc=rprs,scale=rprs_unc,size=nwalkers)
            start,stop = (-0.2 - ecosw) / ecosw_unc, (0.2 - ecosw) / ecosw_unc
            p0[...,i*6+4+4] = truncnorm.rvs(
                start,stop,loc=ecosw,scale=ecosw_unc,size=nwalkers)
            start,stop = (-0.2 - esinw) / esinw_unc, (0.2 - esinw) / esinw_unc
            p0[...,i*6+4+5] = truncnorm.rvs(
                start,stop,loc=esinw,scale=esinw_unc,size=nwalkers)


        #lcjitter
        start,stop = 0.0, 10.
        p0[...,-1] = truncnorm.rvs(start,stop,
            loc=0.0,scale=0.1*np.median(self.err),size=nwalkers)
        return p0




def logchi2(fitsol,nplanets,rho_0,rho_0_unc,rho_prior,
    ld1_0,ld1_0_unc,ld2_0,ld2_0_unc,ldp_prior,
    flux,err,fixed_sol,time,itime,ntt,tobs,omc,datatype,
    n_ldparams=2,ldfileloc='/Users/tom/svn_code/tom_code/',
    onlytransits=False,tregion=0.0):

    """
    fitsol should have the format
    rho_0,zpt_0,ld1,ld2
    plus for each planet
    T0_0,per_0,b_0,rprs_0,ecosw_0,esinw_0

    fixed_sol should have
    dil, veloffset,rvamp,
            occ,ell,alb

    """
    minf = -np.inf

    rho = fitsol[0]
    if rho < 1.E-8 or rho > 100.:
        return minf

    zpt = fitsol[1]
    if np.abs(zpt) > 1.E-2:
        return minf


    ld1 = fitsol[2]
    ld2 = fitsol[3]
    #some lind darkening constraints
    #from Burke et al. 2008 (XO-2b)
    if ld1 < 0.0:
        return minf
    if ld1 + ld2 > 1.0:
        return minf
    if ld1 + 2.*ld2 < 0.0:
        return minf
    if ld2 < -0.8:
        return minf

    if n_ldparams == 2:
        ld3, ld4 = 0.0,0.0


    #T0, period, b, rprs, ecosw, esinw

    rprs = fitsol[np.arange(nplanets)*6 + 7]
    if np.any(rprs < 0.) or np.any(rprs > 0.5):
        return minf

    ecosw = fitsol[np.arange(nplanets)*6 + 8]
    if np.any(ecosw < -1.0) or np.any(ecosw > 1.0):
        return minf

    esinw = fitsol[np.arange(nplanets)*6 + 9]
    if np.any(esinw < -1.0) or np.any(esinw > 1.0):
        return minf

    #avoid parabolic orbits
    ecc = np.sqrt(esinw**2 + ecosw**2)
    if np.any(ecc > 1.0):
        return minf

    #avoid orbits where the planet enters the star
    per = fitsol[np.arange(nplanets)*6 + 5]
    ar = get_ar(rho,per)
    if np.any(ecc > (1.-(1./ar))):
        return minf

    b = fitsol[np.arange(nplanets)*6 + 6]
    if np.any(b < 0.) or np.any(b > 1.0 + rprs):
        return minf

    if onlytransits:
        T0 = fitsol[np.arange(nplanets)*6 + 4]
        if np.any(T0 < T0 - tregion) or np.any(T0 > T0 + tregion):
            return minf


    jitter_lc = fitsol[-1]


    ### eccentricity max = 0.337
    ### avoids crossing orbits
    if np.any(ecc > 0.337):
        return minf



    if jitter_lc < 0.0:
        return minf
    err_jit = np.sqrt(err**2 + jitter_lc**2)
    err_jit2 = err**2 + jitter_lc**2

    lds = np.array([ld1,ld2,ld3,ld4])

    #need to do some funky stuff
    #so I can use the same calc_model as
    #the other transitemcee routines

    fitsol_model_calc = np.r_[fitsol[0:2],fitsol[4:]] #cut out limb darkening
    fixed_sol_model_calc = np.r_[lds,fixed_sol]

    time_model_calc = time
    itime_model_calc = itime
    datatype_model_calc  = datatype

    model_lcrv = calc_model(fitsol_model_calc,
        nplanets,fixed_sol_model_calc,
        time_model_calc,itime_model_calc,ntt,tobs,omc,datatype_model_calc)

    model_lc = model_lcrv - 1.


    ecc[ecc == 0.0] = 1.E-10

    npt_lc = len(err_jit)



    loglc = (
        - (npt_lc/2.)*np.log(2.*np.pi)
        - 0.5 * np.sum(np.log(err_jit2))
        - 0.5 * np.sum((model_lc - flux)**2 / err_jit2)
        )


    if rho_prior:
        logrho = (
            - 0.5 * np.log(2.*np.pi)
            - np.log(rho_0_unc)
            - 0.5 * (rho_0 - rho)**2 / rho_0_unc**2
            )
    else:
        logrho = 0.0

    if ldp_prior:
        logld1 = (
            - 0.5 * np.log(2.*np.pi)
            - np.log(ld1_0_unc)
            - 0.5 * (ld1_0 - ld1)**2 / ld1_0_unc**2
            )

        logld2 = (
            - 0.5 * np.log(2.*np.pi)
            - np.log(ld2_0_unc)
            - 0.5 * (ld2_0 - ld2)**2 / ld2_0_unc**2
            )

        logldp = logld1 + logld2
    else:
        logldp = 0.0

    logecc = - np.sum(np.log(ecc))

    # lets do my sick beta distribution in ecc
    logecc2 = np.sum(np.log(beta_func(ecc)))

    logLtot = loglc + logrho + logldp + logecc + logecc2

    return logLtot

def logchi2_circ(fitsol,nplanets,rho_0,rho_0_unc,rho_prior,
    ld1_0,ld1_0_unc,ld2_0,ld2_0_unc,ldp_prior,
    flux,err,fixed_sol,time,itime,ntt,tobs,omc,datatype,
    n_ldparams=2,ldfileloc='/Users/tom/svn_code/tom_code/',
    onlytransits=False,tregion=0.0):

    """
    fitsol should have the format
    rho_0,zpt_0,ld1,ld2
    plus for each planet
    T0_0,per_0,b_0,rprs_0,ecosw_0,esinw_0

    fixed_sol should have
    dil, veloffset,rvamp,
            occ,ell,alb

    """
    minf = -np.inf

    rho = fitsol[0]
    if rho < 1.E-8 or rho > 100.:
        return minf

    zpt = fitsol[1]
    if np.abs(zpt) > 1.E-2:
        return minf


    ld1 = fitsol[2]
    ld2 = fitsol[3]
    #some lind darkening constraints
    #from Burke et al. 2008 (XO-2b)
    if ld1 < 0.0:
        return minf
    if ld1 + ld2 > 1.0:
        return minf
    if ld1 + 2.*ld2 < 0.0:
        return minf
    if ld2 < -0.8:
        return minf

    if n_ldparams == 2:
        ld3, ld4 = 0.0,0.0


    #T0, period, b, rprs, ecosw, esinw

    rprs = fitsol[np.arange(nplanets)*6 + 7]
    if np.any(rprs < 0.) or np.any(rprs > 0.5):
        return minf

    ecosw = 0.0

    esinw = 0.0

    #avoid parabolic orbits
    ecc = 0.0

    #avoid orbits where the planet enters the star
    per = fitsol[np.arange(nplanets)*6 + 5]
    ar = get_ar(rho,per)
    if np.any(ecc > (1.-(1./ar))):
        return minf

    b = fitsol[np.arange(nplanets)*6 + 6]
    if np.any(b < 0.) or np.any(b > 1.0 + rprs):
        return minf

    if onlytransits:
        T0 = fitsol[np.arange(nplanets)*6 + 4]
        if np.any(T0 < T0 - tregion) or np.any(T0 > T0 + tregion):
            return minf


    jitter_lc = fitsol[-1]


    ### eccentricity max = 0.337
    ### avoids crossing orbits
    if np.any(ecc > 0.337):
        return minf



    if jitter_lc < 0.0:
        return minf
    err_jit = np.sqrt(err**2 + jitter_lc**2)
    err_jit2 = err**2 + jitter_lc**2

    lds = np.array([ld1,ld2,ld3,ld4])

    #need to do some funky stuff
    #so I can use the same calc_model as
    #the other transitemcee routines

    fitsol_model_calc = np.r_[fitsol[0:2],fitsol[4:]] #cut out limb darkening
    fixed_sol_model_calc = np.r_[lds,fixed_sol]

    time_model_calc = time
    itime_model_calc = itime
    datatype_model_calc  = datatype

    model_lcrv = calc_model(fitsol_model_calc,
        nplanets,fixed_sol_model_calc,
        time_model_calc,itime_model_calc,ntt,tobs,omc,datatype_model_calc)

    model_lc = model_lcrv - 1.



    npt_lc = len(err_jit)



    loglc = (
        - (npt_lc/2.)*np.log(2.*np.pi)
        - 0.5 * np.sum(np.log(err_jit2))
        - 0.5 * np.sum((model_lc - flux)**2 / err_jit2)
        )


    if rho_prior:
        logrho = (
            - 0.5 * np.log(2.*np.pi)
            - np.log(rho_0_unc)
            - 0.5 * (rho_0 - rho)**2 / rho_0_unc**2
            )
    else:
        logrho = 0.0

    if ldp_prior:
        logld1 = (
            - 0.5 * np.log(2.*np.pi)
            - np.log(ld1_0_unc)
            - 0.5 * (ld1_0 - ld1)**2 / ld1_0_unc**2
            )

        logld2 = (
            - 0.5 * np.log(2.*np.pi)
            - np.log(ld2_0_unc)
            - 0.5 * (ld2_0 - ld2)**2 / ld2_0_unc**2
            )

        logldp = logld1 + logld2
    else:
        logldp = 0.0

    

    logLtot = loglc + logrho + logldp 

    return logLtot

def beta_func(ec,x=0.44969629,y=1.79381137):
    p = (1./beta(x,y)) * ec**(x-1) * (1-ec)**(y-1)
    return p


def calc_model(fitsol,nplanets,fixed_sol,
    time,itime,ntt,tobs,omc,datatype):
    """
    what is in fitsol??
    rho, zpt

    what is in fixed_sol??
    ld1,ld2,ld3,ld4, dil, veloffset,rvamp,
            occ,ell,alb
    """
    sol = np.zeros([8 + 10*nplanets])
    rho = fitsol[0]
    zpt = fitsol[1]
    ld1,ld2,ld3,ld4 = fixed_sol[0:4]
    dil = fixed_sol[4]
    veloffset = fixed_sol[5]


    sol[0:8] = np.array([rho,ld1,ld2,ld3,ld4,
        dil,veloffset,zpt])
    for i in range(nplanets):
        sol[8+(i*10):8+(i*10)+10] = np.r_[
            fitsol[2+i*6:8+i*6], fixed_sol[6:]]

    tmodout = tmod.transitmodel(nplanets,sol,time,itime,
        ntt,tobs,omc,datatype)

    return tmodout


class transitemcee_koi1422_special(transitemcee_koi1422):

    def __init__(self,nplanets,cadence=1625.3,
        ldfileloc='/Users/tom/svn_code/tom_code/',
        codedir='/Users/tom/svn_code/tom_code/'):
        super(transitemcee_koi1422,self).__init__(nplanets,
            cadence=cadence,
            ldfileloc=ldfileloc,
            codedir=codedir)
        sys.path.append(codedir)

    def get_sol(self,args,**kwargs):
        tom = args
        assert np.shape(tom) == (6,self.nplanets)


        print('running with variable dil')
        dil = 0.222

        if 'veloffset' in kwargs.keys():
            veloffset = kwargs['veloffset']
        else:
            veloffset = 0.0
        if 'rvamp' in kwargs.keys():
            rvamp = kwargs['rvamp']
        else:
            rvamp = 0.0
        if 'occ' in kwargs.keys():
            occ = kwargs['occ']
        else:
            occ = 0.0
        if 'ell' in kwargs.keys():
            ell = kwargs['ell']
        else:
            ell = 0.0
        if 'alb' in kwargs.keys():
            alb = kwargs['alb']
        else:
            alb = 0.0

        try:
            if self.zpt_0 == 0.:
                self.zpt_0 = 1.E-10
        except AttributeError:
            self.zpt_0 = 1.E-10
            self.zpt_0_unc = 1.E-6

        fit_sol = np.array([self.rho_0,self.zpt_0,
            self.ld1,self.ld2])

        for i in range(self.nplanets):
            T0_0 = args[0,i]
            per_0 = args[1,i]
            b_0 = args[2,i]
            rprs_0 = args[3,i]
            ecosw_0 = args[4,i]
            esinw_0 = args[5,i]

            new_params = np.array([T0_0,per_0,
                b_0,rprs_0,ecosw_0,esinw_0])
            fit_sol = np.r_[fit_sol,new_params]

        self.fit_sol = fit_sol
        self.fit_sol_0 = deepcopy(self.fit_sol)

        self.fixed_sol = np.array([dil,veloffset,rvamp,
            occ,ell,alb])


    def get_guess(self,nwalkers):
        """
        pick sensible starting ranges for the guess parameters
        T0, period, impact paramter, rp/rs, ecosw and esinw
        """
        rho_unc = 0.001
        zpt_unc = 1.E-9
        ld1_unc = 0.02
        ld2_unc = 0.02
        T0_unc = 0.0002
        per_unc = 0.00001
        b_unc = 0.001
        rprs_unc = 0.0001
        ecosw_unc = 0.001
        esinw_unc = 0.001
        dil_unc = 0.05

        #add a parameter for the dilution
        p0 = np.zeros([nwalkers,4+self.nplanets*6+1+1])

        rho = self.fit_sol[0]
        zpt = self.fit_sol[1]
        ld1 = self.fit_sol[2]
        ld2 = self.fit_sol[3]
        dil = self.fixed_sol[0]

        start,stop = ((0.0001 - rho) / rho_unc,
            (50.0 - rho) / rho_unc)
        p0[...,0] = truncnorm.rvs(start,stop
                ,loc=rho,scale=rho_unc,size=nwalkers)

        p0[...,1] = np.random.normal(loc=zpt,scale=zpt,
            size=nwalkers)

        start,stop = ((0.0 - ld1) / ld1_unc,
            (1.0 - ld1) / ld1_unc)
        p0[...,2] = truncnorm.rvs(start,stop
                ,loc=ld1,scale=ld1_unc,size=nwalkers)

        start,stop = ((0.0 - ld2) / ld2_unc,
            (1.0 - ld2) / ld2_unc)
        p0[...,3] = truncnorm.rvs(start,stop
                ,loc=ld2,scale=ld2_unc,size=nwalkers)

        for i in range(self.nplanets):
            (T0,per,b,rprs,ecosw,
                esinw) = self.fit_sol[i*6+4:i*6+10]

            b = 0.2
            ecosw = 0.0
            esinw = 0.0
            p0[...,i*6+4] = np.random.normal(
                T0,T0_unc,size=nwalkers)
            p0[...,i*6+4+1] = np.random.normal(
                per,per_unc,size=nwalkers)
            start,stop = (0.0 - b) / b_unc, (1.0 - b) / b_unc
            p0[...,i*6+4+2] = truncnorm.rvs(
                start,stop,loc=b,scale=b_unc,size=nwalkers)
            start,stop = (0.0 - rprs) / rprs_unc, (0.5 - rprs) / rprs_unc
            p0[...,i*6+4+3] = truncnorm.rvs(
                start,stop,loc=rprs,scale=rprs_unc,size=nwalkers)
            start,stop = (-0.2 - ecosw) / ecosw_unc, (0.2 - ecosw) / ecosw_unc
            p0[...,i*6+4+4] = truncnorm.rvs(
                start,stop,loc=ecosw,scale=ecosw_unc,size=nwalkers)
            start,stop = (-0.2 - esinw) / esinw_unc, (0.2 - esinw) / esinw_unc
            p0[...,i*6+4+5] = truncnorm.rvs(
                start,stop,loc=esinw,scale=esinw_unc,size=nwalkers)

        #dil
        start,stop = 1.E-10, 1.-1.E-10
        p0[...,-2] = truncnorm.rvs(start,stop,
            loc=dil,scale=dil_unc,size=nwalkers)

        #lcjitter
        start,stop = 0.0, 10.
        p0[...,-1] = truncnorm.rvs(start,stop,
            loc=0.0,scale=0.1*np.median(self.err),size=nwalkers)
        return p0


def logchi2_special(fitsol,nplanets,rho_0,rho_0_unc,rho_prior,
    ld1_0,ld1_0_unc,ld2_0,ld2_0_unc,ldp_prior,
    flux,err,fixed_sol,time,itime,ntt,tobs,omc,datatype,
    n_ldparams=2,ldfileloc='/Users/tom/svn_code/tom_code/',
    onlytransits=False,tregion=0.0):

    """
    fitsol should have the format
    rho_0,zpt_0,ld1,ld2
    plus for each planet
    T0_0,per_0,b_0,rprs_0,ecosw_0,esinw_0

    fixed_sol should have
    dil, veloffset,rvamp,
            occ,ell,alb

    """
    minf = -np.inf

    rho = fitsol[0]
    if rho < 1.E-8 or rho > 200.:
        return minf

    zpt = fitsol[1]
    if np.abs(zpt) > 1.E-2:
        return minf


    ld1 = fitsol[2]
    ld2 = fitsol[3]
    #some lind darkening constraints
    #from Burke et al. 2008 (XO-2b)
    if ld1 < 0.0:
        return minf
    if ld1 + ld2 > 1.0:
        return minf
    if ld1 + 2.*ld2 < 0.0:
        return minf
    if ld2 < -0.8:
        return minf

    if n_ldparams == 2:
        ld3, ld4 = 0.0,0.0


    #T0, period, b, rprs, ecosw, esinw

    rprs = fitsol[np.arange(nplanets)*6 + 7]
    if np.any(rprs < 0.) or np.any(rprs > 0.5):
        return minf

    ecosw = fitsol[np.arange(nplanets)*6 + 8]
    if np.any(ecosw < -1.0) or np.any(ecosw > 1.0):
        return minf

    esinw = fitsol[np.arange(nplanets)*6 + 9]
    if np.any(esinw < -1.0) or np.any(esinw > 1.0):
        return minf

    #avoid parabolic orbits
    ecc = np.sqrt(esinw**2 + ecosw**2)
    if np.any(ecc > 1.0):
        return minf

    #avoid orbits where the planet enters the star
    per = fitsol[np.arange(nplanets)*6 + 5]
    ar = get_ar(rho,per)
    if np.any(ecc > (1.-(1./ar))):
        return minf

    b = fitsol[np.arange(nplanets)*6 + 6]
    if np.any(b < 0.) or np.any(b > 1.0 + rprs):
        return minf

    if onlytransits:
        T0 = fitsol[np.arange(nplanets)*6 + 4]
        if np.any(T0 < T0 - tregion) or np.any(T0 > T0 + tregion):
            return minf

    dil = fitsol[-2]
    if dil <= 0.0 or dil >= 1.0:
        return minf

    jitter_lc = fitsol[-1]


    ### eccentricity max = 0.337
    ### avoids crossing orbits
    if np.any(ecc > 0.337):
        return minf



    if jitter_lc < 0.0:
        return minf
    err_jit = np.sqrt(err**2 + jitter_lc**2)
    err_jit2 = err**2 + jitter_lc**2

    lds = np.array([ld1,ld2,ld3,ld4])

    #need to do some funky stuff
    #so I can use the same calc_model as
    #the other transitemcee routines

    fitsol_model_calc = np.r_[fitsol[0:2],fitsol[4:]] #cut out limb darkening
    fixed_sol_model_calc = np.r_[lds,dil,fixed_sol[1:]]

    time_model_calc = time
    itime_model_calc = itime
    datatype_model_calc  = datatype

    model_lcrv = calc_model(fitsol_model_calc,
        nplanets,fixed_sol_model_calc,
        time_model_calc,itime_model_calc,ntt,tobs,omc,datatype_model_calc)

    model_lc = model_lcrv - 1.


    ecc[ecc == 0.0] = 1.E-10

    npt_lc = len(err_jit)



    loglc = (
        - (npt_lc/2.)*np.log(2.*np.pi)
        - 0.5 * np.sum(np.log(err_jit2))
        - 0.5 * np.sum((model_lc - flux)**2 / err_jit2)
        )


    # these shouldn't have a prior
    logrho = 0.0
    logldp = 0.0

    logecc = - np.sum(np.log(ecc))

    # lets do my sick beta distribution in ecc
    logecc2 = np.sum(np.log(beta_func(ecc)))

    logLtot = loglc + logrho + logldp + logecc + logecc2

    return logLtot




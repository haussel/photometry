__author__ = 'haussel'
"""
This module provides function to obtain standard spectra.
"""
import numpy as np
import os
import inspect
from astropy import units as u
from astropy import constants as const
from astropy.table import Table
from .spectrum import BasicSpectrum
from .phottools import is_frequency, is_wavelength, is_flam, is_fnu, \
    quantity_scalar, quantity_1darray, quantity_2darray,\
    velc, nu_unit, lam_unit, flam_unit, fnu_unit


def power_law(x0, f0, alpha=None, x1=None, f1=None):
    """
    Returns a BasicSpectrum with a power law.

    The spectrum type depends on the units of the input. It will be:
        fnu = fnu0 (nu/nu_0)**alpha
    or  flam = flam0 (lam/lam0)**alpha

    Instead of alpha, other x and f values can be passed with the parameters
    x1 and f1. If none is give, returns a flat spectrum and issues a warning.

    Parameters
    ----------
    x0 : astropy.units.quantity
      The reference frequency (nu_0) or wavelength (lam_0), depending on its
      unit
    f0 : astropy.units.quantity
      The reference specral irradiance per unit of frequency (i.e. fnu) or
      wavelength (i.e. flam) depending on its unit
    alpha: float or (N, ) ndarray
      The slope(s) of the power law.
    x1 : float or (N,) ndarray
      The other frequencies or wavelengths
    f1 : float or (N,) ndarray
      The other fluxes

    Returns
    -------
    output: BasicSpectrum
    """
    if is_frequency(x0.unit):
        if is_fnu(f0.unit):
            if x1 is None and f1 is None:
                if alpha is None:
                    walpha = 0
                else:
                    walpha = alpha
                nu = x0.value * np.logspace(-2,2, 5)
            elif x1 is not None and f1 is not None:
                if alpha is not None:
                    raise ValueError("Incompatible parameters alpha and x1,f1")
                else:
                    if is_frequency(x1.unit):
                        if is_fnu(f1.unit):
                            walpha = np.log10(f1/f0) / np.log10(x1/x0)
                            if isinstance(x1, np.ndarray):
                                nu = np.unique(np.hstack((x0.value,
                                                          (x1.to(x0.unit)).\
                                                          value)))
                            else:
                                nu = np.sort(np.array([x0, x1]))
                        else:
                            ValueError("incompatible units between f0 and f1")
                    else:
                        ValueError("incompatible units between x0 and x1")
            else:
                raise ValueError("missing input parameters")
            if isinstance(walpha, np.ndarray):
                fnu = (f0.value * (nu[:, np.newaxis]/x0.value)**alpha.T).T
            else:
                fnu = f0.value * (nu/x0.value)**alpha
#            print("nu = {}".format(nu * x0.unit))
#            print("fnu.shape = {}".format(fnu.shape))
#            print("fnu = {}".format(fnu * f0.unit))
            result = BasicSpectrum(x=nu * x0.unit, y=fnu * f0.unit,
                                   interpolation_method='log-log-linear',
                                   extrapolate=True)
        else:
            raise ValueError("incompatible units between x_0 and f_0")
    elif is_wavelength(x0.unit):
        if is_flam(f0.unit):
            if x1 is None and f1 is None:
                if alpha is None:
                    walpha = 0
                else:
                    walpha = alpha
                lam = x0.value * np.logspace(-2,2, 5)
            elif x1 is not None and f1 is not None:
                if alpha is not None:
                    raise ValueError("Incompatible parameters alpha and x1,f1")
                else:
                    if is_wavelength(x1.unit):
                        if is_flam(f1.unit):
                            walpha = (np.log10(f1/f0) / np.log10(x1/x0)).value
                            if isinstance(x1, np.ndarray):
                                lam = np.unique(np.hstack((x0.value,
                                                           (x1.to(x0.unit)).\
                                                           value)))
                            else:
                                lam = np.sort(np.array([x0, x1]))
                        else:
                            ValueError("incompatible units between f0 and f1")
                    else:
                        ValueError("incompatible units between x0 and x1")
            else:
                raise ValueError("missing input parameters")
            if isinstance(walpha, np.ndarray):
                flam = (f0.value * (lam[:, np.newaxis]/x0.value)**alpha.T).T
            else:
                flam = f0.value * (lam/x0.value)**alpha
            print("lam = {}".format(lam * x0.unit))
            print("flam.shape = {}".format(flam.shape))
            print("flam = {}".format(flam * f0.unit))
            result = BasicSpectrum(x=lam * x0.unit, y=flam * f0.unit,
                                   interpolation_method='log-log-linear',
                                   extrapolate=True)
        else:
            raise ValueError("incompatible units between x_0 and f_0")
    else:
        raise ValueError("Invalid unit for parameter x_0")
    return result





def vega_cohen_1992():
    """

    :return:
    """
    dummy = BasicSpectrum()
    classpath = inspect.getfile(dummy.__class__)
    basepath = os.path.dirname(classpath)
    default_vega_dir = os.path.join(basepath, 'data/spectra/vega')
    vegafile = os.path.join(default_vega_dir, 'alp_lyr.cohen_1992')
    data = np.genfromtxt(vegafile)
    result = BasicSpectrum(x=data[:,0], x_type='lam', x_unit='micron',
                           y=data[:,1], y_type='flam', y_unit='W/cm**2/micron')
    # TBC
    return result

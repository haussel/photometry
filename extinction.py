"""
This module provides the following classes:
- Extinction
"""
__author__ = 'Herve Aussel'

import re
import os
import inspect
import numpy as np
from scipy import integrate
from astropy import units as u
from .phottools import is_wavelength, is_frequency, velc, \
    PhotometryInterpolator, ndarray_1darray
# from .spectrum import BasicSpectrum
from .config import DEBUG


class ExtinctionLaw:
    """
    A class to implement various extinction law of the classical form:
    A_lambda / A_V = a(x) + b(x) / R_V

    with R_V = A_V / E(B-V)

    where x = 1/lambda is expressed in microns**-1

    Published extinction laws provide a fit of the values of a and b as a
    function of x in various domains of validity, so an extinction law is
    specified by the limits of x and the corresponding functions to compute a
    and b.

    Attributes
    ----------
    _xlims: numpy.ndarray
        boundaries defining the intervals of x. Stored as inverse microns
    _a_extfunc: list of fuctions
        functions allowing to compute a(x) from x
    _b_extfunc:  list of fuctions
        functions allowing to compute b(x) from x
    x: numpy.ndarray
        values of the wavenumber
    a: numpy.ndarray
        values of a(x)
    b: numpy.ndarray
        values of b(x)
    a_b_set: bool
        True if a and b have been computed
    R_V: float
        Value of the R_V parameter
    R_V_set: bool
        True if R_V as been set.
    Alam_AV: PhotometryInterpolator
        Allow to compute Alam/AV for any x
    ready: bool
        True if ready to be used

    Methods
    -------
    set_curves(x): computes a, b and x and sets a_b_set, if R_V set, computes
        Alam_AV
    input2wavenumber(x): convert wavelength, frequency or wavenumber to
        wavenumber
    set_R_V(R_V): set R_V. If a_b_set, computes Alam_AV
    compute_extinction(x, Av, recompute=False, Rv=None): derive the
        extinction at x for given Av

    """
    def __init__(self):
        self._xlims = None
        self._a_extfunc = None
        self._b_extfunc = None
        self.x = None
        self.a = None
        self.b = None
        self.a_b_set = False
        self.R_V = None
        self.R_V_set = False
        self.Alam_AV = None
        self.ready = False

    def input2wavenumber(self, x):
        """
        Convert input to wavenumber

        Parameters:
        -----------
        x : astropy.quantity
            input wavelength, frequency or wavenumber

        Returns:
        -------
        numpy.ndarray
            the wavenumber in micron-1
        """
        if not isinstance(x, u.Quantity):
            raise ValueError('x is not a quantity')
        if is_frequency(x.unit):
            wx = (x.to(Hz) / (velc * u.m).to(u.micron)).value
        elif is_wavelength(x.unit):
            wx = (1. / x.to(u.micron)).value
        elif is_wavelength(1/x.unit):
            wx = (x.to(u.micron**-1)).value
        else:
            raise ValueError("x has to be a wavelength, a frequency or a "
                             "wave number")
        return wx

    def set_curve(self, x):
        self.x = self.input2wavenumber(x)
        self.a = np.zeros(len(self.x)) * np.NaN
        self.b = np.zeros(len(self.x)) * np.NaN
        ids = np.digitize(self.x, self._xlims)
        uids = np.unique(ids)
        for uid in uids:
            idx, = np.where(ids == uid)
            if DEBUG:
                print("{} - > {} : {}".format(uid, idx, self._a_extfunc[uid-1]))
            self.a[idx] = (self._a_extfunc[uid-1])(self.x[idx])
            self.b[idx] = (self._b_extfunc[uid-1])(self.x[idx])
        self.a_b_set = True
        if self.R_V_set:
            self.set_R_V(self.R_V)

    def set_R_V(self, R_V):
        self.R_V = R_V
        self.R_V_set = True
        if self.a_b_set:
            self.Alam_AV = PhotometryInterpolator(self.x,
                                                  self.a + self.b / self.R_V,
                                                  kind = 'linear',
                                                  extrapolate = 'no',
                                                  positive=False)
            self.ready = True
        else:
            self.ready = False

    def compute_extinction(self, x, Av, recompute=False, Rv=None):
        """
        Compute the extinctions
        """
        if Rv is not None:
            self.set_R_V(Rv)
        if recompute is True:
            self.set_curve(x)
        wx = self.input2wavenumber(x)
        alam_av = self.Alam_AV(wx)
        if DEBUG:
            print('compute_extinction: interpolator x={}'.format(
                np.any(np.isfinite(self.Alam_AV.interpolator.x))))
            print('compute_extinction: interpolator slopes={}'.format(
                np.any(np.isfinite(self.Alam_AV.interpolator.slopes))))
            print('compute_extinction: interpolator ordinates={}'.format(
                np.any(np.isfinite(self.Alam_AV.interpolator.ordinates))))
            print('compute_extinction: interpolator slopes={}'.format(
                np.any(np.isfinite(self.Alam_AV.interpolator.slopes))))
            print('compute_extinction: np.any(np.isfinite(alam_av))={}'.format(
                np.any(np.isfinite(alam_av))))
        msg = ndarray_1darray(Av)
        if msg is None:
            fact = 10.**(-0.4 * self.Alam_AV(wx)[np.newaxis,:]*Av[:,np.newaxis])
        else:
            fact = 10. ** (-0.4 * self.Alam_AV(wx) * Av)
        if DEBUG:
            print('compute_extinction: np.any(np.isfinite(fact))={}'.format(
                np.any(np.isfinite(fact))))
        return fact

class CCMExtinction(ExtinctionLaw):
    """
    Implementation of the Cardelli, Clayton & Mathis (1989)
    """
    def __init__(self):
        super().__init__()
        self._xlims = np.array([0., 0.3, 1.1, 3.3, 5.9, 8.0, 10.0])
        self._a_extfunc = [self._undefined, self._a_ir_extinction,
                           self._a_opt_extinction,
                           self._a_uv_extinction, self._a_uvbump_extinction,
                           self._a_fuv_extinction, self._undefined]
        self._b_extfunc = [self._undefined, self._b_ir_extinction,
                           self._b_opt_extinction,
                           self._b_uv_extinction, self._b_uvbump_extinction,
                           self._b_fuv_extinction, self._undefined]

    def _undefined(self, x):
        return np.NaN * x

    def _a_ir_extinction(self, x):
        return 0.574 * x**1.61

    def _b_ir_extinction(self, x):
        return -0.527 * x**1.61

    def _a_opt_extinction(self, x):
        y = x - 1.82
        a_coeff = [+1.00000, +0.17699, -0.50447, -0.023427, +0.72085, +0.01979,
                   -0.77530, +0.32999]
        return np.polynomial.polynomial.polyval(y, a_coeff)

    def _b_opt_extinction(self, x):
        y = x - 1.82
        b_coeff = [0.00000, +1.41338, +2.28305, +1.07233, -5.38434, -0.62251,
                   +5.30260, -2.09002]
        return np.polynomial.polynomial.polyval(y, b_coeff)

    def _a_uv_extinction(self, x):
        return +1.752 - 0.316 * x - 0.104 / ((x - 4.67)**2 + 0.341)

    def _b_uv_extinction(self, x):
        return -3.090 + 1.825 * x + 1.206 / ((x - 4.62)**2 + 0.263)

    def _a_uvbump_extinction(self, x):
        return +1.752 - 0.316 * x - 0.104 / ((x - 4.67)**2 + 0.341)\
                 - 0.04473 * (x - 5.9)**2 - 0.009779 * (x - 5.9)**3

    def _b_uvbump_extinction(self, x):
       return -3.090 + 1.825 * x + 1.206 / ((x - 4.62)**2 + 0.263)\
                 + 0.2130 * (x - 5.9)**2 + 0.1207 * (x - 5.9)**3

    def _a_fuv_extinction(self, x):
        y = x - 8.0
        a_coeff = [-1.073, -0.628, +0.137, -0.070]
        return np.polynomial.polynomial.polyval(y, a_coeff)

    def _b_fuv_extinction(self, x):
        y = x - 8.0
        b_coeff = [13.670, +4.257, -0.420, +0.374]
        return np.polynomial.polynomial.polyval(y, b_coeff)


class ODonnellExtinction(CCMExtinction):
    """
    Implementation of the O'Donnell (1994) extinction law, which is a
    correction of the  Cardelli, Clayton & Mathis (1989) one in the optical
    domain.
    """
    def __init__(self):
        super().__init__()
        self._a_extfunc[2] = self._a_opt_extinction_od
        self._b_extfunc[2] = self._b_opt_extinction_od

    def _a_opt_extinction_od(self, x):
        y = x - 1.82
        a_coeff = [+1.000, +0.104, -0.609, +0.701, +1.137, -1.718, -0.827,
                   +1.647, -0.505]
        return np.polynomial.polynomial.polyval(y, a_coeff)

    def _b_opt_extinction_od(self, x):
        y = x - 1.82
        b_coeff = [0.000, +1.952, +2.908, -3.989, -7.985, +11.102, +5.491,
                   -10.805, +3.347]
        return np.polynomial.polynomial.polyval(y, b_coeff)

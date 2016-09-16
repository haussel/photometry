__author__ = 'haussel' 
"""
This modules provides useful functions for photometric computations.
"""
import numpy as np
from astropy import units as u
from astropy import constants as const


velc = const.c.to(u.m / u.s).value
nu_unit = u.Hz
lam_unit = u.m
fnu_unit = u.W / u.m**2 / u.Hz
flam_unit = u.W / u.m**3
nufnu_unit = u.W / u.m**2


def is_wavelength(unit):
    """
    Check whether the unit is compatible with a wavelength.
    :param unit: astropy.unit
    :return: True / False
    """
    try:
        unit.to(u.m)
        result = True
    except:
        result = False
    return result

def is_frequency(unit):
    """
    Check whether the unit is compatible with a frequency.
    :param unit: astropy.unit
    :return: True / False
    """
    try:
        unit.to(u.Hz)
        result = True
    except:
        result = False
    return result

def is_energy(unit):
    """
    Check whether a unit is an energy
    :param unit: astropy.unit
    :return: True / False
    """
    try:
        unit.to(u.J)
        result = True
    except:
        result = False
    return result

def is_flam(unit):
    """
    Check a unit is a spectral irradiance per unit wavelength
    :param unit: astropy.unit
    :return: True / False
    """
    try:
        unit.to(u.W/u.m**2/u.Angstrom)
        result = True
    except:
        result = False
    return result

def is_fnu(unit):
    """
    Check a unit is a spectral irradiance per unit frequency
    :param unit: astropy.unit
    :return: True / False
    """
    try:
        unit.to(u.Jy)
        result = True
    except:
        result = False
    return result

def is_flux(unit):
    """
    Check a unit is an irradiance
    :param unit: astropy.unit
    :return: True / False
    """
    try:
        unit.to(u.W/u.m**2)
        result = True
    except:
        result = False
    return result

def quantity_scalar(x):
    """
    Check that input is a scalar quantity
    :param x:
    :return:
    """
    msg = None
    if not isinstance(x, u.Quantity):
        msg = ' has to be a quantity'
        return msg
    if not x.isscalar:
        msg = ' has to be a scalar'
        return msg
    return msg

def ndarray_1darray(x, length=None, other=None):
    """
    Check that an input is a quantity 1d array with at least 2 elements
    :param x:
    :return: boolean
    """
    msg = None
    if not isinstance(x, np.ndarray):
        msg = " has to be an array"
        return msg
    if len(x.shape) > 1:
        msg = " has to be 1D"
        return msg
    if len(x) < 2:
        msg = " has to have 2 elements or more"
        return msg
    if length is not None:
        if len(x) != length:
            msg = " must have the same number of elements as " + other
            return msg
    return msg


def quantity_1darray(x, length=None, other=None):
    """
    Check that an input is a quantity 1d array with at least 2 elements
    :param x:
    :return: boolean
    """
    msg = None
    if not isinstance(x, u.Quantity):
        msg = " has to be an array of quantity"
        return msg
    if x.isscalar:
        msg = " has to be an array"
        return msg
    if len(x.shape) > 1:
        msg = " has to be 1D"
        return msg
    if len(x) < 2:
        msg = " has to have 2 elements or more"
        return msg
    if length is not None:
        if len(x) != length:
            msg = " must have the same number of elements as " + other
            return msg
    return msg

def quantity_2darray(x, length=None, other=None):
    """
    Check that an input is a quantity 2d array with at least 2 elements
    :param x:
    :return: boolean
    """
    msg = None
    if not isinstance(x, u.Quantity):
        msg = " has to be an array of quantity"
        return msg
    if x.isscalar:
        msg = " has to be an array"
        return msg
    if x.ndim > 2:
        msg = " has to be 1D or 2D"
        return msg
    if x.shape[0] < 2:
        msg = " has to have 2 elements or more"
        return msg
    if length is not None:
        if x.shape[-1] != length:
            msg = " must have the same number of" \
                  " elements along axis 1 as " + other
            return msg
    return msg



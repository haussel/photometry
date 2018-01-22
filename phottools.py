"""
This modules provides useful functions for photometric computations:
- is_wavelength(unit) : check whether unit is a wavelength unit
- is_frequency(unit)  : check whether unit is a frequency unit
- is_energy(unit)     : check whether unit is a energy unit
- is_flam(unit)       : check whether unit is a spectral irradiance per unit
                        wavelength
- is_fnu(unit)        : check whether unit is a spectral irradiance per unit
                        frequency
- is_flux(unit)       : check whether unit is an irradiance
- quantity_scalar(x)  : check whether x is a scalar
- ndarray_1darray(x)  : check whether x is a 1d array, test length if needed
- quantity_1darray(x) : check whether x is a 1d Quantity, test length if needed
- quantity_2darray(x) : check whether x is a 2d Quantity, test length if needed
"""

__author__ = 'haussel'

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

def is_lum(unit):
    """
    Check a unit is a luminosity
    :param unit: astropy.unit
    :return: True / False
    """
    try:
        unit.to(u.W)
        result = True
    except:
        result = False
    return result


def quantity_scalar(x):
    """
    Check that input is a scalar quantity
    :param x: variable to test
    :return: True / False
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
    Check that an input is a 1d array with at least 2 elements

    Parameters
    ----------
    x : any
        input ot be tested
    length: int
        if not None (default), test that x and length elements
    other: str
        name of variable to be added to message

    Returns
    -------
    if x checks OK, returns None, otherwize returns a string message
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
    Check that an input is a 1d Quantity with at least 2 elements

    Parameters
    ----------
    x : any
        input ot be tested
    length: int
        if not None (default), test that x and length elements
    other: str
        name of variable to be added to message

    Returns
    -------
    if x checks OK, returns None, otherwize returns a string message

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
    Check that an input is a 2d Quantity with at least 2 elements in its
    first dimension. Length allows to test the number of elements in the
    second dimension.

    Parameters
    ----------
    x : any
        input ot be tested
    length: int
        if not None (default), test that x and length elements in its last
        dimension.
    other: str
        name of variable to be added to message

    Returns
    -------
    if x checks OK, returns None, otherwize returns a string message

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

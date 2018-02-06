"""
This module provides useful functions and classes for photometric computations,

Package constants and units:
----------------------------
- velc : velocity of light in m/s
- nu_unit : frequency unit: Hz
- lam_unit : wavelength unit: m
- fnu_unit : spectral irradiance per unit frequency unit. SI W/m**2/Hz
- flam_unit : spectral irradiance per unit wavelength: SI  W/m**2/m
- nufnu_unit : spectral irradiance: W/m**2

Utility functions:
------------------
- is_wavelength(unit)    : check whether unit is a wavelength unit
- is_frequency(unit)     : check whether unit is a frequency unit
- is_energy(unit)        : check whether unit is a energy unit
- is_flam(unit)          : check whether unit is a spectral irradiance per unit
                           wavelength
- is_fnu(unit)           : check whether unit is a spectral irradiance per unit
                           frequency
- is_flux(unit)          : check whether unit is an irradiance
- quantity_scalar(x)     : check whether x is a scalar
- ndarray_1darray(x)     : check whether x is a 1d array, test length if needed
- quantity_1darray(x)    : check whether x is a 1d Quantity, test length if
                           needed
- quantity_2darray(x)    : check whether x is a 2d Quantity, test length if
                           needed
- read_photometry_file() : read a photometry ascii file
- write_photometry_file(): write a photometry ascii file

Classes:
--------
- PhotometryInterpolator()  : interface to the various interpolation classes
- NearestInterpolator()     : nearest neighbour interpolation
- LinearInterpolator()      : linear interpolation
- QuadraticInterpolator()   : quadratic interpolation
- LogLogLinearInterpolator(): linear interpolation in log space
- PhotometryHeader()        : header of photometry files
"""

__author__ = 'haussel'

import numpy as np
from astropy import units as u
from astropy import constants as const
import re
import os


# package constants and units.
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
    except u.UnitConversionError:
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
    except u.UnitConversionError:
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
    except u.UnitConversionError:
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
    except u.UnitConversionError:
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
    except u.UnitConversionError:
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
    except u.UnitConversionError:
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
    except u.UnitConversionError:
        result = False
    return result


def quantity_scalar(x):
    """
    Check that input is a scalar quantity
    :param x: variable to test
    :return: None or an error message
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


def ndarray_2darray(x, length=None, other=None):
    """
    Check that an input is a 2d ndarray with at least 2 elements in its
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
    if not hasattr(x, 'ndim'):
        msg = " has to be an array"
        return msg
    if x.ndim == 0:
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
    msg = ndarray_2darray(x, length=length, other=other)
    if not isinstance(x, u.Quantity):
        msg = " has to be an array of quantity"
    return msg


def read_photometry_file(filename):
    """
    Read a photometry ascii file

    Parameters
    ----------
    filename: str
        The name of the file. If the file is not found, the curve is
        searched for in the data/ directory of the installation of
        the photometry package. If more than one is found, an error
        is raised

    Returns
    -------
    tuple (values, header)

    """
    if not os.path.exists(filename):
        result = []
        basedir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'data/')
        for root, dirs, files in os.walk(basedir):
            if filename in files:
                result.append(os.path.join(root, filename))
        if len(result) == 1:
            fullpath = result[0]
        else:
            raise ValueError('could not locate {} filename, found {}'.format(
                filename, result))
    else:
        fullpath = filename
    values = np.genfromtxt(fullpath, comments='#',
                           dtype='float64', names=['x', 'y'])
    f = open(fullpath, 'r')
    line = f.readline()
    sheader = []
    while line[0] == '#':
        sheader.append(line)
        line = f.readline()
    f.close()
    return values, sheader


def write_photometry_file(photobject, xunit, filename=None, dirname=None,
                          overwrite=False, xfmt=':> 0.6f', yfmt=':>0.6f'):
    """
    Write a photometry object (Passband, Atmosphere) to a ascii file.

    Parameters
    ----------
    photobject: Passband or Atmosphere
        the object to be written.
    xunit: astropy.unit
        The unit to use to write the file.
    filename: str
        If given, the name of the file to write. If not given, uses the object
        header 'file' card
    dirname: str
        If given, the name of the directory where to write. If not given,
        uses the object default_dir() method
    overwrite: bool
        If True, will overwrite the existing passband file. Defaulted
        to False
    xfmt: str
        format for the x values
    yfmt: str
        format for the y values

    Returns
    -------
    str : the fullpath of the written file
    """
    if dirname is None:
        dirname = photobject.default_dir()
    if filename is None:
        filename = photobject.header['file']
    fullfilename = os.path.join(dirname, filename)
    if os.path.exists(fullfilename):
        if not overwrite:
            raise ValueError("The passband file {} already exists !"
                             " Aborting".format(fullfilename))
    # check xunit
    if photobject.is_lam:
        if is_wavelength(xunit):
            xvals = (photobject.x * photobject.x_si_unit).to(xunit).value
        else:
            raise ValueError("Invalid output unit {} for photobject in"
                             " wavelenght".format(xunit))
    elif photobject.is_nu:
        if is_frequency(xunit):
            xvals = (photobject.x * photobject.x_si_unit).to(xunit).value
        else:
            raise ValueError("Invalid output unit {} for photobject in"
                             " frequency".format(xunit))
    else:
        raise ValueError("Photobject is neither in freq or lam ")

    with open(fullfilename, 'w') as f:
        f.write("{}\n".format(photobject.header.
                              format_card('file', photobject.header['file'])))
        for key, value in photobject.header.items():
            if key is not 'xunit' and key is not 'file':
                f.write("{}\n".format(photobject.header.format_card(key,
                                                                    value)))
        f.write("# xunit: {}\n".format(xunit))
        outstr = "{" + xfmt + "}    {" + yfmt + "}\n"
        for i, xval in enumerate(xvals):
            f.write(outstr.format(xval, photobject.y[i]))

    return fullfilename


class PhotometryInterpolator:
    """
    Class for interpolation of passbands, spectra, atmospheres, etc...
    The need for a  specific class arises from the fact that
    scipy.interpolate raises an exception when out of bound values are
    requested. Here the user has the possibility to use extrapolation.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``.  This class returns a function whose call method uses
    interpolation to find the value of new points.

    Contrary to scipy.interpolate.interp1d, the coefficient for interpolation
    are only evaluated once, at object initialization.

    Parameters
    ----------
    x : (N, ) ndarray
        A 1-D array of real values, sorted in ascending order, and not
        containing duplicate values.
    y : (N, ) ndarray or (M, N) ndarray
        A 1-D array of real values. The length of `y` must be equal to the
        length of `x`
    kind: str optional
        Specifies the kind of interpolation as a string:
        'nearest' : nearest neighbor interpolation
        'linear' : linear interpolation
        'quadratic' : piecewize quadratic interpolation
        'log-log-linear' : linear interpolation in log space
        default value is 'log-log-linear'
    extrapolate: str
        can be either:
        - 'no' : no extrapolation. Extrapolated values will be set to NaN
        - 'zero' : Extrapolated values are set to zero. This is convenient
                   for passbands.
        - 'yes': extrapolated values are returned.
    positive: bool
        ensure the output is above 0 if True. All negative values are set to 0.
        Note that positive = True is not equivalent to extrapolate='zero',
        positive test everywhere, while extrapolate deals only with what
        happens outside.

    Attributes
    ----------
    interpolator: the appropriate interpolator object

    Methods
    -------

    __call__(x) : performs the interpolation at x

    """
    def __init__(self, x, y, kind, extrapolate='no', positive=True):
        if kind == 'nearest':
            self.interpolator = NearestInterpolator(x, y, positive=positive,
                                                    extrapolate=extrapolate)
        elif kind == 'linear':
            self.interpolator = LinearInterpolator(x, y, positive=positive,
                                                   extrapolate=extrapolate)
        elif kind == 'log-log-linear':
            # log-log is always positive by design.
            self.interpolator = \
                LogLogLinearInterpolator(x, y, extrapolate=extrapolate)
        elif kind == 'quadratic':
            self.interpolator = QuadraticInterpolator(x, y, positive=positive,
                                                      extrapolate=extrapolate)
        else:
            raise ValueError("Invalid interpolation scheme '{}'".format(kind))

    def __call__(self, x):
        return self.interpolator.__call__(x)

    def __str__(self):
        return self.interpolator.__str__()

    @staticmethod
    def available_interpolations():
        return ('nearest',
                'linear',
                'log-log-linear',
                'quadratic')

    @staticmethod
    def available_extrapolations():
        return 'no', 'zero', 'yes'


class NearestInterpolator:
    """
    Nearest neighbour interpolation

    Attributes:
    -----------
    extrapolate: str
        can be 'no' (default), 'yes' or 'zero'.
    positive: bool
        if yes, negative result values are set to zero
    x_low: float
        lowest bound of the input
    x_high: float
        highest bound of the input
    x_bounds: ndarray[N]
        midpoints between inputs
    y: ndarray [N] or [M, N]
        input values at x

    Methods:
    --------
    __call__(x) : performs the interpolation at x

    """
    def __init__(self, x, y, extrapolate='no', positive=True):
        """
        :param x: (N, ) ndarray of a abscissae
        :param y: (N, ) or (M, N) ndarray of ordinates
        :param extrapolate: 'yes', 'no', or 'zero'
        :param positive: bool
        """
        if extrapolate not in ['no', 'zero', 'yes']:
            raise ValueError("Invalid extrapolation scheme: '{}'".format(
                extrapolate))
        self.extrapolate = extrapolate
        self.positive = positive
        self.x_low = x[0]
        self.x_high = x[-1]
        self.x_bounds = (x[1:] + x[:-1]) / 2.
        self.x_bounds = np.append(self.x_bounds, [x[-1]])
        self.y = y

    def __call__(self, x):
        idx = np.searchsorted(self.x_bounds, x)
        idx = np.where(idx == len(self.x_bounds), len(self.x_bounds)-1, idx)
        if self.y.ndim == 2:
            result = self.y[:, idx]
        else:
            result = self.y[idx]

        if self.positive:
            result = np.where(result > 0, result, 0)

        if self.extrapolate is not 'yes':
            if self.extrapolate is 'zero':
                badval = 0.
            else:
                badval = np.NaN
            result = np.where((x >= self.x_low) & (x <= self.x_high), result,
                              badval)
        return result

    def __str__(self):
        output = "NearestInterpolator, extrapolate:{}, positive: {}".format(
            self.extrapolate, self.positive)
        output += "\nDefinition domain: {} to {}".format(self.x_low,
                                                         self.x_high)
        return output


class LinearInterpolator:
    """
    Linear Interpolation.

    Attributes:
    -----------
    extrapolate: str
        can be 'no' (default), 'yes' or 'zero'.
    positive: bool
        If True all negative values a set to zero.
    x: ndarray[n]
        abscissae of the input
    slopes: ndarray[n] or [M, n]
        slopes of the segments
    ordinates : ndarray[n] or [M, n]
        ordinates of the segments

     Method
     ------
     __call__(x) perform interpolation at x

     __str__

    """
    def __init__(self, x, y, extrapolate='no', positive=True):
        """
        :param x: (N, ) ndarray of a abscissae
        :param y: (N, ) or (M, N) ndarray of ordinates
        :param extrapolate: 'yes', 'no', or 'zero'
        :param positive: bool
       """
        if extrapolate not in ['no', 'zero', 'yes']:
            raise ValueError("Invalid extrapolation scheme: '{}'".format(
                extrapolate))
        self.extrapolate = extrapolate
        self.positive = positive
        self.x = x
        if y.ndim == 2:
            self.slopes = (y[:, 1:] - y[:, :-1]) / (x[1:] - x[:-1])
            self.ordinates = y[:, :-1] - self.slopes * x[:-1]
        else:
            self.slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
            self.ordinates = y[:-1] - self.slopes * x[:-1]

    def __call__(self, x):
        idx = np.digitize(x, self.x)-1
        idx = idx.clip(0, len(self.x)-2)
        if self.slopes.ndim == 2:
            result = self.slopes[:, idx] * x + self.ordinates[:, idx]
        else:
            result = self.slopes[idx] * x + self.ordinates[idx]

        if self.positive:
            result = np.where(result > 0, result, 0.)

        if self.extrapolate is not 'yes':
            if self.extrapolate is 'zero':
                badval = 0.
            else:
                badval = np.NaN
            result = np.where((x >= self.x[0]) & (x <= self.x[-1]), result,
                              badval)
        return result

    def __str__(self):
        output = "LinearInterpolator, extrapolate:{}, positive: {}".format(
            self.extrapolate, self.positive)
        output += "\nDefinition domain: {} to {}".format(self.x[0],
                                                         self.x[-1])
        return output


class LogLogLinearInterpolator(LinearInterpolator):
    """
    Linear interpolator in Log space. Uses Line
    """
    def __init__(self, x, y, extrapolate='No'):
        super().__init__(np.log10(x), np.log10(y),
                         positive=False, extrapolate=extrapolate)

    def __call__(self, x):
        return 10.**(super().__call__(np.log10(x)))


class QuadraticInterpolator:
    """
    Quadratic interpolation:
    - computes the 2nd degree polynom coefficients at each Xi that pass
      through (Xi-1,Yi-1),(Xi,Yi),(Xi+1,Yi+1).
    - for any point between Xi and Xi+1, the output is the weighted
      sum of the polynoms Pi and Pi+1, the weights beeing the distance
      to the Xi and Xi+1

    Attributes
    ----------
    extrapolate: str
        can be 'no' (default), 'yes' or 'zero'.
    positive: bool
        If True all negative values a set to zero.
    x: ndarray[n]
        abscissae of the input
    n: len(x)
    a: degree 0 polynom coeffs
    b: degree 1 polynom coeffs
    c: degree 2 polynom coeffs

    Methods
    -------
     __call__(x) perform interpolation at x

     __str__
    """
    def __init__(self, x, y, extrapolate='no', positive=True):
        if extrapolate not in ['no', 'zero', 'yes']:
            raise ValueError("Invalid extrapolation scheme: '{}'".format(
                extrapolate))
        self.extrapolate = extrapolate
        self.positive = positive
        self.x = x
        self.n = len(x)
        xc = np.arange(self.n, dtype='int64')
        xc[0] = 1
        xc[self.n - 1] = self.n - 2
        xm = xc - 1
        xp = xc + 1
        discrim = (x[xm] - x[xc]) * (x[xm] - x[xp]) * (x[xp] - x[xc])
        zz, = np.where(discrim == 0)
        if len(zz) > 0:
            raise ValueError("quadratic interpolation: discrimiment has "
                             "0 values \ndiscrim = {}".format(discrim))
        if y.ndim == 2:
            self.b = ((y[:, xm] - y[:, xc]) *
                      (x[xm] * x[xm] - x[xp] * x[xp]) -
                      ((x[xm] * x[xm] - x[xc] * x[xc]) *
                       (y[:, xm] - y[:, xp]))) / discrim
            self.c = ((y[:, xm] - y[:, xp]) * (x[xm] - x[xc]) -
                      (x[xm] - x[xp]) * (y[:, xm] - y[:, xc])) / discrim
            self.a = y[:, xc] - self.b * x[xc] - self.c * x[xc] * x[xc]
        else:
            self.b = ((y[xm] - y[xc]) * (x[xm] * x[xm] - x[xp] * x[xp]) -
                      ((x[xm] * x[xm] - x[xc] * x[xc]) *
                       (y[xm] - y[xp]))) / discrim
            self.c = ((y[xm] - y[xp]) * (x[xm] - x[xc]) -
                      (x[xm] - x[xp]) * (y[xm] - y[xc])) / discrim
            self.a = y[xc] - self.b * x[xc] - self.c * x[xc] * x[xc]

    def __call__(self, x):
        nexti = np.digitize(x, self.x)
        nexti[nexti < 1] = 1
        nexti[nexti >= self.n] = self.n-1
        previ = nexti - 1
        distprev = x - self.x[previ]
        distnext = self.x[nexti] - x
        dist = self.x[nexti] - self.x[previ]
        # computes the fit
        if self.a.ndim == 2:
            res = (distnext * (self.a[:, previ] +
                               (self.b[:, previ] + x * self.c[:, previ]) * x) +
                   distprev * (self.a[:, nexti] +
                               (self.b[:, nexti] + x * self.c[:, nexti]) *
                               x)) / dist
        else:
            res = (distnext * (self.a[previ] + (self.b[previ] +
                                                x * self.c[previ]) * x) +
                   distprev * (self.a[nexti] + (self.b[nexti] +
                                                x * self.c[nexti]) * x)) / dist

        if self.positive:
            res = np.where(res > 0, res, 0.)

        if self.extrapolate is not 'yes':
            if self.extrapolate is 'zero':
                badval = 0.
            else:
                badval = np.NaN
            res = np.where((x >= self.x[0]) & (x <= self.x[-1]),
                           res, badval)
        return res

    def __str__(self):
        output = "QuadraticInterpolator, extrapolate:{}, positive: {}".format(
            self.extrapolate, self.positive)
        output += "\nDefinition domain: {} to {}".format(self.x[0],
                                                         self.x[-1])
        return output


class PhotometryHeader:
    """
    A Class to handle the headers of photometry files files. Behaves like a
    dict.

    Attributes:
    -----------
    content : dict
        key, values of the header cards
    re : compiled regular expression
        matching pattern to decode header


    The class provides methods to format to and read from header lines.
    - Formatting is handled by overloading the __str__ method, so that print(hd)
      will correctly print header as it would appear in a file.
    - Reading from a line is done by the import_line() method
    - Dictionaries can be imported by the import_dict() method
    - Card and values can be added by the add_card_value() method.
    """

    def __init__(self):
        """
        Creates an empty dictionary and pre compile matching strings
        """
        self.content = dict()
        self.key_card = re.compile('^#\s+(.+):\s+(\S+.*)$')
        self.split_cr = re.compile(r'\n')

    def __getitem__(self, item):
        return self.content[item]

    def __setitem__(self, card, value):
        # Some cards should have a unique value
        if card in ['xref', 'xtype', 'ytype', 'file', 'instrument', 'filter',
                    'system', 'atmosphere', 'airmass']:
            self.edit_card_value(card, value)
        else:
            self.add_card_value(card, value)

    def __contains__(self, item):
        return self.content.items()

    def items(self):
        return self.content.items()

    def add_card_value(self, card, value):
        """
        Add a the pair card, value to the header. If card is already present,
        add to the existing value by introducing a carriage return before.

        Parameters
        ----------
        card: str
          name of the card
        value: any
          value is formatted to string.
        """
        if not isinstance(card, str):
            wcard = str(card)
        else:
            wcard = card
        if wcard in self.content:
            self.content[wcard] = self.content[wcard] + '\n{}'.format(value)
        else:
            self.content[wcard] = '{}'.format(value)

    def edit_card_value(self, card, value):
        """
        Edit the value of keyword value pair in the header.

        Parameters
        ----------
        card: str
          name of the card
        value: any
          value is formatted to string.
        """
        if not isinstance(card, str):
            wcard = str(card)
        else:
            wcard = card
        self.content[wcard] = '{}'.format(value)

    def import_line(self, line):
        """
        decript a file header line and add the corresponding key, value to the
        header dictionnary.

        Parameters
        ----------
        line : str
        """
        mcard = self.key_card.match(line)
        if mcard:
            card = mcard.group(1)
            value = mcard.group(2)
            if card == 'xref (xunit)':
                card = 'xref'
            self.add_card_value(card, value)
        else:
            raise ValueError('The following  header line was not parsed:\n{}'.
                             format(line))

    def import_dict(self, header):
        """
        Add a dictionary to the header

        Parameters
        ----------
        header: dict
        """
        for k, v in header.items():
            self.add_card_value(k, v)
        return None

    def format_card(self, key, value):
        """
        Format the card, value to print header.
        :param key: card name
        :param value: value
        :return: sting "# card: value". If value is multilines return
        "# card: first bit of value"
        ...
        "# card: last bit of value"
        """
        result = ''
        bits = self.split_cr.split(value)
        for bit in bits:
            if len(result) == 0:
                result = '# {}: {}'.format(key, bit)
            else:
                result += '\n# {}: {}'.format(key, bit)
        return result

    def __str__(self):
        """
        Overloading of string representation
        :return: str
        """
        if len(self.content) == 0:
            result = 'Header: None'
        else:
            result = '############### Header #################'
            for k, v in self.content.items():
                result += '\n{}'.format(self.format_card(k, v))
            result += '\n############### End of Header #################'
        return result

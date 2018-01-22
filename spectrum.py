__author__ = 'haussel'
"""
This module provides the following classes:
- BasicSpectrum: a class to represent one or more spectra
- SpectrumInterpolator: a class to perform various interpolation
- GalaxySpectrum : a class to represent galaxy spectra.
"""
import numpy as np
import os
import inspect
from astropy import units as u
from astropy.table import Table
from .phottools import  is_frequency, is_wavelength, is_flam, \
    is_fnu, is_flux, quantity_scalar, quantity_1darray, quantity_2darray,\
    ndarray_1darray,velc , nu_unit, lam_unit, flam_unit, fnu_unit


class BasicSpectrum:
    """
    A class to represent spectra.


    Attributes
    ----------
    org_x_type : 'nu' or 'lam'
        original type of x input
    org_x_unit : astropy.units
        original unit of x input
    org_y_type : 'fnu' or 'flam'
        original type of y input
    org_y_unit : astropy.units
        original units of input
    x : astropy.units.Quantity 1D
        array of wavelength or frequencies in SI units
    x_si_unit: astropy.units
        unit of x
    is_lam: bool
        True if x is an array of wavelengths
    is_nu: bool
        True if x is an array of frequencies
    nb: int
        Number of spectra
    y: astropy.units.Quantity 1D [N] or 2D [nb, N] where N is the number of
       x points
       Spectra
    y_si_unit: astropy.units
        unit of y
    is_flam: bool
        True if y is a spectral irradiance per unit wavelength
    is_fnu: bool
        True if y  is a spectral irradiance per unit frequency
    yscale: float or ndarray
        scale factor applied to the spectrum
    xshift: float
        shift applied to the x array
    interpolation_method: str
        method to interpolate the spectrum. Can be:
            - nearest : nearest neighbor interpolation
            - linear : linear interpolation
            - quadratic : quadratic interpolation
            - log-log-linear : linear interpolation in log space
    can_extrapolate: bool
        if True, interpolation can extrapolate outside spectrum definition
        without raising an exception
    interpolation_set: bool
        True if the spectrum interpolator has been initialized
    interpolate: SpectrumInterpolator
        The object performing the interpolation


    Methods
    -------

    - set_interpolation(interpolation_method, extrapolate=False)
        Set the interpolation method for the spectrum
    - scale(factor)
        Multiply the spectra by factor(s).
    - unscale()
    - shift(factor)
        Multiply the x by factor
    - unshift()
        Remove any shift
    - nu(unit=None)
        Returns x as an array of frequencies
    - lam(unit=None)
        Returns x as an array of wavelengths
    - fnu(unit=None, ispec=None)
        Returns spectra as spectral irradiance per unit frequency, with the
        possibility to choose the unit and to only output one selected spectrum
    - flam(unit=None, ispec=None)
        Returns the spectrum in spectral irradiance per unit wavelength,
        with the possibility to choose the unit and to only output one
        selected spectrum
    - in_nu()
        Set the spectrum to 'nu' type.
    - in_lam()
        Set the spectrum to 'lam' type
    - in_flam()
        Set the spectrum to 'flam' type
    - in_fnu()
        Set the spectrum to 'fnu' type
    - fnu_nu(nu)
        Interpolate the spectrum in the form of Fnu at the requested
        frequencies. The spectrum is converted to 'nu', 'fnu' type
    - flam_lam(lam)
        Interpolate the spectrum in the form of Flam at the requested
        wavelengths. The spectrum is converted to 'lam', 'flam' type
    """
    def __init__(self, data=None, name_x=None, names_y=None,
                 x=None, x_type=None, x_unit=None,
                 y=None, y_type=None, y_unit=None,
                 interpolation_method='log-log-linear',
                 extrapolate=False):
        """
        Initialization. Valid calls are:

        - initialization from a Table:
        >>> import photometry as pt
        >>> spec = pt.BasicSpectrum(data=mydata)
        or specifying which columns to use
        >>> spec = pt.BasicSpectrum(data=mydata, name_x='colname',
                                    names_y=['colname1',...,'colnameN'])

        - initialization from quantities:
        >>> spec = pt.BasicSprectrum(x = my_x, y=my_y)
        - initialization from arrays
        >>> spec = pt.BasicSprectrum(x = my_x, x_type='my_x_type',
                                     x_unit=my_x_unit, y=my_y,
                                     y_type='my_y_type', y_unit=my_y_unit)

        Parameters
        ----------
        data: astropy.table.Table or numpy.ndarray
            Contains the x and y values
        name_x : str
            name of data column for the x values. If not provided, tries with
            the first column
        names_y: str or list of str
            name(s) of the columns containing the y values. If not provided,
            tries with all the remaining columns
        x : numpy.ndarray or astropy.Quantity
            values of x
        x_type : str 'lam' or 'nu'
            type of x
        x_unit : astropy.units or str
            unit of x
        y : numpy.ndarray or astropy.Quantity
            values of y
        y_type : str 'flam' or 'fnu'
            type of y
        y_unit : astropy.units or str
            unit of y
        interpolation_method: str
            interpolation method, can be 'nearest', 'linear', 'quadratic',
            'log-log-linear'. Defaulted to 'log-log-linear'
        extrapolate: bool
            If True, spectra can be interpolated outside of the x array
            bounds without raising an exception. Defaulted to False.


        """

        self.org_x_type = None
        self.org_x_unit = None
        self.org_y_type = None
        self.org_y_unit = None
        self.x = None
        self.x_si_unit = None
        self.is_lam = None
        self.is_nu = None
        self.nb = None
        self.y = None
        self.y_si_unit = None
        self.is_flam = None
        self.is_fnu = None
        self.yfactor = 1.0
        self.xshift = 1.0
        self.interpolation_method = None
        self.can_extrapolate = False
        self.interpolation_set = False
        self.interpolate = None

        if data is not None:
            if (x is not None) or (y is not None):
                raise ValueError('Cannot supply both `data` and `x` '
                                 'or `y` values')
            if isinstance(data, Table):
                self._init_from_table(data, colname_x = name_x,
                                      colnames_y = names_y)
            elif isinstance(data, np.ndarray):
                self._init_from_ndarray(data, name_x = name_x, names_y =
                names_y)
            else:
                raise ValueError('`data` has to be a table or an ndarray')
        else:
            if x is not None:
                if y is None:
                    raise ValueError('Both `x` and `y` values must '
                                     'be provided')
                elif isinstance(x, u.quantity.Quantity):
                    if isinstance(y, u.quantity.Quantity):
                        self._init_from_quantities(x, y)
                    else:
                        raise ValueError('Both `x` and `y` must be quantities')
                else:
                    if (x_type is not None) and (x_unit is not None) and \
                       (y_type is not None) and (y_unit is not None):
                        self._init_from_fullinfo(x, x_type, x_unit,
                                                 y, y_type, y_unit)
        if self.x is not None and self.y is not None:
            self.interpolation_set = self.set_interpolation(
                interpolation_method, extrapolate=extrapolate)
            self.yfactor = np.ones(self.nb)
            self.xshift = 1.0

    def _init_from_table(self, data, colname_x=None, colnames_y=None):
        """
        Initialize from a table.

        Parameters
        ----------
        data: astropy.table.Table
            table containing the data
        colname_x: str
            name of the column containing the frequencies or wavelengths.
            If not given, try with the first column
        colnames_y: str
            name(s) of the column(s) containing the fluxes.
            If not given, try with the second column
        """
        if not isinstance(data, Table):
            raise ValueError('`data` is not a Table')

        colnames = data.colnames
        if len(colnames) < 2:
            raise ValueError("data must have at least two columns")
        if colname_x is not None:
            if colname_x in colnames:
                if hasattr(data[colname_x], 'unit'):
                    x = data[colname_x].data * data[colname_x].unit
                else:
                    raise ValueError("Table has no unit for column {}".
                                     format(colname_x))
            else:
                raise ValueError("Column `{}` not found in Table".
                                 format(colname_x))
        else:
            print("Warning: Unspecified column for x, "
                  "will try to use the first: {}".format(colnames[0]))
            if hasattr(data[colnames[0]], 'unit'):
                x = data[colnames[0]].data * data[colnames[0]].unit
            else:
                raise ValueError("Table has no unit for column {}".
                                 format(colnames[0]))

        if colnames_y is not None:
            foundone = False
            foundunit = None
            for colname_y in colnames_y:
                if colname_y in colnames:
                    if hasattr(data[colname_y], 'unit'):
                        if foundone:
                            if data[colname_y].unit == foundunit:
                                y = np.vstack((y, data[colname_y].data))
                                y.unit = foundunit
                            else:
                                raise ValueError("column {} has not the same "
                                                 "unit ({}) as the others "
                                                 "cols ({}) ".
                                                 format(colname_y,
                                                        data[colname_y].unit,
                                                        foundunit))
                        else:
                            foundone = True
                            foundunit = data[colname_y].unit
                            y = data[colname_y].data
                    else:
                        raise ValueError("Table has no unit for column {}".
                                         format(colname_y))
                else:
                    raise ValueError("Column `{}` not found in Table".
                                     format(colname_y))
        else:
            print("Warning: Unspecified column for y, "
                  "will try to use the rest: {}".
                  format(colnames[1:]))
            foundone = False
            foundunit = None
            for colname_y in colnames[1:]:
                if hasattr(data[colname_y], 'unit'):
                    if foundone:
                        if data[colname_y].unit == foundunit:
                            y = np.vstack((y, data[colname_y].data))
#                            y.unit = foundunit
                        else:
                            raise ValueError("column {} has not the same unit "
                                             "({}) as the others cols ({}) ".
                                             format(colname_y,
                                                    data[colname_y].unit,
                                                    foundunit))
                    else:
                        foundone = True
                        foundunit = data[colname_y].unit
                        y = data[colname_y].data
                else:
                    raise ValueError("Table has no unit for column {}".
                                     format(colname_y))
#                print("y.shape={}".format(y.shape))
#                print("y.unit={}".format(y.unit))
            y = y * foundunit
        print('y.shape = {}'.format(y.shape))
        self._init_from_quantities(x,y)
        return None


    def _init_from_ndarray(self, data, colname_x=None,
                           colnames_y=None, interpolation_method=None):
        """
        Init from an ndarray. Not yet implemented

        """
        if not isinstance(data, np.ndarray):
            raise ValueError('`data` is not a ndarray')
        raise NotImplementedError('initialization from ndarray not yet '
                                  'implemented')


    def _init_from_quantities(self, x, y, interpolation_method=None):
        """
        initialize the spectrum from two quantities

        Parameters
        x: astropy.unit.quantity
            array of wavelengths or frequencies
        y: astropy.unit.quantity
            array of spectral irradiances, fnu or flam

        """

        msg = quantity_1darray(x)
        if msg is not None:
            raise ValueError("`x`" + msg)
        msg = quantity_2darray(y, length= len(x), other='x')
        if msg is not None:
            raise ValueError("`y`" + msg)
        if is_frequency(x.unit):
            self.is_lam = False
            self.is_nu = True
            self.x_si_unit = nu_unit
            self.org_x_type = 'nu'
            self.org_x_unit = x.unit
        elif is_wavelength(x.unit):
            self.x_si_unit = lam_unit
            self.is_lam = True
            self.is_nu = False
            self.org_x_type = 'lam'
            self.org_x_unit = x.unit
        else:
            raise ValueError('invalid unit {} for input quantity `x`'.
                             format(x.unit))
        xv = x.to(self.x_si_unit).value
        idx = np.argsort(xv)

        if y.ndim == 1:
            self.nb = 1
        elif y.ndim == 2:
            self.nb = y.shape[0]
        else:
            raise ValueError('`y` has to be a 1 or 2 dimension array')
        if is_fnu(y.unit):
            self.is_fnu = True
            self.is_flam = False
            self.y_si_unit = fnu_unit
            self.org_y_type = 'fnu'
            self.org_y_unit = y.unit
        elif is_flam(y.unit):
            self.y_si_unit = flam_unit
            self.is_flam = True
            self.is_fnu = False
            self.org_y_type = 'flam'
            self.org_y_unit = y.unit
        elif is_flux(y.unit):
            if self.org_x_type == 'lam':
                self.org_y_type = 'lamflam'
                self.org_y_unit = y.unit
                self.y_si_unit = flam_unit
                self.is_flam = True
                self.is_fnu = False
                y = y / x
            else:
                self.org_y_type = 'nufnu'
                self.org_y_unit = y.unit
                self.y_si_unit = fnu_unit
                self.is_flam = False
                self.is_fnu = True
                y = y / x
        else:
            raise ValueError('invalid unit {} for input quantity `y`'.
                             format(y.unit))

        yv = y.to(self.y_si_unit).value
        self.x = xv[idx]
        if self.nb > 1:
            self.y = yv[:,idx]
        else:
            self.y = yv[idx]
        return None

    def _init_from_fullinfo(self, x, x_type, x_unit, y, y_type, y_unit):
        """
        Initialize from full information

        Parameters
        ----------
        x: astropy.units.Quantity or numpy.ndarray
            x values
        x_type: str 'lam' or 'nu'
            type of x values
        x_unit: str or astropy.units
            unit of x
        y: astropy.units.Quantity or numpy.ndarray
            y values
        y_type: str 'flam' or 'fnu'
            type of y values
        y_unit: str or astropy.units
            unit of y

        Returns
        -------
        None
        """
        if not isinstance(x_unit, u.Unit):
            try:
                xunit = u.Unit(x_unit)
            except:
                raise ValueError('invalid `x_unit` : {}'.format(x_unit))
        else:
            xunit = x_unit

        if isinstance(x_type, str):
            if x_type == 'lam':
                if not is_wavelength(xunit):
                    raise ValueError('incompatible `x_type`:{} and '
                                     '`x_unit`:{}'.format(x_type, x_unit))
            elif x_type == 'nu':
                if not is_frequency(xunit):
                    raise ValueError('incompatible `x_type`:{} and '
                                     '`x_unit`:{}'.format(x_type, x_unit))
            else:
                raise ValueError('invalid `x_type`: {}'.format(x_type))
        else:
            raise ValueError('Invalid type for `x_typet`: {}'.
                             format(type(x_type)))

        if isinstance(x, u.Quantity):
            if x.unit is not xunit:
                raise ValueError('Quantity unit `x` ({}) and `x_unit` ({}) '
                                 'are incompatible'.format(x.unit, xunit))
            else:
                xv = x
        else:
            xv = x * xunit

        if not isinstance(y_unit, u.Unit):
            try:
                yunit = u.Unit(y_unit)
            except:
                raise ValueError('invalid `y_unit` : {}'.format(y_unit))
        else:
            yunit = y_unit

        if isinstance(y_type, str):
            if y_type == 'flam':
                if not is_flam(yunit):
                    raise ValueError('incompatible `y_type`:{} and '
                                     '`y_unit`:{}'.format(y_type, y_unit))
            elif y_type == 'fnu':
                if not is_fnu(yunit):
                    raise ValueError('incompatible `y_type`:{} and '
                                     '`y_unit`:{}'.format(y_type, y_unit))
            elif y_type == 'flux':
                if not is_flux(yunit):
                    raise ValueError('incompatible `y_type`:{} and '
                                     '`y_unit`:{}'.format(y_type, y_unit))
            else:
                raise ValueError('invalid `y_type`: {}'.format(y_type))
        else:
            raise ValueError('Invalid type for `y_unit`: {}'.
                             format(type(y_unit)))

        if isinstance(y, u.Quantity):
            if y.unit is not yunit:
                raise ValueError('Quantity unit `y` ({}) and `y_unit` ({}) '
                                 'are incompatible'.format(y.unit, yunit))
            else:
                yv = y
        else:
            yv = y * yunit
        self._init_from_quantities(xv, yv)
        return None

    def set_interpolation(self, interpolation_method, extrapolate=False):
        """
        Set the interpolation method, and initialize it.

        Parameters
        ----------
        interpolation_method: str
            interpolation method, can be:
                - 'nearest': nearest neighbor
                - 'linear' : linear interpolation
                - 'quadratic' : quadratic interpolation
                - 'log-log-linear' : linear in log space
        extrapolate: bool
            if True, the spectrum can be interpolated outside of its x support.
            Defaulted to false

        Returns
        -------
        bool : True if successul, False if not

        """
        try:
            self.interpolate = SpectrumInterpolator(self.x, self.y,
                                                    kind=interpolation_method,
                                                    extrapolate=extrapolate)
            result = True
        except:
            print("Problem setting interpolation with {}".
                  format(interpolation_method))
            result = False
        if result == True:
            self.interpolation_method = interpolation_method
            self.interpolation_set = True
        return result

    def scale(self, factor):
        """
        Scale the spectrum in the sense y *= factor

        Parameter
        ---------
        factor: float or numpy.array of nb spectra values
          the scaling factor(s)

        Returns
        -------
        None
        """
        # Ensure factor has no unit
        if isinstance(factor, u.Quantity):
            wfactor = factor.value
        else:
            wfactor = factor
        msg = ndarray_1darray(factor, length=self.nb)
        if msg is None:
            # factor is a 1D array
            self.y = self.y * wfactor[:, np.newaxis]
        else:

            self.y *= wfactor
        self.yfactor *= factor
        return None

    def unscale(self):
        """
        Remove any scaling
        """
        factor = 1./self.yfactor
        self.scale(factor)

    def shift(self, factor):
        """
        Shift the spectrum in the sense x *= factor

        Parameter
        ---------
        factor: float
            value by which x are multiplied

        Returns
        -------
        None
        """
        # ensure factor has no unit.
        if isinstance(factor, u.Quantity):
            wfactor = factor.value
        else:
            wfactor = factor
        self.x *= wfactor
        self.xshift *= wfactor
        return None

    def unshift(self):
        """
        Remove the shift in x
        """
        factor = 1./self.xshift
        self.shift(factor)


    def nu(self, unit=None):
        """
        Returns the frequency array of the spectrum in Hz or in requested unit

        Parameters
        ----------
        unit: astropy.unit
            the output unit. If None, returns frequencies in Hz

        Returns
        -------
        astropy.units.Quantity
            array of frequencies
        """
        if self.is_lam:
            if unit is None:
                return velc / self.x
            else:
                return (velc / self.x * nu_unit).to(unit)
        else:
            if unit is None:
                return self.x
            else:
                return (self.x * nu_unit).to(unit)

    def lam(self, unit = None):
        """
        Returns the wavelength array of the spectrum in m or in requested unit

        Parameters
        ----------
        unit: astropy.unit
            the output unit. If None, returns wavelengths in m

        Returns
        -------
        astropy.units.Quantity
            array of wavelengths

        """
        if self.is_lam:
            if unit is None:
                return self.x
            else:
                return (self.x * lam_unit).to(unit)
        else:
            if unit is None:
                return velc / self.x
            else:
                return (velc / self.x * lam_unit).to(unit)

    def fnu(self, unit=None, ispec=None):
        """
        returns the spectrum in spectral irradiance per unit frequency

        if the unit is not given, the spectrum is returned in fnu_unit (i.e
        in W/m**2/Hz) but without unit (ndarray)

        Parameters:
        -----------
        unit: astropy.units
          The output unit of the spectrum
        ispec: int
          If given, returns only the spectrum of indice ispec

        Returns
        -------
        output (, N) ndarray or astropy.units.quantity
        """
        if self.is_fnu:
            result = self.y
        else:
            if self.is_nu:
                result = self.y * velc / self.x / self.x
            else:
                result = self.y * self.x * self.x / velc
        if unit is not None:
            result = (result * fnu_unit).to(unit)
        if ispec is not None:
            if self.nb > 1:
                result = result[ispec, :]
        return result

    def flam(self, unit=None, ispec=None):
        """
        returns the spectrum in spectral irradiance per unit wavelength

        if the unit is not given, the spectrum is returned in flam_unit (i.e
        in W/m**2/m) but without unit (ndarray)

         Parameters:
        -----------
        unit: astropy.units
          The output unit of the spectrum
        ispec: int
          If given, returns only the spectrum of indice ispec

        Returns
        -------
        output (, N) ndarray or astropy.units.quantity

        """
        if self.is_flam:
            result = self.y
        else:
            if self.is_lam:
                result = self.y * velc / self.x / self.x
            else:
                result = self.y * self.x * self.x / velc
        if unit is not None:
            result = (result * flam_unit).to(unit)
        if ispec is not None:
            if self.nb > 1:
                result = result[ispec, :]
        return result

    def in_nu(self):
        """
        Set the spectrum to 'nu' type
        """
        if self.is_lam:
            self.x = velc / self.x[::-1]
            if self.nb > 1:
                self.y = self.y[:,::-1]
            else:
                self.y = self.y[::-1]
            self.is_lam = False
            self.is_nu = True
            self.x_si_unit = nu_unit
            self.xshift = 1./self.xshift
            self.interpolation_set = False
        return None

    def in_lam(self):
        """
        Set the spectrum to 'lam' type
        """
        if self.is_nu:
            self.x = velc / self.x[::-1]
            if self.nb > 1:
                self.y = self.y[:,::-1]
            else:
                self.y = self.y[::-1]
            self.is_lam = True
            self.is_nu = False
            self.x_si_unit = lam_unit
            self.interpolation_set = False
            self.xshift = 1./self.xshift
        return None

    def in_flam(self):
        """
        Set the spectrum to 'flam' type
        """
        if self.is_fnu:
            if self.is_nu:
                self.y = self.y * self.x / (velc / self.x)
            else:
                self.y = self.y * (velc / self.x) / self.x
            self.y_si_unit = flam_unit
            self.is_flam = True
            self.is_fnu = False
            self.interpolation_set = False
        return None

    def in_fnu(self):
        """
        Set the spectrum to 'fnu' type
        """
        if self.is_flam:
            if self.is_nu:
                self.y = self.y * (velc / self.x) / self.x
            else:
                self.y = self.y * self.x / (velc / self.x)
            self.y_si_unit = fnu_unit
            self.is_fnu = True
            self.is_flam = False
            self.interpolation_set = False
        return None

    def fnu_nu(self, nu):
        """
        Interpolate the spectrum in the form of Fnu at the requested
        frequencies. The spectrum is converted to 'nu', 'fnu' type

        Parameters
        ----------
        nu: astropy.units.Quantity or numpy array
            if nu is a numpy array, it is assumed to be in Hz

        Returns
        -------
        Interpolated fnu at the required frequencies.
        """
        self.in_nu()
        self.in_fnu()
        if not self.interpolation_set:
            self.set_interpolation(self.interpolation_method)
        if isinstance(nu, u.Quantity):
            result = self.interpolate((nu.to(nu_unit)).value) * fnu_unit
        else:
            result = self.interpolate(nu)
        return result

    def flam_lam(self, lam):
        """
        Interpolate the spectrum in the form of Flam at the requested
        wavelengths. The spectrum is converted to 'lam', 'flam' type

        Parameters
        ----------
        lam: astropy.units.Quantity or numpy array
            if lam is a numpy array, it is assumed to be in m

        Returns
        -------
        Interpolated flam at the required wavelengths.

        """
        self.in_lam()
        self.in_flam()
        if not self.interpolation_set:
            self.set_interpolation(self.interpolation_method)
        if isinstance(lam, u.Quantity):
            result = self.interpolate(lam.to(lam_unit).value) * flam_unit
        else:
            result = self.interpolate(lam)
        return result

    def __str__(self):
        result = "spectrum:"
        result = result + "\n  nb spectra   : {}".format(self.nb)
        result = result + "\n  nb datapoints: {}".format(len(self.x))
        if self.is_lam:
            result = result + "\n  xtype        : lam"
        elif self.is_nu:
            result = result + "\n  xtype        : nu"
        else:
            result = result + "\n  xtype        : unknown. That's not good !"
        result = result + "\n  coverage     : {:0.3e} to {:0.3e}".\
            format(self.x[0], self.x[-1]*self.x_si_unit)
        if self.is_flam:
            result = result + "\n  ytype        : flam"
        elif self.is_fnu:
            result = result + "\n  ytype        : fnu"
        else:
            result = result + "\n  ytype        : unknown. That's not good !"
        result = result + "\n  interpolation: {}".\
            format(self.interpolation_method)
        result = result + "\n  ready for use: {}".\
            format(self.interpolation_set)
        return result

class SpectrumInterpolator:
    """
    Interpolate a spectrum. The need for a specific class arises from the fact
    that scipy.interpolate raises an exception when out of bound values are
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
    y : (N, M) ndarray
        A 1-D array of real values. The length of `y` must be equal to the
        length of `x`
    kind: str optional
        Specifies the kind of interpolation as a string:
        'nearest' : nearest neighbor interpolation
        'linear' : linear interpolation
        'quadratic' : piecewize quadratic interpolation
        'log-log-linear' : linear interpolation in log space
        default value is 'log-log-linear'

    Method
    ______

    __call__

    Example
    -------

    >>> from matplotlib import pyplot as plt
    >>> import numpy as np
    >>> from photometry import spectrum
    >>> x = np.linspace(5., 15., 11)
    >>> y = np.zeros((3, len(x)))
    >>> y[0, :] = 27 - (x-10.)**2
    >>> y[1, :] = 27 - (x-10.)
    >>> y[2, :] = (x-10.)**2

    >>> fq = spectrum.SpectrumInterpolator(x, y, kind='quadratic',
    extrapolate=True)

    >>> xn = np.arange(21)+0.5
    >>> yq = fq(xn)

    >>> xp = np.linspace(5., 15., 101)
    >>> yp = 27 - (xp-10.)**2
    >>> plt.plot(xp, yp, label = 'True values')
    >>> plt.plot(x, y, 'o', label='Sampled values')
    >>> plt.plot(xn, yq, 'o', label = 'Quadratically interpolated')

    >>> fl = spectrum.SpectrumInterpolator(x, y, kind='linear')
    >>> yl = fl(xn)

    >>> plt.plot(xn, yl, 'o', label='Linearly interpolated')
    >>> plt.legend(loc='upper right')


    """
    def __init__(self, x, y, kind='log-log-linear', extrapolate=False):
        """
        Initialize the interpolator
        """
        # print("Initializing SpectrumInterpolator with")
        # print("  - kind = '{}'".format(kind))
        # print("  - extrapolate = {}".format(extrapolate))

        self.n = len(x)
        self.extrapolate = extrapolate

        if kind == 'nearest':
            self.x_low = x[0]
            self.x_high = x[-1]
            self.y = y
            self.x_bounds = (x[1:] + x[:-1]) / 2.
            self.x_bounds = np.append(self.x_bounds, [x[-1]])
            self._call = self.__class__._call_nearest
        elif kind == 'linear':
            self.x = x
            self.y = y
            if y.ndim == 2:
                self.slopes = (y[:,1:] - y[:,:-1]) / (x[1:] - x[:-1])
                self.ordinates = y[:,:-1] - self.slopes * x[:-1]
            else:
                self.slopes = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
                self.ordinates = y[:-1] - self.slopes * x[:-1]
            self._call = self.__class__._call_linear
        elif kind == 'quadratic':
            self.x = x
            nx = self.n
            xc = np.arange(nx, dtype='int64')
            xc[0] = 1
            xc[nx-1] = nx-2
            xm = xc - 1
            xp = xc + 1
            discrim = (x[xm] - x[xc]) * (x[xm] - x[xp]) * (x[xp] - x[xc])
            zz, = np.where(discrim == 0)
            if len(zz) > 0:
                raise ValueError("quadratic interpolation: discrimiment has "
                                 "0 values")
            if y.ndim == 2:
                self.b = ((y[:,xm] - y[:,xc]) *
                          (x[xm] * x[xm] - x[xp] * x[xp]) -
                          ((x[xm] * x[xm] - x[xc] * x[xc]) *
                           (y[:, xm] - y[:, xp]))) / discrim
                self.c = ((y[:, xm] - y[:, xp]) * (x[xm] - x[xc]) -
                      (x[xm] - x[xp]) * (y[:, xm] - y[:, xc])) / discrim
                self.a = y[:,xc] - self.b * x[xc] - self.c * x[xc] * x[xc]
            else:
                self.b = ((y[xm] - y[xc]) * (x[xm] * x[xm] -x[xp] * x[xp]) -
                         ((x[xm] * x[xm] - x[xc] * x[xc]) *
                          (y[xm] - y[xp]))) / discrim
                self.c = ((y[xm] - y[xp]) * (x[xm] - x[xc]) -
                          (x[xm] - x[xp]) * (y[xm] - y[xc])) / discrim
                self.a = y[xc] - self.b * x[xc] - self.c * x[xc] * x[xc]
            self._call = self.__class__._call_quadratic
        elif kind == 'log-log-linear':
            x = np.log10(x)
            self.x = x
            y = np.log10(y)
            if y.ndim == 2:
                self.slopes = (y[:,1:] - y[:,:-1]) / (x[1:] - x[:-1])
                self.ordinates = y[:,:-1] - self.slopes * x[:-1]
            else:
                self.slopes = (y[1:]-y[:-1])/(x[1:]-x[:-1])
                self.ordinates = y[:-1] - self.slopes * x[:-1]
            self._call = self.__class__._call_log_log_linear
        else:
            raise ValueError("Invalid interpolation method: {}".format(kind))

    def __call__(self, x):
        return self._call(self, x)

    def _call_nearest(self, x):
        idx = np.searchsorted(self.x_bounds, x)
        if self.y.ndim == 2:
            result = np.zeros((self.y.shape[0], x.shape[0]))
        else:
            result = np.zeros(x.shape)
        if self.extrapolate:
            bad = []
            over, = np.where(idx == len(self.x_bounds))
            if len(over) > 0:
                idx[over] = idx[over]-1
            ok = range(len(idx))
        else:
            bad, = np.where((x < self.x_low) | (x > self.x_high))
            ok, = np.where((x >= self.x_low) & (x <= self.x_high))
        if len(ok) > 0:
            if self.y.ndim == 2:
                result[:, ok] = self.y[:, idx[ok]]
            else:
                result[ok] = self.y[idx[ok]]
        if len(bad) > 0:
            if self.y.ndim == 2:
                result[:, bad] = np.NaN
            else:
                result[bad] = np.NaN
        # Normally, the input spectrum has no negative value, so no need to
        # check
        return result

    def _call_linear(self, x):
        idx = np.digitize(x, self.x)-1
        idx = idx.clip(0, self.n-2)
        if self.slopes.ndim == 2:
            result = np.zeros((self.y.shape[0], x.shape[0]))
        else:
            result = np.zeros(x.shape)
        if self.extrapolate:
            bad = []
            ok = range(len(x))
        else:
            ok, = np.where((x >= self.x[0]) & (x <= self.x[-1]))
            bad = np.where((x < self.x[0]) | (x > self.x[-1]))
        if len(ok) > 0:
            if self.slopes.ndim == 2:
                result[:, ok] = self.slopes[:, idx[ok]] * x[ok] + \
                                self.ordinates[:, idx[ok]]
            else:
                result[ok] = self.slopes[idx[ok]] * x[ok] + \
                                self.ordinates[idx[ok]]
        if len(bad) > 0:
            if self.slopes.ndim == 2:
                result[:, bad] = np.NaN
            else:
                result[bad] = np.NaN
        # Normally, the input spectrum has no negative value, so no need to
        # check
        return result

    def _call_quadratic(self, z):
        nexti = np.digitize(z, self.x)
        nexti[nexti < 1] = 1
        nexti[nexti >= self.n] = self.n-1
        previ = nexti - 1
        distprev = z - self.x[previ]
        distnext = self.x[nexti] - z
        dist = self.x[nexti] - self.x[previ]
        # computes the fit
        if self.a.ndim == 2:
            res = (distnext * (self.a[:, previ] + (self.b[:, previ] +
                                                z * self.c[:, previ]) * z) +
                   distprev * (self.a[:, nexti] + (self.b[:, nexti] +
                                                z * self.c[:, nexti]) * z)) / \
                  dist

        else:
            res = (distnext * (self.a[previ] + (self.b[previ] +
                                                z * self.c[previ]) * z) +
                   distprev * (self.a[nexti] + (self.b[nexti] +
                                                z * self.c[nexti]) * z)) / dist

        # sets to zero all negative values:
        oo = np.where(res < 0.)
        if len(oo) > 0:
            res[oo] = 0.

        if not self.extrapolate:
            oo, = np.where((z < self.x[0]) | (z > self.x[-1]))
            if len(oo) > 0:
                if self.a.ndim == 2:
                    res[:, oo] = np.NaN
                else:
                    res[oo] = np.NaN
        return res

    def _call_log_log_linear(self, xin):
        x = np.log10(xin)
        idx = np.digitize(x, self.x)-1
        idx = idx.clip(0, self.n-2)
        if self.slopes.ndim == 2:
            result = self.slopes[:,idx] * x + self.ordinates[:,idx]
        else:
            result = self.slopes[idx] * x + self.ordinates[idx]
        bad, = np.where((x < self.x[0]) | (x > self.x[-1]))
        if len(bad) > 0:
            if self.extrapolate:
                distmax = 10.**np.max(np.abs(x[bad]-self.x[idx[bad]]))
                print("Warning: extrapolation up to {}".format(distmax))
            else:
                print("Warning: extrapolation... expect NaN")
                if self.slopes.ndim == 2:
                    result[:,bad] = np.NaN
                else:
                    result[bad] = np.NaN
        return 10.**result


class GalaxySpectrum(BasicSpectrum):
    """
    A class to represent spectra of galaxies. It inherits BasicSpectrum and
    allows to take into account the redshift.
    """
    def __init__(self, z=None, cosmo=None, dl=None,
                 data=None, name_x=None, names_y=None,
                 x=None, x_type=None, x_unit=None,
                 y=None, y_type=None, y_unit=None,
                 interpolation_method='log-log-linear',
                 extrapolate=False):

        BasicSpectrum.__init__(self, data=data, name_x=name_x, names_y=names_y,
                               x=x, x_type=x_type, x_unit=x_unit,
                               y=y, y_type=y_type, y_unit=y_unit,
                               interpolation_method=interpolation_method,
                               extrapolate=extrapolate)


        self.z = z
        self.cosmo = cosmo
        if dl is None:
            self.dl = cosmo.luminosity_distance(z)
        else:
            self.dl = dl

    def __str__(self):
        result = "***** BasicSpectrum:\n" + BasicSpectrum.__str__(self)
        result += "\n***** GalaxySpectrum:"
        result += "\n   z = {}".format(self.z)
        result += "\n   dl = {}".format(self.dl)
        result += "\n   with cosmology = {}".format(self.cosmo)
        return result

    def redshift(self, redshift):
        """
        redshift the spectrum  to z

        Parameters
        ----------
        redshift: float
        """
        dl = self.cosmo.luminosity_distance(redshift)
        if self.is_nu:
            old_shift = 1. / (1. + self.z)
            new_shift = 1./(1. + redshift)
        elif self.is_lam:
            old_shift = 1. + self.z
            new_shift = 1. + redshift
        else:
            raise ValueError("Spectrum type is neither nu or lam")
        if self.is_fnu:
            old_fact = (1. + self.z) / 4. / np.pi / self.dl**2
            new_fact = (1. + redshift) / 4. / np.pi / dl**2
        elif self.is_flam:
            old_fact = 1./(1. + self.z) / 4. / np.pi / self.dl**2
            new_fact = 1./ (1. + redshift) / 4. / np.pi / dl**2
        else:
            raise ValueError("Spectrum is neither fnu or flam")

        self.shift(new_shift / old_shift)
        self.scale((new_fact / old_fact).value)
        self.z = redshift
        self.dl = dl
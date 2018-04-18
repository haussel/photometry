__author__ = 'haussel'
"""
This module provides the following classes:
- BasicSpectrum: a class to represent one or more spectra
- SpectrumInterpolator: a class to perform various interpolation
- GalaxySpectrum : a class to represent galaxy spectra.
"""
import numpy as np
from astropy import units as u
from .photcurve import PhotCurve
from .phottools import is_flam, \
    is_fnu, is_flux, velc , nu_unit, lam_unit, flam_unit, fnu_unit, nufnu_unit


class BasicSpectrum(PhotCurve):
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
    def __init__(self, file=None, table=None, name_x=None, names_y=None,
                 x=None, x_unit=None,
                 y=None, y_type=None, y_unit=None,
                 header=None,
                 interpolation_method='log-log-linear',
                 extrapolate='no', positive=True):
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
        table: astropy.table.Table or numpy.ndarray
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
        extrapolate: str
            If 'yes', spectra can be interpolated outside of the x array
            bounds without raising an exception. Defaulted to 'no'.


        """
        super().__init__(file=file, x=x, y=y, table=table,
                         colname_x=name_x, colnames_y=names_y,
                         x_unit=x_unit, header=header,
                         interpolation=interpolation_method,
                         extrapolate=extrapolate, positive=positive,
                         delayed_setup=True)

        if not self.initialized:
            return
        # After the init from PhotCurve, the x part has been dealt with, and the
        # y has been ingested as is, with self.y containing the values
        # (without any scaling) and self.org_y_unit containing the unit if any.
        if self.org_y_unit is not None:
            # this is the case where y was a quantity
            if y_unit is not None:
                if y_unit is not self.org_y_unit:
                    raise ValueError('y ({}) and y_unit ({}) do not '
                                     'match'.format(self.org_y_unit, y_unit))
            if is_fnu(self.org_y_unit):
                if y_type is not None:
                    if y_type != 'fnu':
                        raise ValueError('incompatible `y_type`:{} and '
                                         '`y.unit`:{}'.format(y_type,
                                                              self.org_y_unit))
                self.is_fnu = True
                self.is_flam = False
                self.y_si_unit = fnu_unit
                self.org_y_type = 'fnu'
                self.y = self.y * self.org_y_unit.to(fnu_unit)
            elif is_flam(self.org_y_unit):
                if y_type is not None:
                    if y_type != 'flam':
                        raise ValueError('incompatible `y_type`:{} and '
                                         '`y.unit`:{}'.format(y_type,
                                                              self.org_y_unit))
                self.y_si_unit = flam_unit
                self.is_flam = True
                self.is_fnu = False
                self.org_y_type = 'flam'
                self.y = self.y * self.org_y_unit.to(flam_unit)
            elif is_flux(self.org_y_unit):
                if self.org_x_type == 'lam':
                    if y_type is not None:
                        if y_type != 'lamflam':
                            raise ValueError('incompatible `y_type`:{} and '
                                              '`y.unit`:{}'.format(y_type,
                                                                      y_unit))
                    self.org_y_type = 'lamflam'
                    self.y_si_unit = flam_unit
                    self.is_flam = True
                    self.is_fnu = False
                    self.y = ((self.y * self.org_y_unit).to(nufnu_unit)/
                              (self.x * self.x_si_unit)).value
                elif self.org_x_type == 'nu':
                    if y_type is not None:
                        if y_type != 'nufnu':
                            raise ValueError('incompatible `y_type`:{} and '
                                             '`y.unit`:{}'.
                                             format(y_type, self.org_y_unit))
                    self.org_y_type = 'nufnu'
                    self.y_si_unit = fnu_unit
                    self.is_flam = False
                    self.is_fnu = True
                    self.y = ((self.y * self.org_y_unit).to(nufnu_unit) /
                              (self.x * self.x_si_unit)).value
                else:
                    raise ValueError('invalid unit {} for input quantity '
                                     '`x`'.format(self.org_x_unit))
            else:
                raise ValueError('invalid unit {} for input quantity `y`'.
                                 format(self.org_y_unit))
        else:
            if y_unit is not None:
                if is_fnu(y_unit):
                    if y_type is not None:
                        if y_type != 'fnu':
                            raise ValueError('incompatible `y_type`:{} and '
                                             '`y.unit`:{}'.format(y_type,
                                                                  y_unit))
                    self.is_fnu = True
                    self.is_flam = False
                    self.y_si_unit = fnu_unit
                    self.org_y_type = 'fnu'
                    self.org_y_unit = y_unit
                    self.y = self.y * y_unit.to(fnu_unit)
                elif is_flam(y_unit):
                    if y_type is not None:
                        if y_type != 'flam':
                            raise ValueError('incompatible `y_type`:{} and '
                                             '`y.unit`:{}'.format(y_type,
                                                                  y_unit))
                    self.y_si_unit = flam_unit
                    self.is_flam = True
                    self.is_fnu = False
                    self.org_y_type = 'flam'
                    self.org_y_unit = y_unit
                    self.y = self.y * y_unit.to(flam_unit)
                elif is_flux(self.org_y_unit):
                    if self.org_x_type == 'lam':
                        if y_type is not None:
                            if y_type != 'lamflam':
                                raise ValueError('incompatible `y_type`:{} and '
                                                 '`y.unit`:{}'.format(y_type,
                                                                      y_unit))
                        self.org_y_type = 'lamflam'
                        self.org_y_unit = y_unit
                        self.y_si_unit = flam_unit
                        self.is_flam = True
                        self.is_fnu = False
                        self.y = ((self.y * self.org_y_unit).to(nufnu_unit) /
                                  (self.x * self.x_si_unit)).value
                    elif self.org_x_type == 'nu':
                        if y_type is not None:
                            if y_type != 'nufnu':
                                raise ValueError('incompatible `y_type`:{} and '
                                                 '`y.unit`:{}'.format(y_type,
                                                                      y_unit))
                        self.org_y_type = 'nufnu'
                        self.org_y_unit = y_unit
                        self.y_si_unit = fnu_unit
                        self.is_flam = False
                        self.is_fnu = True
                        self.y = ((self.y * self.org_y_unit).to(nufnu_unit) /
                                  (self.x * self.x_si_unit)).value
                    else:
                        raise ValueError('invalid unit {} for input quantity '
                                         '`x`'.format(self.org_x_unit))
                else:
                    raise ValueError('invalid unit {} for input quantity `y`'.
                                     format(self.org_y_unit))
            else:
                raise ValueError("'y' has no units")
        self.interpolate_set = self.set_interpolation(
                interpolation_method, extrapolate=extrapolate,
                positive=positive)
        self.yfactor = np.ones(self.nb)
        self.xshift = 1.0
        self.Av = 0.0
        self.extinction = None

    def __getitem__(self, item):
        return BasicSpectrum(x=self.x * self.x_si_unit,
                             y=self.y[item,:] * self.y_si_unit,
                             interpolation_method=self.interpolate_method,
                             extrapolate=self.extrapolate,
                             positive=self.positive)


    def apply_galactic_extinction(self, Av):
        """
        Apply extinction
        :param Av:
        :return:
        """
        if self.extinction is not None:
            if self.Av > 0:
                oldfactor = self.extinction.compute_extinction(self.x *
                                                               self.x_si_unit,
                                                               self.Av)
            else:
                oldfactor = 1.
            newfactor = self.extinction.compute_extinction(self.x *
                                                           self.x_si_unit, Av)
            self.y *= newfactor/oldfactor

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

    def in_flam(self, reinterpolate=True):
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
            if reinterpolate:
                self.interpolate_set = self.set_interpolation(
                    self.interpolate_method,
                    extrapolate=self.extrapolate,
                    positive=self.positive)
            else:
                self.interpolate = None
                self.interpolate_set = False
        return None

    def in_fnu(self, reinterpolate=True):
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
            if reinterpolate:
                self.interpolate_set = self.set_interpolation(
                    self.interpolate_method,
                    extrapolate=self.extrapolate,
                    positive=self.positive)
            else:
                self.interpolate = None
                self.interpolate_set = False
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
        self.in_nu(reinterpolate=False)
        self.in_fnu(reinterpolate=True)
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
        self.in_lam(reinterpolate=False)
        self.in_flam(reinterpolate=True)
        if isinstance(lam, u.Quantity):
            result = self.interpolate(lam.to(lam_unit).value) * flam_unit
        else:
            result = self.interpolate(lam)
        return result

    def __str__(self):
        if self.header is not None:
            result = '{}\nspectrum:'.format(self.header)
        else:
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
            format(self.interpolate)
        result = result + "\n  ready for use: {}".\
            format(self.interpolate_set)
        return result


class GalaxySpectrum(BasicSpectrum):
    """
    A class to represent spectra of galaxies. It inherits BasicSpectrum and
    allows to take into account the redshift.
    """
    def __init__(self, z=None, cosmo=None, dl=None,
                 table=None, name_x=None, names_y=None,
                 x=None, x_unit=None,
                 y=None, y_type=None, y_unit=None,
                 interpolation_method='log-log-linear',
                 extrapolate='no', positive=True):

        BasicSpectrum.__init__(self, table=table, name_x=name_x,
                               names_y=names_y,
                               x=x, x_unit=x_unit,
                               y=y, y_type=y_type, y_unit=y_unit,
                               interpolation_method=interpolation_method,
                               extrapolate=extrapolate, positive=positive)
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
            old_fact = 1. / (1. + self.z) / 4. / np.pi / self.dl**2
            new_fact = 1. / (1. + redshift) / 4. / np.pi / dl**2
        else:
            raise ValueError("Spectrum is neither fnu or flam")

        self.shift(new_shift / old_shift)
        self.scale((new_fact / old_fact).value)
        self.z = redshift
        self.dl = dl


class StellarLibrary(BasicSpectrum):
    """
    Provide support for spectral libraries, such as the BaSeL 2.2 one (Lejeune
    et al., 1998) or Pickles (1998) as BasicSpectrum.

    The library can consist in a set of ascii files, such as the distribution
    off BaSeL 2.2, or a single fits file.

    The StellarLibrary class provides a mechanism to associate parameters to
    the spectra, allowing to find a match.

    The contructor must be passed either:
        * a reader function that will returns the wavelengths, spectra,
          and a matcher object
        * the wavelengths, spectra and the matcher object

    The matcher object is an object that contains the library parameters with
    the following methods:
    * __getitem__: to allow slicing so that the library parameters keeps in
                   line with slicing of the library
    * __str__: to format nicely the content of the library parameters
    * _closest_match: to allow to find the closest spectra in the library for a
                      set of input parameters
    * _exact_match: to allow to find the library spectrum matching exactly
                    the input parameters. It must raise a ValueError is no
                    match is found.

    See the BaSeL2p2Reader and BaSeL2p2Matcher classes as well as the
    convenience function BaSeL2p2 for examples of implementation

    Parameters:
    -----------
    match_is_exact: bool
        if True uses exact_match() method for match(), closest_match() if False
    matcher: object
        the object containing the parameters and allowing for matching
    inherited parameters from BasicSpectrum

    Methods:
    --------
    __getitem__:
        select a subsample of the library as a StellarLibrary
    __str__:
        print the library content
    closest_match:
        returns the closest matching spectrum as a one element StellarLibrary.
        Uses the _closest_match method of the matcher object.
    index_closest_match:
        returns the index of the closest match
    exact_match:
        returns the exact matching spectrum as a one element Stellar Library.
        Uses the _exact_match method of the matcher object.
    index_exact_match:
        returns the index of the exact match
    match:
        perform a closest or exact match depending on the value of
        match_is_exact
    index_match:
        returns the index of the match
    inherited methods of BasicSpectrum
    """
    def __init__(self, reader=None, x=None, y=None, matcher=None,
                 interpolation_method='log-log-linear',
                 extrapolate='no', positive=True, match_is_exact=True):
        """
        Initialization by providing either a reader method, or a set of x,
        values y and a matcher object:

        lib = StellarLibrary(reader=myReader)

        lib = StellarLibrary(x=x, y=y, matcher=myMatcher)
        """
        self.match_is_exact = match_is_exact
        if reader is not None:
            if x is None and y is None and matcher is None:
                (x, y, matcher) = reader()
            else:
                raise ValueError("Both reader and values are specified")
        if matcher is None:
            raise ValueError("No matching mechanism provided")
        self.matcher = matcher
        super().__init__(x=x, y=y, interpolation_method=interpolation_method,
                         extrapolate=extrapolate, positive=positive)

    def __getitem__(self, item):
        return StellarLibrary(x=self.x * self.x_si_unit,
                              y=self.y[item,:] * self.y_si_unit,
                              matcher=self.matcher[item],
                              interpolation_method=self.interpolate_method,
                              extrapolate=self.extrapolate,
                              positive=self.positive,
                              match_is_exact=self.match_is_exact)

    def __str__(self):
        result = self.matcher.__str__()
        result += super().__str__()
        return result

    def exact_match(self, *args):
        return self.__getitem__(self.matcher._exact_match(*args))

    def index_exact_match(self, *args):
        return self.matcher._exact_match(*args)

    def closest_match(self, *args):
        return self.__getitem__(self.matcher._closest_match(*args))

    def index_closest_match(self, *args):
        return self.matcher._closest_match(*args)

    def match(self, *args):
        if self.match_is_exact:
            return self.exact_match(*args)
        else:
            return self.closest_match(*args)

    def index_match(self, *args):
        if self.match_is_exact:
            return self.index_exact_match(*args)
        else:
            return self.index_closest_match(*args)
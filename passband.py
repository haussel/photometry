"""
This module provides the following classes:
- Passband : a class to hold a passband
- PassbandHeader: a class to hold the content of the header of a passband.
- PassbandInterpolator: a class to perform various interpolation for passbands

Provide also the functions:
- tophat
"""
__author__ = 'Herve Aussel'

import re
import os
import inspect
import numpy as np
from scipy import integrate
from astropy import units as u
from .phottools import is_wavelength, is_frequency, quantity_scalar, \
    quantity_1darray, ndarray_1darray, velc, nu_unit, lam_unit, flam_unit, \
    fnu_unit, nufnu_unit
from .spectrum import BasicSpectrum, GalaxySpectrum
from .config import DEBUG


class Passband:
    """
    A class to perform synthetic photometry.

    Passband is the main workhorse of the photometry package. A typical usage
    goes like this:

    import photometry as pt
    from astropy import units as u
    pb = Passband(file=my_passband_file)
    sp = BasicSpectrum(x=my_array_of_frequencies_or_wavelength,
                       y=my_arrays_of_fnu_or_flams)
    mymag = pb.mag_ab(sp)
    myfnu = pb.fnu_ir(sp)
    bandwidth = pb.bandwidth(u.micron)

    Internally, the passband is stored as an array of relative system response
    ('rsr' for short) or of quantum efficiency ('qe' for short). Depending on
    the quantities that will be computed, this representation will change. Do
    not assume what is in the 'x' attribute without first checking the internal
    representation.

    The internal representation of the passband can be visualised with a
    simple print(pb). The passband attributes is_lam and is_nu tell whether
    the passband is internally represented as a function of the wavelength or
    of the frequency.

    The way the passband interact with the spectrum (interpolation,
    integration) can be tuned with the set_location(), set_interpolation() and
    set_integration() methods (see their doc).
    """
    # Initialization Methods
    def __init__(self, file=None, x=None, y=None, xref=None, ytype=None,
                 header=None, location='both_spectrum_and_passband',
                 interpolation='quadratic', integration='trapezoidal',
                 nowarn=False):
        """

        Initialization. The passband can be:
            - read from a file. For example of passband files, look in the
              photometry/data/passbands. They are stored in subdirectories
              grouped by instrument.
            - initialized from astropy quantities x, y, xref and ytype. A
              header can be also passed in this case to add more information
            - initialized from arrays x, y and a header.

        Mandatory keywords in the passband file header are: xunit, xref,
        ytype.

        Parameters:
        -----------

        file: str
            filename containing the passband. Of the form 'filter.instrument.pb'

        x: astropy.unit.Quantity or numpy.ndarray
            array of frequencies or wavelengths. If x is an ndarray, the header
            must be provided to know what are the units.
        y: numpy.ndarray
            array of quantum efficiencies or relative system responses
        xref: astropy.unit.Quantity
            reference frequency or wavelength for the passband
        ytype: str
            type of transmission: 'qe' or 'rsr'
        header: PassbandHeader or dict or array of str
            the header of the passband
        location: str
            defines where passband and spectra are interpolated. Defaulted to 
            'both_spectrum_and_passband'. Can also be 'spectrum_at_passband'
            or 'passband_at_spectrum'
        interpolation:str
            defines the interpolation mehod. Defaulted to 'quadratic'. Other 
            methods are 'nearest' or 'linear'. See PassbandInterpolator.
        integration: str
            defines the integration method. Defaulted to 'trapezoidal'. Other 
            choice is 'simpson'
        nowarn: bool
            If set to True, no warning message of incomplete information is
            given

        Attributes:
        -----------
        header: PassbandHeader
            A special dictionary containing non crucial information about the
            passband
        file: str
            The name of the file containing the passband
        instrument: str
            The name of the instrument the passband applies to
        filter: str
            The name of the filter
        org_x_type: str
            The original type of the passband. Can be 'lam' or 'nu'
        org_x_unit: astropy.unit
            The original unit of the passband
        org_y_type: str
            The original type of the passband. Can be 'qe' or 'rsr'
        org_x_ref: float
            The original reference
        x : (N, ) numpy.ndarray
            The abscissa values. Can be frequency or wavelength,
            always in SI unit (x_si_unit)
        x_ref: float
            The reference frequency or wavelength for the passband in
            units of x_si_unit.
        x_si_unit: astropy.unit
            The unit for x and x_ref
        is_lam : boolean
            True is x and x_ref are wavelengths
        is_nu: boolean
            True is x and x_ref are frequencies
        y: (N,) numpy.ndarray
            the passband rsr (relative system response) or qe (quantum
            efficiency)
        is_qe: boolean
            True if the passband is a quantum efficiency
        is_rsr: boolean
            True if passband is a relative system response
        location_method: str
            method to determine where to interpolate the spectrum and the
            passband. Can be :
                - 'both_spectrum_and_passband'
                - 'spectrum_at_passband'
                - 'passband_at_spectrum'
        location_set: boolean
            True if the location method is set
        location: function
            The method that returns the place where to interpolate, given the
            spectrum and passband abscissa.
        interpolate_method: str
            method to interpolate the values of the passband. Can be
            - 'nearest' : nearest neighbor
            - 'linear'  : linear
            - 'quadratic' : piecewize quadratic
        interpolate_set: boolean
            True is the interpolation function is set
        interpolate: PassbandInterpolator
            Returns the inrterpolated values of y at requested x.
        integration_method: str
            the method used for numerical integration. Can be:
            - 'trapezoidal'
            - 'simpson'
        integrate_set: boolean
            True if the integration method is set.
        self.integrate: function
            The function that performs the integration
        ready: Boolean
            True if the passband is ready for use.
        initialized: Boolean
            True if enough inputs have been provided to initialized a passband  

        """
        self.header = PassbandHeader()
        self.file = None
        self.instrument = None
        self.filter = None
        self.org_x_type = None
        self.org_x_unit = None
        self.org_y_type = None
        self.org_x_ref = None
        self.x = None
        self.x_ref = None
        self.x_si_unit = None
        self.is_lam = False
        self.is_nu = False
        self.y = None
        self.is_qe = False
        self.is_rsr = False
        self.location_method = None
        self.location_set = False
        self.location = None
        self.interpolate_method = None
        self.interpolate_set = None
        self.interpolate = None
        self.integrate_set = None
        self.integration_method = None
        self.integrate = None
        self.ready = False
        self.intialized = False
        self.nowarn = False

        self.nowarn = nowarn

        if file is not None:
            if ((x is None) and (y is None) and (xref is None) and
                (y is None) and (ytype is None) and (header is None)):
                self.initialized = self.read(file)
            else:
                raise ValueError("Cannot set both file and x, y values")
        elif ((x is not None) and (y is not None) and (xref is not None) and
            (ytype is not None)):
            self.initialized = self._init_from_quantities(x, y, xref, ytype,
                                                          header=header)
        elif x is not None and y is not None and header is not None:
            self.initialized = self._init_from_header(x, y, header)
        else:
            pass

        if not self.initialized:
            raise ValueError("Insufficient data to initialize passband")
        else:
            self.set_location(location)
            self.set_interpolation(interpolation)
            self.set_integration(integration)

        self.fill_header()

        if self.initialized and self.location_set and self.interpolate_set and\
                self.integrate_set:
            self.ready = True
        else:
            self.ready = False

    def _init_from_quantities(self, x, y, xref, ytype, header=None):
        result = False
        # check x
        msg = quantity_1darray(x)
        if msg is not None:
            raise ValueError("`x`" + msg)
        # check xref
        msg = quantity_scalar(xref)
        if msg is not None:
            raise ValueError("`xref`" + msg)
        # check y
        msg = ndarray_1darray(y, length=len(x), other='x')
        if msg is not None:
            raise ValueError("`y`" + msg)
        # check ytype
        if not isinstance(ytype, str):
            raise ValueError('Invalid ytype {}'.format(ytype))

        if ytype.lower() == 'rsr':
            self.is_rsr = True
            self.is_qe = False
            self.org_y_type = 'rsr'
        elif ytype.lower() == 'qe':
            self.is_qe = True
            self.is_rsr = False
            self.org_y_type = 'qe'
        else:
            raise ValueError('Invalid value {} for `ytype`'.format(ytype))

        if is_frequency(x.unit):
            self.is_nu = True
            self.is_lam = False
            self.x_si_unit = nu_unit
            self.org_x_type = 'nu'
            self.org_x_unit = x.unit
        elif is_wavelength(x.unit):
            self.is_lam = True
            self.is_nu = False
            self.x_si_unit = lam_unit
            self.org_x_type = 'lam'
            self.org_x_unit = x.unit
        else:
            raise ValueError('invalid unit {} for input quantity `x`'.
                             format(x.unit))
        xv = x.to(self.x_si_unit).value
        idx = np.argsort(xv)
        self.y = y[idx]
        self.x = xv[idx]

        idx, = np.where(self.y < 0)
        if len(idx) > 0:
            print("Negative passband value have been set to zero")
            self.y[idx] = 0.

        self.org_x_ref = xref.value
        if is_frequency(xref.unit):
            if self.is_nu:
                self.x_ref = xref.to(self.x_si_unit).value
            else:
                self.x_ref = (velc / xref.to(lam_unit)).value
        elif is_wavelength(xref.unit):
            if self.is_lam:
                self.x_ref = xref.to(self.x_si_unit).value
            else:
                self.xref = (velc / xref.to(nu_unit)).value
        else:
            raise ValueError('`xref` unit {}'.format(xref.unit) +
                             'must be a wavelength or a frequency unit')

        # copy the header values
        if header is not None:
            if isinstance(header, PassbandHeader):
                self.header = header
            elif isinstance(header, dict):
                self.header.import_dict(header)
            elif isinstance(header, list):
                for elem in header:
                    self.header.import_line(elem)
            else:
                raise ValueError('Invalid type for header')
        result = True
        return result

    def _init_from_header(self, x, y, header):
        result = False
        # ingest the header
        if not isinstance(header, PassbandHeader):
            phd = PassbandHeader()
            if isinstance(header, dict):
                phd.import_dict(header)
            elif isinstance(header, list):
                for elem in header:
                    phd.import_line(elem)
            else:
                raise ValueError('Invalid type for header')
        else:
            phd = header
        if 'xunit' not in phd.content:
            raise ValueError('Missing keyword xunit in header')
        try:
            x_unit = u.Unit(phd.content['xunit'])
        except:
            raise ValueError('Invalid `xunit` {}'.format(phd.content['xunit']))
        # xtype
        if 'xtype' not in phd.content:
            raise ValueError('Missing keyword xtype in header')
        if phd.content['xtype'].lower() == 'lam' or \
            phd.content['xtype'].lower() == 'wave' or \
            phd.content['xtype'].lower() == 'wavelength':
            xtype = 'lam'
        elif phd.content['xtype'].lower() == 'nu' or \
            phd.content['xtype'].lower() == 'freq' or \
            phd.content['xtype'].lower() == 'frequency':
            xtype = 'nu'
        else:
            raise ValueError('Invalid key value for xtype in header: {}'.
                             format(phd.content['xtype']))
        # xref
        if 'xref' not in phd.content:
            raise ValueError('Missing keyword xref in header')
        try:
            xrefval = u.Quantity(phd.content['xref'])
            if xrefval.unit == u.dimensionless_unscaled:
                xrefval = xrefval * x_unit
        except:
            raise ValueError('Invalid key value for xref in header: {}'.
                             format(phd.content['xref']))

        # ytype
        if 'ytype' not in phd.content:
            raise ValueError('Missing keyword ytype in header')
        if phd.content['ytype'].lower() == 'rsr':
            ytype = 'rsr'
        elif phd.content['ytype'].lower() == 'qe':
            ytype = 'qe'
        else:
            raise ValueError('Invalid key value for ytype in header: {}'.
                             format(phd.content['ytype']))

        # check x
        msg = ndarray_1darray(x)
        if msg is not None:
            raise ValueError("`x`" + msg)
        # check y
        msg = ndarray_1darray(y, length = len(x), other = 'x')
        if msg is not None:
            raise ValueError("`y`" + msg)

        if isinstance(x, u.Quantity):
            if x.unit is not x_unit:
                raise ValueError('`x` and `xunit` key in header are different')
            xv = x
        else:
            xv = x * x_unit
        result = self._init_from_quantities(xv, y, xrefval, ytype, header=phd)
        return result

    def read(self, filename):
        """
        Read a passband from a file

        Parameters
        ----------
        filename: str
          The name of the file. If the file is not found, the passband is
          searched for in the data/passbands/ directory of the installation of
          the photometry package. For this it assumes that the filename follows
          the convention: filtername.instrumentname.pb.
        """
        result = False
        classpath = inspect.getfile(self.__class__)
        basepath = os.path.dirname(classpath)
        default_passband_dir = os.path.join(basepath, 'data/passbands/')
        if not os.path.exists(filename):
            head, tail = os.path.split(filename)
            (band, instrument, ext) = tail.split('.')
            path = os.path.join(default_passband_dir, instrument)
            full_filename = os.path.join(path, filename)
            if not os.path.exists(full_filename):
                raise IOError("file not found '{}'".format(full_filename))
        else:
            full_filename = filename
        # read the file. first the values, then browse the header
        values = np.genfromtxt(full_filename, comments='#',
                               dtype='float64', names=['x', 'y'])
        f = open(full_filename, 'r')
        line = f.readline()
        sheader = []
        while line[0] == '#':
            sheader.append(line)
            line = f.readline()
        f.close()
#        sheader.append('# file: '+ filename)
        result = self._init_from_header(values['x'], values['y'], sheader)
        return result

    def fill_header(self):
        """
        check self consistency between some attributes and the passband header
        and add them to the header in order to facilitate writing the passband
        to file if needed.
        """
        if 'file' not in self.header.content:
            if self.file is not None:
                self.header.add_card_value('file', self.file)
            else:
                if not self.nowarn:
                    print("Warning ! file needs to be set")
        else:
            if self.file is None:
                self.file = self.header.content['file']
            else:
                if self.file != self.header.content['file']:
                    if not self.nowarn:
                        print("Warning file does not match between header: "
                              "{}\nobject: {}"
                          .format(self.header.content['file'], self.file))

        if 'instrument' not in self.header.content:
            if self.instrument is not None:
                self.header.add_card_value('instrument', self.instrument)
            else:
                if not self.nowarn:
                    print("Warning ! instrument needs to be set")
        else:
            if self.instrument is None:
                self.instrument = self.header.content['instrument']
            else:
                if self.instrument != self.header.content['instrument']:
                    if not self.nowarn:
                        print("Warning instrument does not match between\n"
                              "header: {}\nobject: {}"
                          .format(self.header.content['instrument'],
                                  self.instrument))

        if 'filter' not in self.header.content:
            if self.filter is not None:
                self.header.add_card_value('filter', self.filter)
            else:
                if not self.nowarn:
                    print("Warning ! filter needs to be set")
        else:
            if self.filter is None:
                self.filter = self.header.content['filter']
            else:
                if self.filter != self.header.content['filter']:
                    if not self.nowarn:
                        print("Warning filter does not match between\n header: "
                              "{}\nobject: {}"
                          .format(self.header.content['filter'], self.filter))

        if 'xtype' not in self.header.content:
            if self.org_x_type is not None:
                self.header.add_card_value('xtype', self.org_x_type)
            else:
                raise ValueError("org_x_type is not set !")
        else:
            if self.org_x_type is None:
                raise ValueError("org_x_type is not set !")
            else:
                if self.org_x_type != self.header.content['xtype']:
                    # TODO get rid of this annoying warning with files where
                    # wavelength is indicated with lam
                    if self.nowarn:
                        print("Warning xtype does not match between\n"
                              "header: {}\nobject: {}"
                            .format(self.header.content['xtype'],
                                    self.org_x_type))

        if 'ytype' not in self.header.content:
            if self.org_y_type is not None:
                self.header.add_card_value('ytype', self.org_y_type)
            else:
                raise ValueError("org_y_type is not set !")
        else:
            if self.org_y_type is None:
                raise ValueError("org_y_type is not set !")
            else:
                if self.org_y_type != self.header.content['ytype']:
                    if self.nowarn:
                        print("Warning ytype does not match between\n"
                              "header: {}\nobject: {}"
                            .format(self.header.content['ytype'],
                                    self.org_y_type))

        if 'xref' not in self.header.content:
            if self.org_x_ref is not None:
                self.header.add_card_value('xref', self.org_x_ref)
            else:
                raise ValueError("org_x_ref is not set !")
        else:
            if self.org_x_ref is None:
                raise ValueError("org_x_ref is not set !")
            else:
                if self.org_x_ref != u.Quantity(self.header.content['xref']):
                    if self.nowarn:
                        print("Warning xref does not match between\n"
                              "header: {}\nobject: {}"
                            .format(self.header.content['xref'],
                                    self.org_x_ref))

    def __str__(self):
        result = "{}".format(self.header)
        result += "\nInternal settings:\n"
        result += "    location method: {}\n".format(self.location_method)
        result += "    interpolation method: {}\n".\
            format(self.interpolate_method)
        result += "    integration method: {}\n".format(self.integration_method)
        return result

    def set_location(self, location):
        """
        Defines the rules for interpolating the spectrum in the passband.

        Usually, the passband transmission is defined at some xp values, xs
        being a wavelength or a frequency, and the spectrum flux is defined at
        some other values xs. In order to compute the response of the system, a
        common grid of values x is needed, where both the passband and the
        spectrum will be evaluated. Three cases are possible:
        1) interpolate the passband at the spectrum xs, so x = xs
        2) interpolate the spectrum at the passband xp, so x = xp
        3) interpolate both the spectrum and the passband at the union of their
         abscissae, so x = [xs, xp].
        the set_location() method select which of these methods is going to be
        applied.


        Parameters
        ----------
        location : str
           can be either of three values:
            - 'passband_at_spectrum' : case 1 above
            - 'spectrum_at_passband' : case 2 above
            - 'both_spectrum_and_passband' : case 3 above


         Effects
         -------
         Sets the values of the following 3 class attributes:
         - location
         - location_set
         - location_method

        """
        result = False
        try:
            if location == 'both_spectrum_and_passband':
                self.location = self._interpolate_both_spectrum_and_passband
                result = True
            elif location == 'spectrum_at_passband':
                self.location == self._interpolate_spectrum_at_passband
                result = True
            elif location == 'passband_at_spectrum':
                self.location = self._interpolate_passband_at_spectrum
                result = True
            else:
                print("Invalid interpolation location method {}:".\
                      format(location))
        except:
            raise ValueError("Invalid interpolation location method {}:".\
                             format(location))
        if result == True:
            self.location_set = True
            self.location_method = location
        return None

    # interpolation location methods
    def _interpolate_both_spectrum_and_passband(self, xs, xp):
        idx, = np.where((xs >= xp[0]) & (xs <= xp[-1]))
        if len(idx) > 0:
            allx = np.unique(np.concatenate([xs[idx], xp]))
        else:
            allx = np.unique(np.concatenate([xs, xp]))
        return allx

    def _interpolate_spectrum_at_passband(self, xs, xp):
        return xp

    def _interpolate_passband_at_spectrum(self, xs, xp):
        idx, = np.where((xs >= xp[0]) & (xs <= xp[-1]))
        if len(idx) > 0:
            result = xs[idx]
        else:
            result = xs
        return xs

    def set_interpolation(self, interpolation):
        """
        Sets the interpolation method of the passband.


        Parameters
        ----------
        interpolation: str
            the chosen interpolation method. Can be
            - 'nearest': nearest point interpolation
            - 'linear': linear interpolation
            - 'quadratic': polynomial of order 2 interpolation

        Effects
        -------
        Sets the value of the following attributes:
         - interpolate: the function performing the interpolation
         - interpolate_method: the name of the method
         - interpolate_set: flag indicating whether the passband is ready to
           interpolate.

        """
        try:
            self.interpolate = PassbandInterpolator(self.x, self.y,
                                                    interpolation)
            result = True
        except:
            print("Problem setting interpolation with {}".\
                  format(interpolation))
            result = False
        if result == True:
            self.interpolate_method = interpolation
            self.interpolate_set = True
        return result

    def set_integration(self, method_name):
        """
        select the integration method in the passband

        Parameters
        ----------
        method_name: str
          Can be any of the two methods:
          - trapezoidal : use numpy.trapz
          - simpson     : use numpy.simps
        """
        result = True
        try:
            if method_name == 'trapezoidal':
                self.integrate = self._integrate_trapezoidal
            elif method_name == 'simpson':
                self.integrate = self._integrate_simpson
            else:
                print("invalid integration method {}".format(method_name))
                result = False
        except:
            print("invalid integration method {}".format(method_name))
            result = False
        if result == True:
            self.integration_method = method_name
            self.integrate_set = True
        return result

    def _integrate_trapezoidal(self, y, x):
        return np.trapz(y, x=x, axis=-1)

    def _integrate_simpson(self, y, x):
        return integrate.simps(y, x=x, axis=-1)

    def response(self):
        """
        returns the passband y values
        """
        return self.y

    def nu(self, unit):
        """
        return the passband frequency in unit. Raise an exception if the 
        passband is as a function of wavelength
        
        Parameters:
        -----------
        unit: astropy.unit
            The frequency unit

        Returns:
        --------
        the x of the passband in the requested unit.
        """
        if DEBUG:
            print('Passband.nu()')
        if self.is_nu:
            if is_frequency(unit):
                result = (self.x * self.x_si_unit).to(unit)
            else:
                raise ValueError("Invalid output unit {} for passband in "
                                 "frequency".format(unit))
        else:
            raise ValueError('Convert your passband to frequency first')
        return result

    def lam(self, unit):
        """
        return the passband wavelength in unit. Raise an exception if the 
        passband is as a function of frequency
        
        Parameters:
        -----------
        unit: astropy.unit
            The wavelength unit

        Returns:
        --------
        the x of the passband in the requested unit.
        """
        if self.is_lam:
            if is_wavelength(unit):
                result = (self.x * self.x_si_unit).to(unit)
            else:
                raise ValueError("Invalid output unit {} for passband in "
                                 "wavelength".format(unit))
        else:
            raise ValueError('Convert your passband to wavelength first')
        return result

    def in_nu(self):
        """
        Change the type of the x axis to frequency if needed

        Effects
        -------
        Change the following attributes if needed:
        - x : becomes a frequency array in ascending order
        - x_ref: becomes a frequency
        - x_si_unit: becomes the SI frequency unit (astropy.unit.Hz)
        - y : reversed order
        - is_nu: True
        - is_lam: False
        - interpolate: the interpolation method is been recomputed.

        """
        if self.is_lam:
            self.x = ((velc / self.x.copy())[::-1])
            self.x_ref = velc / self.x_ref
            self.x_si_unit = nu_unit
            self.y = self.y.copy()[::-1]
            self.set_interpolation(self.interpolate_method)
            self.is_lam = False
            self.is_nu = True
        return None

    def in_lam(self):
        """
        Change the type of the x axis to wavelength if needed

        Effects
        -------
        Change the following attributes if needed:
        - x : becomes a wavelength array in ascending order
        - x_ref: becomes a wavelenght
        - x_si_unit: becomes the SI wavelength unit (astropy.unit.m)
        - y : reversed order
        - is_nu: False
        - is_lam: True
        - interpolate: the interpolation is being recomputed
        """
        if self.is_nu:
            self.x = ((velc / self.x.copy())[::-1])
            self.x_si_unit = lam_unit
            self.x_ref = velc / self.x_ref
            self.y = self.y.copy()[::-1]
            self.set_interpolation(self.interpolate_method)
            self.is_lam = True
            self.is_nu = False
        return None

    def fnu_ab(self, spectrum):
        """
        Compute the spectral irradiance in the band of the spectrum, following
        the AB convention, e.g. assuming a fnu = cst flux in the band.

        Parameters
        ----------
        spectrum: BasicSpectrum object

        Returns
        -------
        output: astropy.quantity
            the flux with proper units

        Effects
        -------
        The passband is switched as a function of frequency if needed.
        The spectrum as a fnu as a function of frequency if needed.
        """
        if not isinstance(spectrum, BasicSpectrum):
            raise ValueError("Input must be a Spectrum")
        # work in frequencies
        self.in_nu()
        nus = spectrum.nu()
        allnu = self.location(nus, self.x)
        try:
            ifnu = spectrum.fnu_nu(allnu)
        except:
            raise ValueError("Input spectrum and passband do not overlap"
                             " completely")
        itnu = self.interpolate(allnu)
        if self.is_rsr:
            flux = self.integrate(ifnu * itnu, allnu)
            bandwidth = self.integrate(itnu, allnu)
        else:
            flux = self.integrate(ifnu * itnu * (self.x_ref / allnu), allnu)
            bandwidth = self.integrate(itnu * (self.x_ref / allnu), allnu)
        return flux / bandwidth * fnu_unit

    def mag_ab(self, spectrum):
        """
        Compute the AB magnitude of the spectrum in the band

        Parameters
        ----------
        spectrum: BasicSpectrum object

        Returns
        -------
        output: astropy.quantity
            The ab magnitude. This quantity has no unit.

        Effects
        -------
        The passband is switched as a function of frequency if needed.
        The spectrum as a fnu as a function of frequency if needed.
        """
        test = self.fnu_ab(spectrum)
        return -2.5 * np.log10(test/(3631e-26 * fnu_unit))

    def fnu_ir(self, spectrum):
        """
        Compute the spectral irradiance in the band of the spectrum, assuming
        the IR convention, i.e. quoting the flux fnu that a source with a
        nu.fnu = cst spectrum would have at the reference frequency

        Parameters
        ----------
        spectrum: BasicSpectrum object

        Returns
        -------
        output: astropy.quantity
            the flux in the passband with proper unit

        Effects
        -------
        The passband is switched as a function of frequency if needed.
        The spectrum as a fnu as a function of frequency if needed.
       """
        if not isinstance(spectrum, BasicSpectrum):
            raise ValueError("Input must be a Spectrum")
        # work in frequencies
        self.in_nu()
        nus = spectrum.nu()
        allnu = self.location(nus, self.x)
        try:
            ifnu = spectrum.fnu_nu(allnu)
        except:
            raise ValueError("Input spectrum and passband do not overlap"
                             " completely")
        itnu = self.interpolate(allnu)
        if self.is_rsr:
            flux = self.integrate(ifnu * itnu, allnu)
            bandwidth = self.integrate(itnu * self.x_ref / allnu, allnu)
        else:
            flux = self.integrate(ifnu * itnu / allnu, allnu)
            bandwidth = self.integrate(itnu * (self.x_ref / allnu) / allnu,
                                       allnu)
        return (flux / bandwidth) * fnu_unit

    def flux(self, spectrum):
        """
        Computes the inband irradiance (W/m**2) of the spectrum in the band.

        Parameters
        ----------
        spectrum: BasicSpectrum object

        Returns
        -------
        output: astropy.quantity
            the inband irradiance with proper units.

        Effects
        -------
        If the band is in frequencies, and the spectrum in wavelengths, the
        spectrum is converted in frequencies. Conversely, if the band is in
        wavelength, and the spectrum in frequencies, it is converted in
        wavelength
        """
        if not isinstance(spectrum, BasicSpectrum):
            raise ValueError("Input must be a Spectrum")
        if self.is_nu:
            nus = spectrum.nu()
            allnu = self.location(nus, self.x)
            try:
                ifnu = spectrum.fnu_nu(allnu)
            except:
                raise ValueError("Input spectrum and passband do not overlap"
                                 " completely")
            itnu = self.interpolate(allnu)
            if self.is_rsr:
                flux = self.integrate(ifnu * itnu, allnu)
            else:
                flux = self.integrate(ifnu * itnu * (self.x_ref / allnu),
                                      allnu)
        else:
            lams = spectrum.lam()
            alllam = self.location(lams, self.x)
            try:
                iflam = spectrum.flam_lam(alllam)
            except:
                raise ValueError("Input spectrum and passband do not overlap"
                                 " completely")
            itlam = self.interpolate(alllam)
            if self.is_rsr:
                flux = self.integrate(iflam * itlam, alllam)
            else:
                flux = self.integrate(iflam * itlam * (alllam/self.x_ref),
                                      alllam)
        return flux * nufnu_unit

    def flam_st(self, spectrum):
        """
        Computes the spectral irradiance in a passband of a spectrum, following
        the flam = cst convention (sometimes referred to as the ST convention,
        for hubble Space Telescope, hence the name).

        Parameters
        ----------
        spectrum: BasicSpectrum object

        Returns
        -------
        output: astropy.quantity
            the spectral irradiance with proper units

        Effects
        -------
        The passband is transformed as a function of the wavelength if needed
        The spectrum is transformed as flam as a function of the wavelength if
        needed.
        """
        if not isinstance(spectrum, BasicSpectrum):
            raise ValueError("Input must be a Spectrum")
        # work in wavelengths
        self.in_lam()
        lams = spectrum.lam()
        alllam = self.location(lams, self.x)
        try:
            iflam = spectrum.flam_lam(alllam)
        except:
            raise ValueError("Input spectrum and passband do not overlap"
                             " completely")
        itlam = self.interpolate(alllam)
        if self.is_rsr:
            flux = self.integrate(iflam * itlam, alllam)
            bandwidth = self.integrate(itlam, alllam)
        else:
            flux = self.integrate(iflam * itlam * (alllam/ self.x_ref), alllam)
            bandwidth = self.integrate(itlam * (alllam / self.x_ref), alllam)
        return flux / bandwidth * flam_unit

    def bandwidth(self, unit, normalize=False):
        """
        Computes the bandwidth of the passband

        Parameters
        ----------
        unit: astropy.unit
          The unit of the output bandwidth.

        normalize: bool
          if set to true, the passband is normalized to its maximum transmission

        Returns
        -------
        output: astropy.quantity
            bandwidth of the passband

        Effects:
        --------
        If the unit requested does not match the type of the passband, the
        passband is converted first into the type of the unit (frequency or
        wavelength)
        """
        if is_wavelength(unit):
            if self.is_nu:
                self.in_lam()
        elif is_frequency(unit):
            if self.is_lam:
                self.in_nu()
        else:
            raise ValueError("Invalid unit for bandwidth: {}".format(unit))
        if normalize:
            result = (self.integrate(self.y/np.max(self.y), self.x) *
                      self.x_si_unit).to(unit)
        else:
            result = (self.integrate(self.y, self.x) * self.x_si_unit).to(unit)
        return result

    def xref(self, unit):
        """
        returns the reference frequency in unit  
        
        Parameters:
        -----------
        unit: astropy.unit
            the requested unit. 
        
        Returns:
        --------
        astropy.unit.Quantity
        
        """
        if is_wavelength(unit):
            if self.is_nu:
                self.in_lam()
        elif is_frequency(unit):
            if self.is_lam:
                self.in_nu()
        else:
            raise ValueError("Invalid unit for xref: {}".format(unit))
        return (self.x_ref * self.x_si_unit).to(unit)

    def combine(self, other):
        """
        Combine the passband with an other passband

        Parameters
        ----------
        other: Passband
        The other passband to combine with.

        Returns
        -------
        output: Passband
        Output passband. It is only partially initialized, just with x, y, xref
        (inherited from self) and ytype.
        """
        if not isinstance(other, Passband):
            raise ValueError("Input is not a passband")
        if self.is_nu:
            if other.is_lam:
                self.in_lam()
        else:
            if other.is_nu:
                self.in_nu()
        # determine the type of passband:
        if self.is_rsr:
            if other.is_rsr:
                restype='rsr'
            else:
                restype='qe'
        else:
            if other.is_rsr:
                restype='qe'
            else:
                restype='qe'
        allx = self.location(self.x, other.x)
        intts = self.interpolate(allx)
        intto = other.interpolate(allx)
        ally = intts * intto
        # determine the type of passband
        result = Passband(x=allx * self.x_si_unit,
                          xref=self.xref(self.x_si_unit),  y=ally,
                          ytype=restype, nowarn=True)
        return result


    def write(self, xunit, dir=None, overwrite=False, force=False):
        """
        Write a passband to a file. The unit for the x-axis does nor need to be 
        the same as the internal representation.
        By default, the file name is constructed from from the instrument and 
        filter names of the passband in the form: instrument.filter.pb
        If the 'file' card is present in the passband header and it does not 
        match with the default file, and error is raised, unless the 'force' 
        keyword is True 
        If the file exists, nothing is written unless the overwrite keyword is 
        True.

        Parameters
        ----------
        xunit : astropy.unit
            The unit to use to write the file.
        dir : str
            The directory where to write the file. If not given, the file
            will be written into data/passbands of the photometry package.
            If the file already exists, an error is raised.
        overwrite: bool
            If True, will overwrite the existing passband file. Defaulted 
            to False
        force: bool
            If True, will use the file name in the header, even if does 
            not match the default name.

        Returns
        -------
        output: str
            The full filename (including path) of the file that has been
            written

        Note: The file need some manual editing to put back the header lines
        in a nice order. An intelligent format to accomodate the precision
        would be nice.
        """
        # build teh default file name
        if 'instrument' not in self.header.content:
            raise ValueError('instrument is not set in header')
        instrument = self.header.content['instrument']
        if 'filter' not in self.header.content:
            raise ValueError('filter is not set in header')
        filter = self.header.content['filter']
        file = self.header.content['filter']+'.'+\
               self.header.content['instrument']+'.pb'
        if 'file' in self.header.content:
            if file != self.header.content['file']:
                if (force):
                    print("Warning ! will write to file: {} ".
                          format(self.header.content['file']))
                    print(" and not to default file {}".format(file))
                else:
                    raise ValueError('Default filename {} does not match '
                                     'expected filename {}. Use force keyword'.
                                     format(file,self.header.content['file'] ))

        if dir is None:
            classpath = inspect.getfile(self.__class__)
            basepath = os.path.dirname(classpath)
            default_passband_dir = os.path.join(basepath, 'data/passbands/')
            passband_dir = os.path.join(default_passband_dir, instrument + '/')
            if not os.path.exists(passband_dir):
                os.mkdir(passband_dir)
        else:
            if not os.path.exists(dir):
                raise ValueError("directory {} does not exist".format(dir))
            passband_dir = os.path.join(dir, instrument + '/')
            if not os.path.exists(passband_dir):
                os.mkdir(passband_dir)

        fullfilename = os.path.join(passband_dir, file)
        if os.path.exists(fullfilename):
            if not overwrite:
                raise ValueError("The passband file {} already exists ! Aborting".
                                 format(fullfilename))

        # check xunit
        if self.is_lam:
            if is_wavelength(xunit):
                xvals = (self.x * self.x_si_unit).to(xunit).value
                xrefval = (self.x_ref * self.x_si_unit).to(xunit).value
            else:
                raise ValueError("Invalid output unit {xunit} for passband in "
                                 "wavelenght".format(xunit))
        elif self.is_nu:
            if is_frequency(xunit):
                xvals = (self.x * self.x_si_unit).to(xunit).value
                xrefval = (self.x_ref * self.x_si_unit).to(xunit).value
            else:
                raise ValueError("Invalid output unit {xunit} for passband in"
                                 " frequency".format(xunit))
        else:
            raise ValueError("Passband is neither in freq or lam ")

        with open(fullfilename, 'w') as f:
            for key, value in self.header.content.items():
                if key is not 'xunit':
                    f.write("{}\n".format(self.header.format_card(key, value)))
            f.write("# xunit: {}\n".format(xunit))
            for i, xval in enumerate(xvals):
                f.write("{:>0.6f}    {:>0.6f}\n".format(xval, self.y[i]))
        return fullfilename

    def psf_weight(self, spectrum, xarr):
        """
        Compute the weight to apply to psf computation.

        Parameters:
        -----------
        spectrum: BasicSpectrum object
            the spectrum used for the weighting. If more than one spectrum is
            present... decide what to do.

        xarr :  astropy.unit.Quantity of N elements
            Positons (wavelength or frequency) where the PSF have been
            computed. The PSF is considered constant between two consecutive
            mid-points in xarr: hence the following:

            xarr[0]      xarr[1]   xarr[2]       xarr[N-2]    xarr[N]
            |        |     .     |    .             .       |
            | PSF[0] |   PSF[1]  |    .     ...   PSF[N-2]  | PSF[N-1]
            |        |     .     |    .             .       |

        :return:
            numpy.ndarray of N weigths
        """
        if not isinstance(spectrum, BasicSpectrum):
            raise ValueError("Spectrum is not a BasicSpectrun")
        msg = quantity_1darray(xarr)
        if msg:
            raise ValueError("xarr"+msg)
        if self.is_qe:
            ytype='qe'
        else:
            ytype='rsr'

        lims = np.copy(xarr.value)
        lims[1:] = 0.5 * (xarr[1:].value + xarr[0:-1].value)
        lims = np.append(lims, xarr[-1].value)
        allx = self._interpolate_both_spectrum_and_passband(lims, self.x)
        ally = self.interpolate(allx)
        weights = np.zeros(xarr.shape)
        for i in range(xarr.shape[0]):
            idx, = np.where((allx >= lims[i]) & (allx <= lims[i+1]))
            if len(idx)<2:
                weights[i] = 0
            else:
                result = Passband(x=lims[idx], y=ally[idx],
                                  xref= 0.5*(lims[idx[0]]+lims[idx[-1]]),
                                  ytype=ytype,
                                  interpolation='linear',
                                  integration='trapezoidal',
                                  nowarn=True)
                weights[i] = result.flux(spectrum)
        return weights / np.sum(weights)


#        lowlims = np.copy(xarr.value)
#        lowlims[1:] = 0.5*(xarr[1:].value + xarr[0:-1].value)
#        highlims = np.copy(xarr.value)
#        highlims[0:-1] = lowlims[1:]
#        weights = np.zeros(xarr.shape)
#        for i, low in enumerate(lowlims):
#            high = highlims[i]
#            bit = tophat(np.array([low, high]) * xarr.unit)
#            slice = self.combine(bit)
#            weights[i] = slice.flux(spectrum).value
#        return weights/np.sum(weights)


    def shift_passband(self, shift_factor):
        """
        Shift the passband (and not the atmosphere), by multiplying the
        frequency array by shift_factor.

        Parameters:
        -----------
        shift_factor: float

        """
        self.x = self.x * shift_factor
        try:
            self.shift = self.shift * shift_factor
        except:
            self.shift = shift_factor
        self.set_interpolation(self.interpolate_method)


    def unshift_passband(self):
        """
        Unshift a passband
        """
        try:
            shift_factor = 1./self.shift
        except:
            print("Passband is not shifted")
            shift_factor = 1.
        self.shift_passband(shift_factor)


    def offset_passband(self, offset):
        """
        Shift the passband (and not the atmosphere), by adding the offset to
        the frequencies

        Parameters:
        -----------
        offset: astropy.unit.Quantity

        """
        self.x  = self.x + offset.to(self.x_si_unit).value
        try:
            self.offset = self.offset + offset
        except:
            self.offset = offset
        self.set_interpolation(self.interpolate_method)


    def unoffset_passband(self):
        """
        Unshift a passband
        """
        try:
            offset = -self.offset
        except:
            print("Passband is not offsetted")
            if self.is_nu:
                offset = 0. * u.Hz
            else:
                offset = 0. * u.m
        self.offset_passband(offset)

    def distort_passband(self, alpha):
        """
        Distort a passband by multiplying it by (x/x_ref)**alpha

        Parameter:
        alpha: float
        """
        self.y = self.y * (self.x / self.x_ref)**alpha
        self.y = self.y / np.max(self.y)
        self.set_interpolation(self.interpolate_method)



class PassbandHeader:
    """
    A Class to handle the headers of passband files

    Attribute:
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



class PassbandInterpolator:
    """
    Interpolate a passband. The need for a specific class arises from the fact
    that scipy.interpolate raises an exception when out of bound values are
    requested, while for a passband, the result should be zero.

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
    y : (N, ) ndarray
        A 1-D array of real values. The length of `y` must be equal to the
        length of `x`
    kind: str optional
        Specifies the kind of interpolation as a string:
        'nearest' : nearest neighbor interpolation
        'linear' : linear interpolation
        'quadratic' : piecewize quadratic interpolation
        default value is 'quadratic'

    Method
    ______

    __call__

    Example
    -------

    >>> from matplotlib import pyplot as plt
    >>> import numpy as np
    >>> from photometry import passband
    >>> x = np.linspace(5., 15., 11)
    >>> y = 27 - (x-10.)**2

    >>> fq = passband.PassbandInterpolator(x, y, kind='quadratic')

    >>> xn = np.arange(21)+0.5
    >>> yq = fq(xn)

    >>> xp = np.linspace(5., 15., 101)
    >>> yp = 27 - (xp-10.)**2
    >>> plt.plot(xp, yp, label = 'True values')
    >>> plt.plot(x, y, 'o', label='Sampled values')
    >>> plt.plot(xn, yq, 'o', label = 'Quadratically interpolated')

    >>> fl = passband.PassbandInterpolator(x, y, kind='linear')
    >>> yl = fl(xn)

    >>> plt.plot(xn, yl, 'o', label='Linearly interpolated')
    >>> plt.legend(loc='upper right')


    """
    def __init__(self, x, y, kind='quadratic'):
        """
        Initialize the interpolator. The attributes depends on the kind of
        interpolation selected.
        """
        self.n = len(x)
        self.x = x
        self.y = y
        self.methods = {'nearest', 'linear', 'quadratic'}
        if kind == 'nearest':
            self.x_bounds = (x[1:] + x[:-1]) / 2.
            self.x_bounds = np.append(self.x_bounds, [x[-1]])
            self._call = self.__class__._call_nearest
        elif kind == 'linear':
            self.slopes = (y[1:]-y[:-1])/(x[1:]-x[:-1])
            self.ordinates = (y[:-1] * x[1:] - y[1:] * x[:-1])/(x[1:]-x[:-1])
            self._call = self.__class__._call_linear
        elif kind == 'quadratic':
            nx = self.n
            xc = np.arange(nx, dtype='int64')
            xc[0] = 1
            xc[nx-1] = nx-2
            xm = xc - 1
            xp = xc + 1
            discrim = (x[xm] - x[xc]) * (x[xm] - x[xp]) * (x[xp] - x[xc])
            zz, = np.where(discrim == 0)
            if len(zz) > 0:
                raise ValueError("quadratic interpolation: discrimiment has 0"
                                 " values")
            self.b = ((y[xm] - y[xc]) * (x[xm] * x[xm] -x[xp] * x[xp]) -
                      ((x[xm] * x[xm] - x[xc] * x[xc]) * (y[xm] - y[xp]))) / \
                     discrim
            self.c = ((y[xm] - y[xp]) * (x[xm] - x[xc]) -
                      (x[xm] - x[xp]) * (y[xm] - y[xc])) / discrim
            self.a = y[xc] - self.b * x[xc] - self.c * x[xc] * x[xc]
            self._call = self.__class__._call_quadratic
        else:
            raise ValueError("Invalid interpolation method: {}".format(kind))

    def list_methods(self):
        """
        List the available methods of interpolation
        """
        return self.methods

    def __call__(self, x):
        return self._call(self, x)

    def _call_nearest(self, x):
        idx = np.searchsorted(self.x_bounds, x)
        result = np.zeros(x.shape)
        ok, = np.where((x >= self.x[0]) & (x <= self.x[-1]))
        if len(ok) > 0:
            result[ok] = self.y[idx[ok]]
        # a check for negatives values is not necessary since a passband has
        # only positive values within its bounds
        return result

    def _call_linear(self, x):
        idx = np.digitize(x, self.x)-1
        idx = idx.clip(0, self.n-2)
        result = np.zeros(x.shape)
        ok, = np.where((x >= self.x[0]) & (x <= self.x[-1]))
        if len(ok) > 0:
            result[ok] = self.slopes[idx[ok]] * x[ok] + self.ordinates[idx[ok]]
        # a check for negatives values is not necessary since a passband has
        # only positive values within its bounds
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
        res = (distnext * (self.a[previ] + (self.b[previ] +
                                            z * self.c[previ]) * z) +
               distprev * (self.a[nexti] + (self.b[nexti] +
                                            z * self.c[nexti]) * z)) / dist

        # sets to zero all values outside the range in x,
        # as well as negative values
        oo, = np.where((z < self.x[0]) | (z > self.x[-1]) | (res < 0.))

        if len(oo) > 0:
            res[oo] = 0.

        return res

def tophat(xlims, ytype='rsr'):
    """
    Generate a tophat bandpass, with 1.0 between xfrom and xto, and 0 elsewhere.

    Parameters:
    -----------
    xlims: astropy.Quantity 2 elements

    Return:
    -------
    A tophat passband

    """
    msg = quantity_1darray(xlims, length=2)
    if msg:
        raise ValueError("xlims" + msg)
#    xvals = np.array([xlims[0].value*(1.-1e-10), xlims[0].value,
        # xlims[1].value,
#                    xlims[1].value*(1.+1e-10)])*xlims.unit
    yvals = np.array([1.,1.])
    result = Passband(x=xlims, y=yvals, xref=0.5*np.sum(xlims), ytype=ytype,
                      interpolation='nearest', nowarn=True)
    return result



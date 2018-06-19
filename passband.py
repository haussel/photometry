"""
This module provides the following classes:
- Passband : a class to hold a passband

Provide also the functions:
- tophat : a tophat passband
"""
__author__ = 'Herve Aussel'

import os
import inspect
import numpy as np
from astropy import units as u
from .photcurve import PhotCurve
from .phottools import is_wavelength, is_frequency, quantity_scalar, \
    quantity_1darray, ndarray_1darray, velc, nu_unit, lam_unit, flam_unit, \
    fnu_unit, nufnu_unit, PhotometryInterpolator, PhotometryHeader, \
    read_photometry_file, write_photometry_file
from .spectrum import BasicSpectrum, GalaxySpectrum
from .config import DEBUG


class Passband(PhotCurve):
    """
    A class to perform synthetic photometry.

    Passband is the main workhorse of the photometry package. A typical usage
    goes like this:

    import photometry as pt
    from astropy import units as u
    pb = pt.Passband(file=my_passband_file)
    sp = pt.BasicSpectrum(x=my_array_of_frequencies_or_wavelength,
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


    Attributes inherited from PhotCurve:
    ------------------------------------
    file: str
        The name of the file containing the passband
    header: PhotometryHeader
        A special dictionary containing non crucial information about the
        passband
    x : (N, ) numpy.ndarray
        The abscissa values. Can be frequency or wavelength,
        always in SI unit (x_si_unit)
    x_si_unit: astropy.unit
        The unit for x and x_ref
    org_x_type: str
        The original type of the passband. Can be 'lam' or 'nu'
    org_x_unit: astropy.unit
        The original unit of the passband
    y: (N,) numpy.ndarray
        the passband rsr (relative system response) or qe (quantum
        efficiency)
    org_y_unit: astropy.units.Quantity
        the original unit of y.
    nb: int
        the number of curves: 1 if y is 1D, and y.shape[0] if y is 2D.
    is_lam : boolean
        True is x and x_ref are wavelengths
    is_nu: boolean
        True is x and x_ref are frequencies
    interpolate_method: str
        method to interpolate the values of the passband. Can be
        - 'nearest' : nearest neighbor
        - 'linear'  : linear
        - 'quadratic' : piecewize quadratic
    interpolate_set: boolean
        True is the interpolation function is set
    extrapolate: str
        'yes': the curves can be extrapolated
        'no': the curves cannot be extrapolated, if an extrapolation is
        requested, the extrapolated values will be set to NaN
        'zero' : extrapolated values are set to zero
    positive: bool
        If True, any interpolated value that would lead to a negative result
        is set to zero
    interpolate: PassbandInterpolator
        Returns the inrterpolated values of y at requested x.
    integration_method: str
        the method used for numerical integration. Can be:
        - 'trapezoidal'
        - 'simpson'
    integrate_set: boolean
        True if the integration method is set.
    integrate: function
        The function that performs the integration
    yfactor: float
        scaling factor applied to y
    xshift: float
        scaling factor applied to x
    initialized: bool
        True if properly initialized

    Attributes specific to Passband
    --------------------------------
    instrument: str
        The name of the instrument the passband applies to
    filter: str
        The name of the filter
    org_y_type: str
        The original type of the passband. Can be 'qe' or 'rsr'
    org_x_ref: float
        The original reference
    x_ref: float
        The reference frequency or wavelength for the passband in
        units of x_si_unit.
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
    xoffset: float
        offset applied to x



    Methods:
    --------
    set_location(location):
        defines the rules for interpolating spectra aand passband.
    response():
        returns the passband y values
    in_nu():
        switch the passband in function of frequency
    in_lam():
        switch the passband in function of wavelength
    fnu_ab(spectrum):
        compute the spectral irradiance in the band of the spectrum, following
        the AB convention.
    mag_ab(spectrum):
        compute the AB magnitude of the spectrum in the band
    mag_vega(spectrum[,vegaflux, vega]):
        compute the vega magnitude of the spectrum in the band
    fnu_ir(spectrum):
        compute the spectral irradiance in the band of the spectrum, following
        the IR convention.
    flux(spectrum):
        computes the inband irradiance (W/m**2) of the spectrum in the band.
    flam_st(spectrum):
        computes the spectral irradiance in a passband of a spectrum, following
        the ST convention
    bandwidth(unit[,normalize]):
        computes the bandwidth of the passband in unit
    xref(unit):
        returns the passband reference wavelength or frequency in unit
    combine(other):
        combines two passbands in one
    default_dir():
        returns the default directory of the photometry package where the
        passband file is to be found
    write(xunit[,dir, overwrite, force]):
        write the passband as a photometry file
    psf_weight(spectrum, xarr):
        computes weigths for quantities computed at xarr
    offset(offset):
        offset the passband
    unoffset():
        take out passband offset
    distort(alpha):
        Distort a passband by multiplying it by (x/x_ref)**alpha
    """
    # Initialization Methods
    def __init__(self, file=None, x=None, y=None, xref=None, ytype=None,
                 table=None, colname_x=None, colnames_y=None,
                 header=None, location='both_spectrum_and_passband',
                 interpolation='quadratic', extrapolate='zero',
                 integration='trapezoidal', x_unit=None,
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
        header: PhotometryHeader or dict or array of str
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

        """
        super().__init__(file=file, x=x, y=y, table=table, colname_x=colname_x,
                         colnames_y=colnames_y, header=header, x_unit=x_unit,
                         interpolation=interpolation,
                         extrapolate=extrapolate, positive=True,
                         integration=integration)


        self.instrument = None
        self.filter = None
        self.x_ref = None
        self.org_y_type = None
        self.org_x_ref = None
        self.is_qe = None
        self.is_rsr = None
        self.location_method  = None
        self.location_method = None
        self.nowarn = nowarn
        self.xoffset = 0.


        if not self.initialized:
            return

        if self.nb != 1:
            raise NotImplementedError('Cannot have multiple passbands')



        # Fill ytype
        if ytype is None:
            if 'ytype' not in self.header:
                raise ValueError('`ytype` is undefined for Passband')
            else:
                ytype = self.header['ytype']
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

        # Fill xref
        if xref is None:
            if 'xref' not in self.header:
                raise ValueError('Undefined xref')
            else:
                xref = u.Quantity(self.header['xref'])
                if xref.unit is u.dimensionless_unscaled:
                    xref = xref * self.org_x_unit

        msg = quantity_scalar(xref)
        if msg is not None:
            raise ValueError("`xref`" + msg)

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
                self.x_ref = (velc / xref.to(nu_unit)).value
        else:
            raise ValueError('`xref` unit {}'.format(xref.unit) +
                             'must be a wavelength or a frequency unit')

        self.set_location(location)

        self.fill_header()

        if self.initialized and self.location_set and self.interpolate_set and \
                self.integrate_set:
            self.ready = True
        else:
            self.ready = False

    def fill_header(self):
        """
        check self consistency between some attributes and the passband header
        and add them to the header in order to facilitate writing the passband
        to file if needed.
        """
        if 'file' not in self.header:
            if self.file is not None:
                self.header.add_card_value('file', self.file)
            else:
                if not self.nowarn:
                    print("Warning ! file needs to be set")
        else:
            if self.file is None:
                self.file = self.header['file']
            else:
                if self.file != self.header.content['file']:
                    if not self.nowarn:
                        print("Warning file does not match between header: "
                              "{}\nobject: {}".
                              format(self.header.content['file'],
                                     self.file))

        if 'instrument' not in self.header:
            if self.instrument is not None:
                self.header.add_card_value('instrument', self.instrument)
            else:
                if not self.nowarn:
                    print("Warning ! instrument needs to be set")
        else:
            if self.instrument is None:
                self.instrument = self.header['instrument']
            else:
                if self.instrument != self.header['instrument']:
                    if not self.nowarn:
                        print("Warning instrument does not match between\n"
                              "header: {}\nobject: {}"
                          .format(self.header['instrument'],
                                  self.instrument))

        if 'filter' not in self.header:
            if self.filter is not None:
                self.header.add_card_value('filter', self.filter)
            else:
                if not self.nowarn:
                    print("Warning ! filter needs to be set")
        else:
            if self.filter is None:
                self.filter = self.header['filter']
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
                if self.org_x_type != self.header['xtype']:
                    # TODO get rid of this annoying warning with files where
                    # wavelength is indicated with lam
                    if self.nowarn:
                        print("Warning xtype does not match between\n"
                              "header: {}\nobject: {}"
                            .format(self.header['xtype'],
                                    self.org_x_type))

        if 'ytype' not in self.header:
            if self.org_y_type is not None:
                self.header.add_card_value('ytype', self.org_y_type)
            else:
                raise ValueError("org_y_type is not set !")
        else:
            if self.org_y_type is None:
                raise ValueError("org_y_type is not set !")
            else:
                if self.org_y_type != self.header['ytype']:
                    if self.nowarn:
                        print("Warning ytype does not match between\n"
                              "header: {}\nobject: {}"
                            .format(self.header['ytype'],
                                    self.org_y_type))

        if 'xref' not in self.header:
            if self.org_x_ref is not None:
                self.header.add_card_value('xref', self.org_x_ref)
            else:
                raise ValueError("org_x_ref is not set !")
        else:
            if self.org_x_ref is None:
                raise ValueError("org_x_ref is not set !")
            else:
                if self.org_x_ref != u.Quantity(self.header['xref']):
                    if self.nowarn:
                        print("Warning xref does not match between\n"
                              "header: {}\nobject: {}"
                            .format(self.header['xref'],
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

    def response(self):
        """
        returns the passband y values
        """
        return self.y


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
        super().in_nu(reinterpolate=True)
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
        super().in_lam(reinterpolate=True)
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

    def mag_vega(self, spectrum, vega=None, vegaflux=None):
        """
        Compute the vega magnitude of the spectrum in the band

        Parameters
        ----------
        spectrum: BasicSpectrum
            spectrum for which is to be computed
        vegaflux: astropy.units.Quantity
            in band flux of vega
        vega: BasicSpectrum
            spectrum of vega. If fluxvega is provided, it is ignored

        Returns
        -------
        output: astropy.quantity
            The vega magnitude. This quantity has no unit.
        """
        fs = self.flux(spectrum)
        if vegaflux is None:
            if vega is not None:
                vegaflux = self.flux(vega)
            else:
                raise ValueError('Neither Vega flux or Vega Spectrum was '
                                 'provided')
        return -2.5 * np.log10(fs/vegaflux)

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
                          xref=self.x_ref(self.x_si_unit),  y=ally,
                          ytype=restype, nowarn=True)
        return result

    def default_dir(self):
        """
        Returns the default directory where passband files are located
        """
        if 'instrument' not in self.header:
            raise ValueError('instrument is not set in header')
        instrument = self.header['instrument']
        classpath = inspect.getfile(self.__class__)
        basepath = os.path.dirname(classpath)
        default_passband_dir = os.path.join(basepath, 'data/passbands/')
        return os.path.join(default_passband_dir, instrument + '/')


    def write(self, xunit, dir=None, overwrite=False, force=False,
              xfmt=':> 0.6f', yfmt=':>0.6f'):
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
        xfmt: str
            format for the x values
        yfmt: str
            format for the y values

        Returns
        -------
        output: str
            The full filename (including path) of the file that has been
            written

        Note: The file need some manual editing to put back the header lines
        in a nice order. An intelligent format to accomodate the precision
        would be nice.
        """

        # build the default file name
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

        return write_photometry_file(self, xunit, dirname=dir,
                                     filename=self.header.content['file'],
                                     overwrite=overwrite, xfmt=xfmt, yfmt=yfmt)

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

    def offset(self, offset):
        """
        Shift the passband (and not the atmosphere), by adding the offset to
        the frequencies

        Parameters:
        -----------
        offset: astropy.unit.Quantity

        """
        self.x = self.x + offset.to(self.x_si_unit).value
        self.xoffset = self.xoffset + offset
        self.set_interpolation(self.interpolate_method,
                               extrapolate=self.extrapolate,
                               positive=self.positive)

    def unoffset(self):
        """
        Unshift a passband
        """
        offset = -self.xoffset
        self.offset_passband(offset)

    def distort(self, alpha):
        """
        Distort a passband by multiplying it by (x/x_ref)**alpha

        Parameter:
        alpha: float
        """
        self.y = self.y * (self.x / self.x_ref)**alpha
        self.y = self.y / np.max(self.y)
        self.set_interpolation(self.interpolate_method,
                               extrapolate=self.extrapolate,
                               positive=self.positive)

    def rsr(self):
        if self.is_rsr:
            result = self.y / np.max(self.y)
        else:
            result = self.y * self.lam() / np.max(self.y * self.lam())
        return result

    def qe(self):
        if self.is_qe:
            result = self.y / np.max(self.y)
        else:
            result = self.y / self.lam() / np.max(self.y / self.lam())
        return result


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
    yvals = np.array([1., 1.])
    result = Passband(x=xlims, y=yvals, xref=0.5*np.sum(xlims), ytype=ytype,
                      interpolation='nearest', extrapolate='zero', nowarn=True)
    return result

__author__ = 'haussel'
from .atmosphere import IramAtmosphere
from .spectrum import BasicSpectrum
from .passband import Passband
from .phottools import PhotometryInterpolator
from astropy.coordinates import Angle
from .config import DEBUG




class IramPassband(Passband,IramAtmosphere):
    """
    A class to add atmosphere transparency models into passbands.
    It inherits both the Passbands and IramAtmosphere classes. As far as 
    integrating spectra is concerned, it is used as a normal passband class
    """
    def __init__(self, file=None, x=None, y=None, xref=None, ytype=None,
                 header=None, location='both_spectrum_and_passband',
                 interpolation='quadratic', integration='trapezoidal',
                 model=2009, observatory='iram30m', profile='midlatwinter',
                 gildas_atm_file=None):
        """
        Parameters:
        -----------

        Passband parameters:
        --------------------
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

        Atmosphere parameters:
        --------------------
        model: int
            year of the model used in GILDAS. Can be 1985 or 2009. Defaulted
            to 2009.
        observatory: str
            observatory name for the computations, normally 'iram30' or
            'iramPdb'. Defaulted to 'iram30m'
        profile: str
            the profile used in ATMOSPHERE. Not used in 1985 models. Defaulted
            to 'midlatwinter'
        gildas_atm_file: str
            the filename of the ATMOSPHERE output file to be used.              

        Attributes:
        -----------
        See Passband and Atmosphere docstrings for their 
        respectives Attributes.        

        New attributes:
        ---------------
        nu_atm: numpy.ndarray
            frequencies of the atmosphere model in SI unit (NOT a Quantity)
        x_org: numpy.ndarray 
            original x values of the passband
        y_org: numpy.ndarray 
            original y values of the passband
        interp_pb_org: PassbandInterpolator 
            To interpolate the orginal passband
        elevation: astropy.coordinates.Angle
            current elevation
        atm_trans: numpy.ndarray
            atmospheric transmission resampled at x values.
        """
        IramAtmosphere.__init__(self,model=model, observatory=observatory,
                                profile=profile,
                                gildas_atm_file=gildas_atm_file)
        Passband.__init__(self, file=file, x=x, y=y, xref=xref, ytype=ytype,
                          header=header,
                          location=location, interpolation=interpolation,
                          integration=integration)

        # This should not happen, but who knows ?
        if self.is_lam:
            self.in_nu()
        # At this point we have initialized separately both superclasses
        # IramAtimosphere.frequencies are Quantities.
        self.nu_atm = self.frequencies.to(self.x_si_unit).value
        self.x_org = self.x.copy()
        self.y_org = self.y.copy()
        # this allows to tnterpolate the passband alone
        self.interp_pb_org = PhotometryInterpolator(self.x_org, self.y_org,
                                                    kind='quadratic')
        self.elevation = None
        self.atm_trans = None


    def __str__(self):
        result = "**** Atmosphere: \n" + IramAtmosphere.__str__(self)
        result = result + "**** Passband: \n" + Passband.__str__(self)
        result = result + "\n**** IramPassband: \nelevation ={}\n".format(
            self.elevation)

        return result


    def _set_internals(self):
        if DEBUG:
                print('IramAtmosphere.set_internals()')
        if self.is_lam:
            self.in_nu()

        # This is the heart of things.
        # First, we compute the atmospheric transmission at elevation
        if self.elevation is not None:
            trans = self.transmission(self.elevation)
            # Then we interpolate this transmission at the common location of
            # points between the passband and the model.
            interp = PhotometryInterpolator(self.nu_atm, trans,
                                            kind=self.interpolate_method)
            self.x = self.location(self.nu_atm, self.x_org)
            # here we interpolate the atmospheric transmission
            self.atm_trans = interp(self.x)
            # here we interpolate the passband, and multiply by the transmission
            self.y = self.interp_pb_org(self.x) * self.atm_trans
            # we now have to recompute the interpolation coef for the new
            # passband taking into account the new x and y
            self.set_interpolation(self.interpolate_method)

    def set_elevation(self, elevation):
        """
        Set the elevation for the passband computation

        Parameters:
        -----------
        elevation: astropy.coordinates.Angle
            The elevation of the sources
        
        Effects:
        --------
        the passband x and y are recomputed, and interpolation coef are 
        updated. 
        """
        if isinstance(elevation, Angle):
            self.elevation = elevation
            self._set_internals()
        else:
            raise ValueError("input elevation is not an Angle")

    def shift_passband(self, shift_factor):
        """
        Shift the passband (and not the atmosphere), by multiplying the
        frequency array by shift_factor.

        Parameters:
        -----------
        shift_factor: float

        """
        self.x_org  = self.x_org * shift_factor
        self.interp_pb_org = PhotometryInterpolator(self.x_org, self.y_org,
                                                    kind='quadratic')
        self._set_internals()
        try:
            self.shift = self.shift * shift_factor
        except:
            self.shift = shift_factor

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

import numpy as np
import os
import inspect
from astropy import units as u
import re
from astropy.table import Table
from astropy.time import Time
from astropy import constants as const
from .spectrum import BasicSpectrum


__author__ = 'haussel'

"""
This module provide tools to compute planet fluxes in the (sub-)mm range. 
For now, only Uranus and Neptune are handled. 
"""


tar = re.compile('^Target body name: (\w+)')
soe = re.compile('^\$\$SOE')
eoe = re.compile('^\$\$EOE')
rad = re.compile('^Target radii\s+: (\d+.\d+) x \d+.\d+ x (\d+.\d+) km')


class GiantPlanet:
    """
    A class to compute giant planets (e.g. Neptune and Uranus) spectra.
     
    Both these planets were primary calibrators for the Herschel mission. 
    The spectrum of these planets have been modelled for the calibration. 
    The model spectra can be obtained from ESA: 
    https://www.cosmos.esa.int/web/herschel/calibrator-models 
    These models tabulate the brightness temperature of the planet 
    
    The other ingredient to compute the spectral irradiance at the observer 
    position is the solid angle under which the planetary disk is seen. In 
    order to compute this, we use tabulated output from JPL Horizons, 
    especially the Observer sub-lon & sub-lat (14) and Observer range & 
    range-rate (20).  
    
    We have: 
    .. math::
       S_{\nu} = \Omega \times \frac{ 2 h \nu^{3}}{c^{2}} 
       \frac{1}{e^{\frac{h \nu}{k T_{b}}} -1}
    
    Or alternatively using the Rayleigh-Jeans Temperature:
    .. math::
        S_{\nu} = \Omega \times \frac{2  \nu^{2} k T_{RJ}}{c^{2}}
        
    The solid angle is:

    .. math::
        \Omega = \pi \left(\frac{r_{gm}}{d}\right)^{2}

    Computed with the geometrical radius r_gm : it is the geometrical mean of 
    the equatorial radius r_eq and the apparent polar radius r_p_a:
    
    .. math::
        r_{gm} = \sqrt{r_{eq} r_{p-a}}

    where 
    .. math::
        r_{p-a} = r_{eq} \sqrt{1 - e^{2} cos^{2} \phi}
        
    where phi is the observer sub latitude and e is the body excentricty.
    .. math::
        e = \sqrt{\frac{r_{eq}^{2} - r_{p}^{2}}{r_{eq}^2}}
    
    
    """
    def __init__(self, planet, model_version='esa4', location='Iram30m',
                 datadir=None):
        """
        Initialisation. The planet JPL horizons ephemeris are read, as well as 
        the ESA model. 
        
        Parameters
        ----------
        planet: str
            The name of the planet
        model_version: str
            ESA model version. Defaulted to esa4. 
        location: str
            Location for which the ephemeris was computed. Defaulted to 
            'Iram30m'
        datadir: str
            Directory where the ephemeris and model can be found. 
        
        Attributes
        ----------
        name: str
            name of the planet
        ephemeris_filename: str
            filename of ephemeris
        model_filename: str
            filename of the model used
        r_eq: float
            Equatorial radius of the planet in km
        r_p: float
            Polar radius of the planet in km
        e: float
            Ellipticity of the planet body
        ephemeris: astropy.table.Table
            The planet tabulated ephemeris
        model: astropy.table.Table
            The planet atmosphere tabulated model
        dateset: bool
            test whether time has been set
        dates: astropy.time.Time
            times for which computations are done
        idxl: numpy.ndarray
            indices for interpolation
        idxh: numpy.ndarray
            indices for interpolation
        alpha: numpy.ndarray
            indices for interpolation
        """

        if datadir is None:
            classpath = inspect.getfile(self.__class__)
            basepath = os.path.dirname(classpath)
            default_planet_dir = os.path.join(basepath,
                                              'data/planets/{}/'.format(planet))
        else:
            default_planet_dir = datadir

        filename = os.path.join(default_planet_dir,
                                '{}_Horizons_2010_2025_{}.txt'.format(planet,
                                                                      location))

        if not os.path.exists(filename):
            raise ValueError("Cannot find file {} for planet {}".format(filename, planet))
        self.ephemeris_filename = filename
        f = open(filename, 'r')
        # read the horizon ephemeris
        lines = f.readlines()
        # this is a header we will use for table creation
        ephem_lines = ['JulDate, Sun, Moon, RA, Dec, Az, El, Ilum, AngDiam, '
                       'SubObsLon, SubObsLat, Range, DeltaRange,']
        target_found = False
        radii_found = False
        in_ephem = False
        ephem_done = False
        for line in lines:
            if not target_found:
                test = tar.match(line)
                if test:
                    self.name = test.group(1)
                    # Sanity check that we are using the same planet
                    if self.name != planet:
                        raise ValueError("Target body in file ({}) does not "
                                         "match requested planet {}".format(
                            self.name, planet))
                    target_found = True
            elif not radii_found:
                test = rad.match(line)
                if test:
                    # set the equatorial and polar radii
                    self.r_eq = float(test.group(1)) * u.km
                    self.r_p = float(test.group(2)) * u.km
                    radii_found = True
            elif not ephem_done:
                if not in_ephem:
                    if soe.match(line):
                        in_ephem = True
                else:
                    if eoe.match(line):
                        in_ephem = False
                        ephem_done = True
                    else:
                        ephem_lines.append(line)
            else:
                pass
        if not target_found:
            raise ValueError("Unable to find planet name in file {}".format(filename))
        if not radii_found:
            raise ValueError("Unable to find planet radii in file {}".format(filename))
        if not ephem_done:
            raise ValueError("Unable to find ephemeris in file {}".format(filename))
        self.ephemeris = Table.read(ephem_lines, format='csv')
        # Compute body ellipticity
        self.e = np.sqrt((self.r_eq ** 2 - self.r_p ** 2) / self.r_eq ** 2)

        # read the spectrum model
        model_filename =  self.name + '_' + model_version + '.fits'
        model_fullpath = os.path.join(default_planet_dir, model_filename)
        if not os.path.exists(model_fullpath):
            raise ValueError("Cannot find model file {}"
                             " for planet {}".format(model_fullpath, planet))
        self.model_filename = model_fullpath
        self.model = Table.read(model_fullpath)
        # We add a luminosity column to the model
        hnukt = const.h * self.model['wave'].to(u.Hz) / \
                (const.k_B * self.model['T_b'])
        self.model['Inu'] = 2 * const.h.value * \
                            (self.model['wave'].to(u.Hz).value) ** 3 / \
                            const.c.value ** 2 / (np.exp(hnukt) - 1)

        self.dateset = False
        self.dates = None
        self.idxl = None
        self.idxh = None
        self.alpha = None


    def set_dates(self, dates):
        """
        Compute the element to interpolate for give dates
        
        Parameters:
        -----------
        
        dates: astropy.time.Time

        Return: True
        -------
        
        Update
        -------
        self.idxl
        self.idxh
        self.alpha
        self.dateset
        """
        if not isinstance(dates, Time):
            raise ValueError("date is not an astropy.time")
        self.idxh = np.digitize(dates.jd, self.ephemeris['JulDate'])
        self.idxl = self.idxh-1
        iextrap, = np.where(self.idxl < 0)
        if len(iextrap) > 0:
            self.idxl[iextrap] = 0
            self.idxh[iextrap] = self.idxl[iextrap] + 1
        self.alpha = (dates.jd - self.ephemeris['JulDate'][self.idxl]) / \
                (self.ephemeris['JulDate'][self.idxh] -
                 self.ephemeris['JulDate'][self.idxl])
        self.dates = dates
        self.dateset = True
        return True

    def check_dates(self, dates):
        """
        Return True if the input dates corresponds to the internal dates

        Parameters:
        -----------

        dates: astropy.time.Time
            dates to check
        """""
        if not isinstance(dates, Time):
            raise ValueError("date is not an astropy.time")

        return np.any(dates.jd == self.dates.jd)


    def apparent_polar_radius(self):
        """
        Compute the apparent polar radius at a set of dates.
         
        Returns:
        --------
        astropy.unit.quantity
            apparent polar radius in km.
        """
        if not self.dateset:
            raise ValueError("Date for compution is not set")

        phis = ((self.ephemeris['SubObsLat'][self.idxh] -
                 self.ephemeris['SubObsLat'][self.idxl]) * self.alpha +
                self.ephemeris['SubObsLat'][self.idxl]) * u.deg

        # apparent polar radius
        r_p_a = self.r_eq * np.sqrt(1 - (self.e * np.cos(phis)) ** 2)
        return r_p_a

    def distance_to_observer(self):
        """
        Compute the planet distance for at a set of dates.

        Returns
        -------
        astropy.units.Quantity
            The distance to observer in AU
        """
        if not self.dateset:
            raise ValueError("Date for compution is not set")

        distances = ((self.ephemeris['Range'][self.idxh] -
                      self.ephemeris['Range'][self.idxl]) * self.alpha +
                     self.ephemeris['Range'][self.idxl]) * u.AU
        return distances

    def solid_angle(self):
        """
        Compute the solid angle subtended by the planet at given dates. 

        Returns
        -------
        
        astropy.units.Quantity
            The solid angle in sr.
        """
        if not self.dateset:
            raise ValueError("Date for compution is not set")

        distances = self.distance_to_observer()
        r_p_a = self.apparent_polar_radius()
        r_gm = np.sqrt(self.r_eq * r_p_a)
        ang = np.arctan2(r_gm.to(u.km), distances.to(u.km))
        omega = np.pi * ang.to(u.rad)**2
        return omega

    def spectral_irradiance(self, solid_angle=None, dates=None):
        """
        Computes the spectral irradiance of the planet for given solid angles or
        dates. 
        
        Parameters:
        -----------
        solid_angle: astropy.unit.Quantity
            the solid angle of the planet
        dates: astropy.time.Time
            the dates 

        Return:
        ------
        photometry.spectrum.BasicSpectrum
            The spectras of the planet
        """
        if solid_angle is None:
            if dates is not None:
                self.set_dates(dates)

            if not self.dateset:
                raise ValueError("Date for compution is not set")
            solid_angle = self.solid_angle()
        else:
            if dates is not None:
                if not self.check_dates(dates):
                    raise ValueError("Inconsistent dates and solid angles")
        if solid_angle.isscalar:
            result = BasicSpectrum(x=self.model['wave'].quantity,
                                   y=(self.model['Inu'] *
                                      solid_angle.to(u.sr).value * 1e26) * u.Jy)
        else:
            result = BasicSpectrum(x=self.model['wave'].quantity,
                                   y=(self.model['Inu'].data[np.newaxis, :] *
                                      solid_angle.to(u.sr).value[:, np.newaxis]
                                      * 1e26)  * u.Jy)
        return result

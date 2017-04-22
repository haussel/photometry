__author__ = 'haussel'

"""
This module provide tools o compute planet fluxes in the submm. 
For now, only Uranus and Neptune are handled. 
"""

import numpy as np
import os
import inspect
from astropy import units as u
import re
from astropy.table import Table
from astropy.time import Time
from astropy import constants as const
from .spectrum import BasicSpectrum


tar = re.compile('^Target body name: (\w+)')
soe = re.compile('^\$\$SOE')
eoe = re.compile('^\$\$EOE')
rad = re.compile('^Target radii\s+: (\d+.\d+) x \d+.\d+ x (\d+.\d+) km')

class GiantPlanet:
    """
    A class to compute planets spectra. 
    """
    def __init__(self, planet, model_version='v4'):
        """
        Initialisation. The planet JPL horizons ephemeris are read, as well as the ESA model. 
        
        :param planet: The name of the planet
        
        Attributes
        ----------
        name: str
            name of the planet
        filename: str
            filename of ephemeris 
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
        """

        classpath = inspect.getfile(self.__class__)
        basepath = os.path.dirname(classpath)
        default_planet_dir = os.path.join(basepath, 'data/planets/{}/'.format(planet))

        filename = os.path.join(default_planet_dir, '{}_horizons_2010_2025.txt'.format(planet))

        if not os.path.exists(filename):
            raise ValueError("Cannot find file {} for planet {}".format(filename, planet))
        self.filename = filename
        f = open(filename, 'r')
        # read the horizon ephemeris
        lines = f.readlines()
        ephem_lines = ['JulDate, Sun, Moon, RA, Dec, Az, El, Ilum, AngDiam, SubObsLon, SubObsLat, Range, DeltaRange,']
        target_found = False
        radii_found = False
        in_ephem = False
        ephem_done = False
        for line in lines:
            if not target_found:
                test = tar.match(line)
                if test:
                    self.name = test.group(1)
                    if self.name != planet:
                        raise ValueError("Target body in file ({}) does not match requested planet {}".
                                         format(self.name, planet))
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
        self.e = np.sqrt((self.r_eq ** 2 - self.r_p ** 2) / self.r_p ** 2)

        # read the spectrum model
        model_filename =  self.name + '_' + model_version + '.fits'
        model_fullpath = os.path.join(default_planet_dir, model_filename)
        self.model = Table.read(model_fullpath)
        hnukt = const.h * self.model['wave'].to(u.Hz) / (const.k_B * self.model['T_b'])
        self.model['Inu'] = 2 * const.h.value * (self.model['wave'].to(u.Hz).value) ** 3 / \
                            const.c.value ** 2 / (np.exp(hnukt) - 1)


    def linear_interpolation(self, dates):
        """
        Compute the element to interpolate for give dates
        :param dates: astropy.time.Time
        :return: tuple (idxl, idxh, alpha)
            idxl: index of lower value
            idxh: index of higher values
            alpha: position between values
        """
        if not isinstance(dates, Time):
            raise ValueError("date is not an astropy.time")
        idxh = np.digitize(dates.jd, self.ephemeris['JulDate'])
        idxl = idxh-1
        iextrap, = np.where(idxl < 0)
        if len(iextrap) > 0:
            idxl[iextrap] = 0
            idxh[iextrap] = idxl[iextrap] + 1
        alpha = (dates.jd - self.ephemeris['JulDate'][idxl]) / \
                (self.ephemeris['JulDate'][idxh] - self.ephemeris['JulDate'][idxl])
        return (idxl, idxh, alpha)

    def apparent_polar_radius(self, dates, idxl=None, idxh=None, alpha=None):
        """
        Compute the apparent polar radius at a set of dates. 
        :param dates: astropy.time.Time 
        :param idxl: numpy.ndarray
            index of lower bin for interpolation
        :param idxh:  numpy.ndarray
            index of upper bin for interpolation
        :param alpha: numpy.ndarray
            normalized distance to lower bin         
        :return: astropy.units.Quantity
            The apparent polar radius
        """
        if idxl is None or idxh is None or alpha is None:
            idxl, idxh, alpha = self.linear_interpolation(dates)

        phis = ((self.ephemeris['SubObsLat'][idxh]-self.ephemeris['SubObsLat'][idxl]) * alpha + \
                    self.ephemeris['SubObsLat'][idxl]) * u.deg

        # apparent polar radius
        r_p_a = self.r_eq * np.sqrt(1 - (self.e * np.cos(phis)) ** 2)
        return r_p_a

    def distance_to_observer(self, dates, idxl=None, idxh=None, alpha=None):
        """
        Compute the planet distance for at a set of dates.
        :param dates: astropy.time.Time 
        :param idxl: numpy.ndarray
            index of lower bin for interpolation
        :param idxh:  numpy.ndarray
            index of upper bin for interpolation
        :param alpha: numpy.ndarray
            normalized distance to lower bin         
        :return: astropy.units.Quantity
            The distance to observer in AU
        """
        if idxl is None or idxh is None or alpha is None:
            idxl, idxh, alpha = self.linear_interpolation(dates)

        distances = ((self.ephemeris['Range'][idxh]-self.ephemeris['Range'][idxl]) * alpha + \
                        self.ephemeris['Range'][idxl]) * u.AU
        return distances

    def solid_angle(self, dates, idxl=None, idxh=None, alpha=None):
        """
        Compute the solid angle subtended by the planet at given dates
        :param dates: astropy.time.Time 
        :param idxl: numpy.ndarray
            index of lower bin for interpolation
        :param idxh:  numpy.ndarray
            index of upper bin for interpolation
        :param alpha: numpy.ndarray
            normalized distance to lower bin 
        :return:  astropy.units.Quantity
            The solid angle in sr.
        """
        if idxl is None or idxh is None or alpha is None:
            idxl, idxh, alpha = self.linear_interpolation(dates)

        distances = self.distance_to_observer(dates,idxl=idxl, idxh=idxh, alpha=alpha)
        r_p_a = self.apparent_polar_radius(dates,idxl=idxl, idxh=idxh, alpha=alpha)
        r_gm = np.sqrt(self.r_eq * r_p_a)
        ang = np.arctan2(r_gm.to(u.km), distances.to(u.km))
        omega = np.pi * ang.to(u.rad)**2
        return omega

    def spectrum(self, solid_angle=None, dates=None):
        """
        Computes the spectrum of the planet
        :param solid_angle: astropy.unit.Quantity
            the solid angle of the planet
        :param dates: astropy.time.Time
            the dates 
        :return: photometry.spectrum.BasicSpectrum
            The spectrum of the planet
        """
        if solid_angle is None:
            if dates is not None:
                solid_angle = self.solid_angle(dates)
            else:
                raise ValueError("Either dates or solid angles must be give")
        else:
            if dates is not None:
                raise ValueError("Either dates or solid angles must be give")
        result = BasicSpectrum(x=self.model['wave'].quantity,
                               y=(self.model['Inu']*solid_angle.to(u.sr).value * 1e26) * u.Jy)
        return result

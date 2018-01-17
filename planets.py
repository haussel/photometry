import numpy as np
import os
import inspect
from astropy import units as u
import re
from astropy.table import Table, Column
from astropy.time import Time
from astropy import constants as const
from .spectrum import BasicSpectrum
import sqlite3
import requests
from bs4 import BeautifulSoup

__author__ = 'haussel'

"""
This module provide tools to compute planet fluxes in the (sub-)mm range. 
For now, only Uranus and Neptune are handled. 
"""


tar = re.compile('^Target body name: (\w+)')
soe = re.compile('^\$\$SOE')
eoe = re.compile('^\$\$EOE')
rad = re.compile('^Target radii\s+: (\d+.\d+) x \d+.\d+ x (\d+.\d+) km')

URL_MARS_MODEL = "http://www.lesia.obspm.fr/perso/emmanuel-lellouch/" \
                 "mars/tedate_v1.2_100pts.php"

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


class Mars:
    """
    A class to obtain Mars total fluxes by querying the Version 1.3 of the
    Mars brightness model of E.Lellouche and H. Amri. In order to avoid
    recomputing values for the same input parameters, values are stored in a
    local database.

    """
    def __init__(self, localdb=None, createdb=False, telescope='Iram30m',
                 datetime=Time('2017-01-15T12:00:00.0'), verbose=False,
                 instrument='NIKA2'):
        """
        Inititialization.

        Parameters
        ----------
        localdb: str
            Full path to local database. If not given, use the database file
            coming the the software distribution
        createdb: boolean
            If True, the local database table is created. Defaulted to False
        telescope: str
             Telescope for which the computations are made. Used to set the
             HPWH at 300 GHz for the model. Defaulted to 'Iram30m'
        instrument: str
            Instrument for which the computations are made. Used to set the
            frequencies of which the model is computed. Defaulted to 'NIKA2'
        datetime: astropy.time.Time
            Date and time for which the model compuatations are made.
            Defaulted to Time('2017-01-15T12:00:00.0')
        verbose: Boolean
            if True, a few messages are printed.

        Attributes
        ----------
        verbose: boolean
            if True, print a few messages
        localdb: str
            full path to the local DB file
        connection: sqlite3.Connection
            the SQLite database connection to local DB
        cursor: sqlite3.Cursor
            cursor into the local DB
        telescope: str
            telescope for model
        instrument: str
            instrument for model
        datetime: astropy.time.Time
            time for model
        beam: float
            HPBW at 300 GHZ
        roughness: float
            roughness for model
        penlen: float
            penetration length for model
        dielec: float
            dielectric constant for model
        freq: list of 4 float
            frequencies for model

        """
        self.verbose = verbose
        if localdb is None:
            classpath = inspect.getfile(self.__class__)
            basepath = os.path.dirname(classpath)
            default_planet_dir = os.path.join(basepath,
                                              'data/planets/Mars/')
            localdb = os.path.join(default_planet_dir, 'marsdata.db')
            if self.verbose:
                print("Working with database file {}".format(localdb))
        self.localdb = localdb
        if os.path.exists(localdb):
            try:
                self.connection = sqlite3.connect(localdb)
                self.cursor = self.connection.cursor()
            except:
                raise ValueError("Cannot open local db {}".format(localdb))
        else:
            if createdb:
                self.connection = sqlite3.connect(localdb)
                self.cursor = self.connection.cursor()
                self.cursor.execute('''CREATE TABLE marsmodel
                                    (date text, beam real, roughness real, 
                                     penlen real, dielec real, rapp real, 
                                     freq real, hpbw real, ff real,  flux real, 
                                     tb real, fmb real, mtmb real, 
                                     tmbrj real)''')
            else:
                raise ValueError("Cannot find local db for values {}".format(
                    localdb))
        # default values for query
        self.telescope = telescope
        self.instrument = instrument
        self.datetime = datetime
        self.beam = 0.
        self.set_beam(self.telescope)
        self.roughness = 12
        self.penlen = 12
        self.dielec = 2.25
        self.freq = [0., 0., 0., 0.]
        self.set_instrument(self.instrument)

    def __del__(self):
        """
        Ensure DB is properly closed when object is destroyed.
        """
        self.connection.close()

    def __str__(self):
        """
        Format model parameters to print
        """
        result = "-"*77
        result += "\nMars model Version 1.3 of E.Lellouche and H. Amri"
        result += "\nLocal values stored in {}".format(self.localdb)
        result += "\nTelecope: {}".format(self.telescope)
        result += "\nFor a HPBW at 300 GHz of {} arcsec".format(self.beam)
        result += "\nInstrument: {}".format(self.instrument)
        result += "\nModel frequencies: {}, {}, {}, {} GHz".\
            format(self.freq[0], self.freq[1],self.freq[2], self.freq[3])
        result += "\nRoughness: {}".format(self.roughness)
        result += "\nPenetration length: {}".format(self.penlen)
        result += "\nDielectric constant: {}\n".format(self.dielec)
        result += "-" * 77
        return result

    def set_beam(self, telescope):
        """
        Set the beam HPBW @ 300 GHz value in arcsec from telescope name

        Parameter
        ---------
        telescope: str
            Name of the telescope

        Returns
        -------
            beam in arcsec at 300 GHz.
        """
        if telescope == 'Iram30m':
            self.beam = 9.5
        else:
            raise ValueError("Unknown location: {}".format(telescope))
        return self.beam * u.arcsec

    def set_instrument(self, instrument):
        """
        Set the frequency for a given instrument

        Parameters
        ----------

        instrument: str
            Name of the instrument

        Returns
        -------
            list of 4 frequencies
        """
        if instrument == 'NIKA2':
            self.freq = [150., 260., 1600., 3200.]
        else:
            raise ValueError("Unknown instrument: {}".format(instrument))
        return self.freq * u.GHz

    def set_datetime(self, datetime):
        """
        Set the time, ensuring an ISOT format

        Parameter
        ---------
        datetime: astropy.time.Time or str
            date and time for model computations

        Returns
        -------
            time set, as an astropy.time.Time in ISOT format

        """
        if not isinstance(datetime, Time):
            try:
                thetime = Time(datetime)
            except:
                raise ValueError("Input cannot be converted to Time {"
                                 "}".format(datetime))
            thetime.format = 'isot'
            self.datetime = thetime
        else:
            thetime = datetime.copy(format='isot')
            self.datetime = thetime
        return self.datetime


    def dbdata2table(self, dbdata):
        """
        Put the results of a database query into an astropy.table.Table

        Parameters
        ----------
        dbdata tuple or list
            output of the database or web query.

        Returns
        -------
        an astropy.table.Table
        """
        if type(dbdata) == tuple:
            wdata = [dbdata]
            nbrow = 1
        else:
            wdata = dbdata
            nbrow = len(dbdata)
        t = Table()
        times = [row[0] for row in wdata]
        t['datetime'] = Column(Time(times))
        beams = [row[1] for row in wdata]
        t['hpbw@300GHz'] = Column(beams, unit=u.arcsec,
                                  description="Beam HPBW at 300 GHz")
        roughness = [row[2] for row in wdata]
        t['roughness'] = Column(roughness, unit=u.degree,
                                description='Roughness')
        penlen = [row[3] for row in wdata]
        t['penlen'] = Column(penlen, description='Penetration length in unit '
                                                 'of lambda')
        dielec = [row[4] for row in wdata]
        t['dielec']= Column(dielec, description='dielectric constant')
        
        rapp = [row[5] for row in wdata]
        t['Rapp'] = Column(rapp, unit=u.arcsec,
                           description='apparent diameter')
        freq = [row[6] for row in wdata]
        t['nu'] = Column(freq, unit=u.GHz, description='Frequency')
        hpbw = [row[7] for row in wdata]
        t['hpbw'] = Column(hpbw, unit=u.arcsec,
                           description="Half Power Bean Width")
        ff = [row[8] for row in wdata]
        t['fillfact'] = Column(ff, description="Filling factor")
        flux = [row[9] for row in wdata]
        t['fnu'] = Column(flux, unit=u.Jy, description='Total flux')
        tb = [row[10] for row in wdata]
        t['Tb'] = Column(tb, unit = u.K, description='Brightness temperature')
        fmb = [row[11] for row in wdata]
        t['Fmb'] = Column(fmb, unit=u.Jy, description='Flux in Main Beam')
        mtmb = [row[12] for row in wdata]
        t['AveTbMb'] = Column(mtmb, unit=u.K,
                              description='Average Tb over Main Beam')
        tmbrj = [row[13] for row in wdata]
        t['TmbRJ'] = Column(tmbrj, unit=u.K,
                            description = 'RJ temperature in Main Beam')
        return t


    def fnu(self, frequency, datetime):
        """
        Return Mars total flux for given frequency and date. If the values
        are not in the database, query the web model.

        Parameters
        ----------
        frequency: real or astropy.units.Quantity
            The frequency where to compute the model
        datetime: str or astropy.time.Time
            the time for the computations

        Returns
        -------
        """
        if isinstance(frequency, u.Quantity):
            freq = frequency.to(u.GHz).value
        else:
            freq = frequency

        self.set_datetime(datetime)

        myquery = "SELECT * FROM marsmodel WHERE date = ? AND beam = ? AND " \
                  "roughness = ? AND penlen = ? AND dielec = ? AND " \
                  "freq = ?"

        myvalues = (self.datetime.value, self.beam, self.roughness,
                    self.penlen, self.dielec, freq)

        dbdata = self.cursor.execute(myquery, myvalues).fetchall()

        if len(dbdata) == 0:
            imin = np.argmin(np.abs(freq-np.array(self.freq)))
            self.freq[imin] = freq
            result = self.queryweb()
        else:
            result = self.dbdata2table(dbdata)

        return result[result['nu'] == freq]['fnu'].data[0] * u.Jy

    def dumpdb(self):
        """
        Dump the local database content in a table

        Returns
        -------
        astropy.table.Table
        """
        return self.dbdata2table(self.cursor.execute('select * from '
                                               'marsmodel').fetchall())

    def queryweb(self):
        """
        Perform a model query to the web service

        Returns
        -------
        astropy.table.Table
            The model values. The internal database is updated
        """
        payload = {'jour': self.datetime.datetime.day,
                   'mois': self.datetime.datetime.strftime("%B"),
                   'annee': self.datetime.datetime.year,
                   'heur': self.datetime.jd + 0.5 - np.floor(
                       self.datetime.jd + 0.5),
                   'rough': self.roughness,
                   'taille': self.beam,
                   'ctedelectrique': self.dielec,
                   'longpenetration': self.penlen,
                   'freq1': self.freq[0],
                   'freq2': self.freq[1],
                   'freq3': self.freq[2],
                   'freq4': self.freq[3]}
        r = requests.post(URL_MARS_MODEL, data=payload)
        if r.status_code != 200:
            raise ValueError("Request to model failed")
        else:
            soup = BeautifulSoup(r.text, "html5lib")
            tables = soup.find_all('tbody')
            rowstab1 = tables[0].find_all('tr')
            radius = float((rowstab1[3].find_all('th'))[1].get_text())
            rowstab2 = tables[1].find_all('tr')
            hpbw1 = float((rowstab2[1].find_all('td'))[1].get_text())
            ff1 = float((rowstab2[1].find_all('td'))[2].get_text())
            flux1 = float((rowstab2[1].find_all('td'))[3].get_text())
            tb1 = float((rowstab2[1].find_all('td'))[4].get_text())
            fmb1 = float((rowstab2[1].find_all('td'))[5].get_text())
            mtmb1 = float((rowstab2[1].find_all('td'))[6].get_text())
            tmbrj1 = float((rowstab2[1].find_all('td'))[7].get_text())

            hpbw2 = float((rowstab2[2].find_all('td'))[1].get_text())
            ff2 = float((rowstab2[2].find_all('td'))[2].get_text())
            flux2 = float((rowstab2[2].find_all('td'))[3].get_text())
            tb2 = float((rowstab2[2].find_all('td'))[4].get_text())
            fmb2 = float((rowstab2[2].find_all('td'))[5].get_text())
            mtmb2 = float((rowstab2[2].find_all('td'))[6].get_text())
            tmbrj2 = float((rowstab2[2].find_all('td'))[7].get_text())

            hpbw3 = float((rowstab2[3].find_all('td'))[1].get_text())
            ff3 = float((rowstab2[3].find_all('td'))[2].get_text())
            flux3 = float((rowstab2[3].find_all('td'))[3].get_text())
            tb3 = float((rowstab2[3].find_all('td'))[4].get_text())
            fmb3 = float((rowstab2[3].find_all('td'))[5].get_text())
            mtmb3 = float((rowstab2[3].find_all('td'))[6].get_text())
            tmbrj3 = float((rowstab2[3].find_all('td'))[7].get_text())

            hpbw4 = float((rowstab2[4].find_all('td'))[1].get_text())
            ff4 = float((rowstab2[4].find_all('td'))[2].get_text())
            flux4 = float((rowstab2[4].find_all('td'))[3].get_text())
            tb4 = float((rowstab2[4].find_all('td'))[4].get_text())
            fmb4 = float((rowstab2[4].find_all('td'))[5].get_text())
            mtmb4 = float((rowstab2[4].find_all('td'))[6].get_text())
            tmbrj4 = float((rowstab2[4].find_all('td'))[7].get_text())

            result = [(self.datetime.value, self.beam, self.roughness,
                       self.penlen, self.dielec, radius, self.freq[0],
                       hpbw1, ff1, flux1, tb1, fmb1, mtmb1, tmbrj1),
                      (self.datetime.value, self.beam, self.roughness,
                       self.penlen, self.dielec, radius, self.freq[1],
                       hpbw2, ff2, flux2, tb2, fmb2, mtmb2, tmbrj2),
                      (self.datetime.value, self.beam, self.roughness,
                       self.penlen, self.dielec, radius, self.freq[2],
                       hpbw3, ff3, flux3, tb3, fmb3, mtmb3, tmbrj3),
                      (self.datetime.value, self.beam, self.roughness,
                       self.penlen, self.dielec, radius, self.freq[3],
                       hpbw4, ff4, flux4, tb4, fmb4, mtmb4, tmbrj4)]

            if self.verbose:
                print("Result obtained, updating internal db")
            sqlcmd = "INSERT INTO marsmodel VALUES (?, ?, ?, ?, ?, ?, ?," \
                     "?, ?, ?, ?, ?, ?, ?)"
            self.cursor.executemany(sqlcmd, result)
            self.connection.commit()
        return self.dbdata2table(result)



__author__ = 'haussel'
"""
This module provides the following classes:
- IramAtmosphere : a class to handle GILDAS ATMOSPHERE models
"""

import struct
import numpy
from astropy.coordinates import Angle
from astropy import units as u
import inspect
import os
from .spectrum import BasicSpectrum
from .config import DEBUG

class IramAtmosphere:
    """
    A class to compute opacities and transmissions from GILDAS ATMOSPHERE grid
    models.




    A typical session would go as this:
    from iramatmosphere import IramAtmosphere
    from astropy import units as u
    from astropy.coordinates import Angle
    from matplotlib import pyplot as plt
    atm = IramAtmosphere()
    atm.select_grid(pressure = 650. * u.hPa, temperature = 0. * u.deg_C)
    atm.set_tau_225(tau_225 = 0.1)
    plt.plot(atm.nu(), atm.transmission(Angle(45. * u.deg)))

    print(atm) allows to review the current status of the model
    """
    def __init__(self, model=2009, observatory='iram30m',
                 profile='midlatwinter', gildas_atm_file=None):
        """
        Initialization. The atmosphere model is read from a file that is the
        output of the GILDAS ATMOSPHERE program.

        The filename can be supplied directly, or is determined from the model,
        observatory and profile parameters. The latest is the default way.

        Parameters
        ----------
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

        model: int
            year of the model used in GILDAS. Can be 1985 or 2009
        observatory: str
            observatory name
        profile: str
            atrmosphere profile name used for computations
        filename: str
            name of the file that was read
        conditions_set: boolean
            True if all parameters have been set and atmosphere transmissions
            and emissions can be computed
        tau_225_set: boolean
            True if a value of opacity a 225 GHz or of water vapour content was
            provided
        tau_225: float
            value of the opacity at 225 GHz
        mm_H2O: astropy.units.quantity
            value of the water vapour content
        temperature: astropy.units.quantity
            value of the temperature
        pressure:  astropy.units.quantity
            value of the pressure
        nf: int
            Number of frequencies in the atmosphere grid
        np: int
            Number of pressures in the atmosphere grid
        nt: int
            Number of temperatures in the atmosphere grid
        frequencies: astropy.units.quantity
            array of frequencies where the opacities are tabulated
        pressures: astropy.units.quantity
            array of pressures where the opacities are tabulated
        temperatures:
            array of temperatures where the opacities are tabulated
        grid_tau_dry: (nf, nt, pt) ndarray
            grid of opacities for the dry component of the model
        grid_tau_wet: (nf, nt, pt) ndarray
            grid of opacities for the wet component of the model
        grid_selected: boolean
            True if a grid was selected
        idp: int
            index of the selected model on the pressure axis (3)
        idt: int
            index of the selected model on the temperature axis (2)
        tau: (nf, ) ndarray:
            current opacity model
        """
        self.model = None
        self.observatory = None
        self.profile = None
        self.filename = None
        self.conditions_set = False
        self.tau_225_set = False
        self.tau_225 = None
        self.mm_H2O = None
        self.temperature = None
        self.pressure = None
        self.nf = None
        self.np = None
        self.nt = None
        self.frequencies = None
        self.pressures = None
        self.temperatures = None
        self.grid_tau_dry = None
        self.grid_tau_wet = None
        self.grid_selected = False
        self.idp = None
        self.idt = None
        self.tau = None

        if gildas_atm_file is not None:
            self._read_atmosphere_file(gildas_atm_file)
            self.file = os.path.abspath(gildas_atm_file)
        else:
            if (model == 1985) or (model == 2009):
                self.model = model
            else:
                raise ValueError("invalid model value: {}".format(model))
            self.observatory = observatory
            self.profile = profile
            classpath = inspect.getfile(self.__class__)
            basepath = os.path.dirname(classpath)
            default_passband_dir = os.path.join(basepath,
                                                'data/atmosphere/iram/')
            if model != 1985:
                filename="atmosphere_{}_{}_{}.atm".format(self.model,
                                                          self.profile,
                                                          self.observatory )
            else:
                filename="atmosphere_{}_{}.atm".format(self.model,
                                                          self.observatory )
            full_filename = os.path.join(default_passband_dir, filename)
            self._read_atmosphere_file(full_filename)
            self.file = full_filename

    def _read_atmosphere_file(self, filename):
        """
        Read an atmosphere file
        """
        if not os.path.exists(filename):
            raise IOError("file not found '{}'".format(filename))
        with open(filename, 'rb') as f:
            data = f.read()
        f.close()
        (np, nt, nf, nw, na) = struct.unpack('iiiii', data[4:24])
        self.np = np
        self.nf = nf
        self.nt = nt
        pressures = numpy.empty(np)
        s = 24
        for i in numpy.arange(np):
            e = s + 4
            pressures[i] = struct.unpack('f', data[s:e])[0]
            s = e
        self.pressures = pressures * u.hPa
        temperatures = numpy.empty(nt)
        for i in numpy.arange(nt):
            e = s + 4
            temperatures[i] = struct.unpack('f', data[s:e])[0]
            s = e
        self.temperatures = temperatures * u.K
        frequencies = numpy.empty(nf)
        for i in numpy.arange(nf):
            e = s + 4
            frequencies[i] = struct.unpack('f', data[s:e])[0]
            s = e
        self.frequencies = frequencies * u.GHz
        water = numpy.empty(nw)
        for i in numpy.arange(nw):
            e = s + 4
            water[i] = struct.unpack('f', data[s:e])[0]
            s = e
        airmass = numpy.empty(na)
        for i in numpy.arange(na):
            e = s + 4
            airmass[i] = struct.unpack('f', data[s:e])[0]
            s = e

        tau1 = numpy.empty(nf * nt * np)
        for i in numpy.arange(nf * nt * np):
            e = s + 4
            tau1[i] = struct.unpack('f', data[s:e])[0]
            s = e

        tau2 = numpy.empty(nf * nt * np)
        for i in numpy.arange(nf * nt * np):
            e = s + 4
            tau2[i] = struct.unpack('f', data[s:e])[0]
            s = e

        self.grid_tau_dry = numpy.reshape(tau1, (nf, nt, np), order='F')
        self.grid_tau_wet = numpy.reshape(tau2, (nf, nt, np), order='F')
        return None

    def transmission(self, elevation, pressure=None, temperature=None,
                     tau_225 = None, mm_H2O = None):
        """
        Compute the atmospheric transmission at given elevation.

        The atmosphere model can be changed on the fly by providing new models
        parameters (pressure, temperature, opacity at 225 GHz or water vapour
        content). If none of these parameters are provided, the current model
        of atmosphere is used.

        Parameters
        ----------
        elevation: astropy.coordinates.Angle
            requested elevation. If this is the only parameter given, the
            current model is used.

        pressure: astropy.units.quantity
            pressure for which a new model is to be computed before determining
            the transmission at given elevation

         temperature: astropy.units.quantity
            temperature for which a new model is to be computed before
            determining the transmission at given elevation

        tau_225: float
            opacity at 225 GHz at which a new model is to be computed before
            determining the transmission at given elevation

        mm_H2O: astropy.units.quantity
            water vapour content at which a new model is to be computed before
            determining the transmission at given elevation


        Returns
        -------
        output: (nf,) ndarray
            array of transmissiom at frequencies self.frequencies
        """
        if pressure is not None or temperature is not None:
            self.select_grid(pressure=pressure, temperature=temperature)
            if tau_225 is None and mm_H2O is None:
                # the quantity that does not change is tau_225,
                # while mm_H2O will adjust depending on the conditions
                self.set_tau_225()
        if tau_225 is not None and mm_H2O is not None:
            raise ValueError('cant set both tau_225 and mm_H2O')
        if tau_225 is not None:
            self.set_tau_225(tau_225=tau_225)
        if mm_H2O is not None:
            self.set_mm_H2O(mm_H2O=mm_H2O)
        if not self.conditions_set:
            raise ValueError("conditions are not set")
        if not isinstance(elevation, Angle):
            raise ValueError('elevation must be an angle')
        secz = 1./numpy.sin(elevation)
        result = numpy.exp(-secz.value * self.tau)
        return result

    def emission(self, elevation, pressure=None, temperature=None,
                 tau_225 = None, mm_H2O = None):
        """
        Compute the atmospheric emission at given elevation.

        The returned emission is simply exp(secz * tau), where secz is the
        airmass. So this is not the true atmosphere emission, it is missing
        a normalization factor

        The atmosphere model can be changed on the fly by providing new models
        parameters (pressure, temperature, opacity at 225 GHz or water vapour
        content). If none of these parameters are provide, the current model of
        atmosphere is used.

        Parameters
        ----------
        elevation: astropy.coordinates.Angle
            requested elevation. If this is the only parameter given, the
            current model is used.

        pressure: astropy.units.quantity
            pressure for which a new model is to be computed before determining
            the transmission at given elevation

         temperature: astropy.units.quantity
            temperature for which a new model is to be computed before
            determining the transmission at given elevation

        tau_225: float
            opacity at 225 GHz at which a new model is to be computed before
            determining the transmission at given elevation

        mm_H2O: astropy.units.quantity
            water vapour content at which a new model is to be computed before
            determining the transmission at given elevation

        Returns
        -------
        output: (nf,) ndarray
            array of emission at frequencies self.frequencies (or self.nu())
        """
        if pressure is not None or temperature is not None:
            self.select_grid(pressure=pressure, temperature=temperature)
            if tau_225 is None and mm_H2O is None:
                # the quantity that does not change is tau_225,
                # while mm_H2O will adjust depending on the conditions
                self.set_tau_225()
        if tau_225 is not None and mm_H2O is not None:
            raise ValueError('cant set both tau_225 and mm_H2O')
        if tau_225 is not None:
            self.set_tau_225(tau_225=tau_225)
        if mm_H2O is not None:
            self.set_mm_H2O(mm_H2O=mm_H2O)
        if not self.conditions_set:
            raise ValueError("conditions are not set")
        if not isinstance(elevation, Angle):
            raise ValueError('elevation must be an angle')
        secz = 1./numpy.sin(elevation)
        result = self.temperature * (1. - numpy.exp(-secz.value * self.tau))
        return result

    def emission_spectrum(self, elevation, pressure = None, temperature = None,
                 tau_225 = None, mm_H2O = None):
        """
        Returns a BasicSpectrum of the atmosphere emission at requested
        elevation

        The returned emission is simply exp(secz * tau) in Jy, where secz is
        the airmass. So this is not the true atmosphere emission, it is missing
        a normalization factor

        The atmosphere model can be changed on the fly by providing new models
        parameters (pressure, temperature, opacity at 225 GHz or water vapour
        content). If none of these parameters are provide, the current model of
        atmosphere is used.

        Parameters
        ----------
        elevation: astropy.coordinates.Angle
            requested elevation. If this is the only parameter given, the
            current model is used.

        pressure: astropy.units.quantity
            pressure for which a new model is to be computed before determining
            the transmission at given elevation

         temperature: astropy.units.quantity
            temperature for which a new model is to be computed before
            determining the transmission at given elevation

        tau_225: float
            opacity at 225 GHz at which a new model is to be computed before
            determining the transmission at given elevation

        mm_H2O: astropy.units.quantity
            water vapour content at which a new model is to be computed before
            determining the transmission at given elevation

        Returns
        -------
        output: BasicSpectrum
            atmosphere emission spectrum
        """
        y = (self.emission(elevation, pressure = pressure,
                          temperature = temperature, tau_225 = tau_225,
                          mm_H2O = mm_H2O)).value *u.Jy
        return BasicSpectrum(x = self.frequencies.copy(), y=y)

    def tau_as_spectrum(self, pressure = None, temperature = None,
                 tau_225 = None, mm_H2O = None):
        """
        Returns a BasicSpectrum of the atmosphere opacity at requested
        elevation

        The returned spectrum is (tau Jy), which is of course a physical non-
        sense since tau as no units, but the BasicSpectrum class only accept
        an spectral irradiance as input. It is however useful to derive average
        opacities in a band.

        The atmosphere model can be changed on the fly by providing new models
        parameters (pressure, temperature, opacity at 225 GHz or water vapour
        content). If none of these parameters are provide, the current model of
        atmosphere is used.

        Parameters
        ----------
        pressure: astropy.units.quantity
            pressure for which a new model is to be computed before determining
            the transmission at given elevation

         temperature: astropy.units.quantity
            temperature for which a new model is to be computed before
            determining the transmission at given elevation

        tau_225: float
            opacity at 225 GHz at which a new model is to be computed before
            determining the transmission at given elevation

        mm_H2O: astropy.units.quantity
            water vapour content at which a new model is to be computed before
            determining the transmission at given elevation

        Returns
        -------
        output: BasicSpectrum
            tau as a spectrum
        """
        y = self.tau *u.Jy
        return BasicSpectrum(x = self.frequencies.copy(), y=y)

    def opacity(self, pressure=None, temperature=None, tau_225 = None,
            mm_H2O = None):
        """
        returns the array of opacity for the current atmosphere model.

        The atmosphere model can be changed on the fly by providing new models
        parameters (pressure, temperature, opacity at 225 GHz or water vapour
        content). If none of these parameters are provide, the current model of
        atmosphere is used.

        Parameters
        ----------
        pressure: astropy.units.quantity
            pressure for which a new model is to be computed before determining
            the transmission at given elevation

         temperature: astropy.units.quantity
            temperature for which a new model is to be computed before
            determining the transmission at given elevation

        tau_225: float
            opacity at 225 GHz at which a new model is to be computed before
            determining the transmission at given elevation

        mm_H2O: astropy.units.quantity
            water vapour content at which a new model is to be computed before
            determining the transmission at given elevation

        Returns
        -------
        output: (nf,) ndarray
            array of opacities at frequencies self.frequencies (or self.nu())
        """
        if pressure is not None or temperature is not None:
            self.select_grid(pressure=pressure, temperature=temperature)
            if tau_225 is None and mm_H2O is None:
                # the quantity that does not change is tau_225,
                # while mm_H2O will adjust depending on the conditions
                self.set_tau_225()
        if tau_225 is not None and mm_H2O is not None:
            raise ValueError('cant set both tau_225 and mm_H2O')
        if tau_225 is not None:
            self.set_tau_225(tau_225=tau_225)
        if mm_H2O is not None:
            self.set_mm_H2O(mm_H2O=mm_H2O)
        if not self.conditions_set:
            raise ValueError("conditions are not set")
        return self.tau

    def nu(self, unit):
        """
        Returns the frequencies at which the model is computed

        Parameters
        ----------
        unit: astropy.unit  

        Returns
        -------
        output: (nf, ) array like astropy.units.quantity
          the frequencies at which the model is computed
        """
        if DEBUG:
            print('Atmosphere.nu()')

        return self.frequencies.to(unit)


    def set_tau_225(self, tau_225=None):
        """
        Compute the water vapor content and the opacity values

        Determines the water vapour content to reproduce the input opacity at
        225 GHz. At this frequency, we have:
          tau_225 = tau_dry + mm_H2O * tau_wet

        Parameters
        ----------
        tau_225: float
          The opacity at 225 GHz

        Returns
        -------
        mmH20: astropy.units.quantity
          the amount of water vapor in mm in the atmosphere.

        Effects
        -------
        The following attributes are modified:
        tau_225, mm_H2O, tau_225_set, tau
        """
        if tau_225 is None:
            if self.tau_225_set:
                tau_225 = self.tau_225
            else:
                raise ValueError('tau_225 not given')
        if not self.grid_selected:
            raise ValueError("set pressure and temperature first")
        self.tau_225 = tau_225
        imin = numpy.argmin(numpy.abs(self.frequencies - 225*u.GHz))
        taud = self.grid_tau_dry[imin, self.idt, self.idp]
        tauw = self.grid_tau_wet[imin, self.idt, self.idp]
        result = (tau_225 - taud) / tauw
        self.mm_H2O = result * u.mm
        if (result < 0):
            raise Warning("Water Vapor content is negative: {}".\
                          format(self.mm_H2O))
        self.tau_225_set = True
        if self.tau_225_set and self.grid_selected:
            self.conditions_set = True
            self.tau = self.grid_tau_dry[:, self.idt, self.idp] + \
                        self.mm_H2O.value * self.grid_tau_wet[:, self.idt,
                                           self.idp]
            # this is to ensure that everything goes smoothly in IramAtmosphere
            self._set_internals()
        return self.mm_H2O

    def set_mm_H2O(self, mm_H2O=None):
        """
        Compute the water vapor content and the opacity values

        Determines the water vapour content to reproduce the input opacity at
        225 GHz. At this frequency, we have:
          tau_225 = tau_dry + mm_H2O * tau_wet

        Parameters
        ----------
        mm_H20: astropy.units.quantity or float
          Water vapour content of the atmosphere. If a float is given, it is
          assumed to be in mm

        Returns
        -------
        tau_225: float
          the sky opacity at 225 GHz

        Effects
        -------
        The following attributes are modified:
        tau_225, mm_H2O, tau_225_set, tau
        """
        if mm_H2O is None:
            if self.tau_225_set:
                mm_H2O = self.mm_H2O
            else:
                raise ValueError('mm_H2O not given')
        if not self.grid_selected:
            raise ValueError("set pressure and temperature first")
        if not isinstance(mm_H2O, u.Quantity):
            self.mm_H2O = mm_H2O * u.mm
        else:
            if mm_H2O.unit is not u.mm:
                raise ValueError('mm_H2O unit should be mm')

        imin = numpy.argmin(numpy.abs(self.frequencies - 225*u.GHz))
        taud = self.grid_tau_dry[imin, self.idt, self.idp]
        tauw = self.grid_tau_wet[imin, self.idt, self.idp]
        result = taud + mm_H2O.value * tauw
        self.mm_H2O = mm_H2O
        self.tau_225 = result
        self.tau_225_set = True
        if self.tau_225_set and self.grid_selected:
            self.conditions_set = True
            self.tau = self.grid_tau_dry[:, self.idt, self.idp] + \
                        mm_H2O.value * self.grid_tau_wet[:, self.idt, self.idp]
            self._set_internals()
        return self.tau_225

    def _set_internals(self):
        """
        A dummy method that will be overloaded in IramAtmosphere
        """
        if DEBUG:
            print("Amosphere.set_internals()")
        pass


    def select_grid(self, temperature=None, pressure=None):
        """
        Select the atmosphere model among the grid of models

        The closest matching model is used, without any interpolation. Models
        are close enough that this is accurate enough.

        Parameters
        ----------
        temperature: astropy.unit.quantity
          the temperature

        pressure: astropy.unit.quantity
          the pressure

        Returns
        -------
        None

        Effects
        -------
        The following attributes are modified:
        temperature, pressure, idp, idf, grid_selected

        """
        if temperature is None:
            if self.grid_selected:
                temperature = self.temperature
            else:
                raise ValueError('temperature not given')
        if pressure is None:
            if self.grid_selected:
                pressure = self.pressure
            else:
                raise ValueError('pressure not given')
        self.conditions_set = False
        self.temperature = temperature.to(u.K, equivalencies=u.temperature())
        self.pressure = pressure
        self.idt = numpy.argmin(numpy.abs(self.temperatures-self.temperature))
        self.idp = numpy.argmin(numpy.abs(self.pressures-pressure))
        dp = numpy.abs(pressure - self.pressures[self.idp])
        if dp > (10. * u.hPa):
            print("WARNING: the requested pressure is more than 10 hPa away "
                  "from grid values")
        dt = numpy.abs(temperature - self.temperatures[self.idt])
        if dt > (10. * u.K):
            print("WARNING: the requested temperature is more than 10 K away "
                  "from grid values")
        self.grid_selected = True
        return None

    def current_model_temperature(self):
        """
        Returns the current model temperature

        Since the select_grid() method picks the closest model, the temperature
        given is not usually the temperature used to compute the model. This
        method returns the model temperature

        Parameters
        ----------
        None

        Returns
        -------
        output: astropy.units.quantity
          Model temperature
        """
        if self.grid_selected:
            result = self.temperatures[self.idt]
        else:
            result = None
        return result

    def current_model_pressure(self):
        """
        Returns the model pressure

        Since the select_grid() method picks the closest model, the pressure
        given is not usually the pressure used to compute the model. This
        method returns the model pressure

        Parameters
        ----------
        None

        Returns
        -------
        output: astropy.units.quantity
          Model pressure
        """
        if self.grid_selected:
            result = self.pressures[self.idp]
        else:
            result = None
        return result

    def __str__(self):
        """
        overloading string representation
        """
        result = "Model:               {}\n".format(self.model)
        result = result + "Observatory:         {}\n".format(self.observatory)
        result = result + "Profile:             {}\n".format(self.profile)
        result = result + "File:                {}\n".format(self.file)
        result = result + "Grid set:            {}\n".format(self.grid_selected)
        result = result + "Current temperature: {}\n".format(self.temperature)
        result = result + "Model temperature:   {}\n".format(
            self.current_model_temperature())
        result = result + "Current pressure:    {}\n".format(
            self.pressure)
        result = result + "Model pressure:      {}\n".format(
            self.current_model_pressure())
        result = result + "Tau set:             {}\n".format(self.tau_225_set)
        result = result + "Tau 225 GHz:         {}\n".format(self.tau_225)
        result = result + "mm H2O:              {}\n".format(self.mm_H2O)
        result = result + "Ready to compute     {}\n".format(
            self.conditions_set)
        return result

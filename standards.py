__author__ = 'haussel'
"""
This module provides function to obtain standard spectra.

- blackbody
- greybody
- cohen 2003 standards
- vega cohen 1992
- vega stis 005
- vega stis 008
"""
import numpy as np
import os
import inspect
import glob
import re
from astropy import units as u
from astropy import constants as const
from astropy.table import Table
from astropy.io import fits
from .spectrum import BasicSpectrum, StellarLibrary
from scipy.spatial import cKDTree
from .phottools import is_frequency, is_wavelength, is_flam, is_fnu, \
    quantity_scalar, quantity_1darray, ndarray_1darray, lam_unit, fnu_unit, \
    PhotometryHeader

from .config import STELLAR_LIBRARY_DIR

def blackbody(temperature, x):
    """
    Returns black body spectra.

    Parameters
    ----------
    temperature: astropy.units.Quantity
        The black body temperatures
    x: astropy.units.Quantity
        The array of frequencies or wavelengths.

    Returns
    -------
    BasicSpectrum object
    """
    msg = quantity_1darray(x)
    if msg is not None:
        raise ValueError('x '+msg)
    if not isinstance(temperature, u.Quantity):
        raise ValueError("Temperature must be a quantity")
    msg = quantity_1darray(temperature)
    if msg is None:
        wt = temperature[:,np.newaxis]
        wx = x[np.newaxis,:]
    else:
        msg = quantity_scalar(temperature)
        if msg is None:
            wt = temperature
            wx = x
        else:
            raise ValueError("temperature is neither a scalar or 1d array "
                             "Quantity")
    if is_frequency(x.unit):
        hnukt = np.exp(const.h * wx / (const.k_B * wt))
        Bnu = 2.0 * const.h / const.c**2 * wx**3/(hnukt-1.0)
    elif is_wavelength(x.unit):
        hnukt = np.exp(const.h * const.c / (wx * const.k_B * wt))
        Bnu = 2.0 * const.h * const.c**2 / wx**5 / (hnukt - 1.0)
    else:
        raise ValueError("x is neither a frequency or a wavelength")
    print("x.shape = {}".format(x.shape))
    print("Bnu.shape = {}".format(Bnu.shape))
    result = BasicSpectrum(x=x, y=Bnu, interpolation_method='log-log-linear',
                           extrapolate=False)
    result.temperature = temperature
    return result

def power_law(x0, f0, alpha=None, x1=None, f1=None):
    """
    Returns a BasicSpectrum with a power law.

    The spectrum type depends on the units of the input. It will be:
        fnu = fnu0 (nu/nu_0)**alpha
    or  flam = flam0 (lam/lam0)**alpha

    Instead of alpha, other x and f values can be passed with the parameters
    x1 and f1. If none is given, returns a flat spectrum and issues a warning.

    Parameters
    ----------
    x0 : astropy.units.quantity
      The reference frequency (nu_0) or wavelength (lam_0), depending on its
      unit
    f0 : astropy.units.quantity
      The reference specral irradiance per unit of frequency (i.e. fnu) or
      wavelength (i.e. flam) depending on its unit
    alpha: float or (N, ) ndarray
      The slope(s) of the power law.
    x1 : float or (N,) ndarray
      The other frequencies or wavelengths
    f1 : float or (N,) ndarray
      The other fluxes

    Returns
    -------
    output: BasicSpectrum
    """
    if is_frequency(x0.unit):
        if is_fnu(f0.unit):
            if x1 is None and f1 is None:
                if alpha is None:
                    walpha = 0
                else:
                    walpha = alpha
                nu = x0.value * np.logspace(-2,2, 5)
            elif x1 is not None and f1 is not None:
                if alpha is not None:
                    raise ValueError("Incompatible parameters alpha and x1,f1")
                else:
                    if is_frequency(x1.unit):
                        if is_fnu(f1.unit):
                            walpha = np.log10(f1/f0) / np.log10(x1/x0)
                            if isinstance(x1, np.ndarray):
                                nu = np.unique(np.hstack((x0.value,
                                                          (x1.to(x0.unit)).\
                                                          value)))
                            else:
                                nu = np.sort(np.array([x0, x1]))
                        else:
                            ValueError("incompatible units between f0 and f1")
                    else:
                        ValueError("incompatible units between x0 and x1")
            else:
                raise ValueError("missing input parameters")
            if isinstance(walpha, np.ndarray):
                fnu = (f0.value * (nu[:, np.newaxis]/x0.value)**alpha.T).T
            else:
                fnu = f0.value * (nu/x0.value)**walpha
            result = BasicSpectrum(x=nu * x0.unit, y=fnu * f0.unit,
                                   interpolation_method='log-log-linear',
                                   extrapolate='yes')
        else:
            raise ValueError("incompatible units between x_0 and f_0")
    elif is_wavelength(x0.unit):
        if is_flam(f0.unit):
            if x1 is None and f1 is None:
                if alpha is None:
                    walpha = 0
                else:
                    walpha = alpha
                lam = x0.value * np.logspace(-2,2, 5)
            elif x1 is not None and f1 is not None:
                if alpha is not None:
                    raise ValueError("Incompatible parameters alpha and x1,f1")
                else:
                    if is_wavelength(x1.unit):
                        if is_flam(f1.unit):
                            walpha = (np.log10(f1/f0) / np.log10(x1/x0)).value
                            if isinstance(x1, np.ndarray):
                                lam = np.unique(np.hstack((x0.value,
                                                           (x1.to(x0.unit)).\
                                                           value)))
                            else:
                                lam = np.sort(np.array([x0, x1]))
                        else:
                            ValueError("incompatible units between f0 and f1")
                    else:
                        ValueError("incompatible units between x0 and x1")
            else:
                raise ValueError("missing input parameters")
            if isinstance(walpha, np.ndarray):
                flam = (f0.value * (lam[:, np.newaxis]/x0.value)**alpha.T).T
            else:
                flam = f0.value * (lam/x0.value)**alpha
            print("lam = {}".format(lam * x0.unit))
            print("flam.shape = {}".format(flam.shape))
            print("flam = {}".format(flam * f0.unit))
            result = BasicSpectrum(x=lam * x0.unit, y=flam * f0.unit,
                                   interpolation_method='log-log-linear',
                                   extrapolate='yes')
        else:
            raise ValueError("incompatible units between x_0 and f_0")
    else:
        raise ValueError("Invalid unit for parameter x_0")
    return result


def cohen2003_table():
    """
    Reads the internal compilation of tables 7 and 8 of Cohen et al. (2003),
    ApJ 125, 2645.

    Returns
    -------
    returns an astropy.table.Table
    """
    dummy = BasicSpectrum()
    classpath = inspect.getfile(dummy.__class__)
    basepath = os.path.dirname(classpath)
    cohen2003_dir = os.path.join(basepath, 'data/spectra/cohen2003')
    tabfile = os.path.join(cohen2003_dir, 'table7and8.csv')
    t = Table.read(tabfile)
    return t

def cohen2003_star(starname):
    """
    Read one of the Cohen et al. (2003), ApJ 125, 2645 spectrum

    Parameters
    ----------
    starname: str
        Name of the star in table 7 or 8 of Cohen et al. (2003)
        The available stars are given by cohen2003_table()

    Returns
    -------
    Returns a BasicSpectrum object
    """
    t = cohen2003_table()
    idx, = np.where(t['Name'].data == starname)
    if len(idx) !=1 :
        raise ValueError("Star {} is not in Cohen+2003".format(starname))
    dummy = BasicSpectrum()
    classpath = inspect.getfile(dummy.__class__)
    basepath = os.path.dirname(classpath)
    cohen2003_dir = os.path.join(basepath, 'data/spectra/cohen2003')
    starfile = os.path.join(cohen2003_dir, t['File'][idx[0]])
    t = Table.read(starfile, format='ascii', data_start=t['DataStart'][idx[0]])
    # some wavelength are dupicated.
    idx, = np.where(t['col1'].data[1:]-t['col1'].data[:-1] == 0)
    for i in idx:
        t['col1'][i] = t['col1'][i]-(t['col1'][i]-t['col1'][i-1])*0.1
    t['col1'].unit = u.micron
    t['col2'].unit = u.W/u.cm**2/u.micron
    spec = BasicSpectrum(table=t, name_x='col1', names_y=['col2'])
    return spec

def vega_cohen_1992():
    """
    Read Vega spectrum from Cohen et al. (1992)

    :return: BasicSpectrum
    """
    dummy = BasicSpectrum()
    classpath = inspect.getfile(dummy.__class__)
    basepath = os.path.dirname(classpath)
    default_vega_dir = os.path.join(basepath, 'data/spectra/vega')
    vegafile = os.path.join(default_vega_dir, 'alp_lyr.cohen_1992')
    data = np.genfromtxt(vegafile)
    hd = PhotometryHeader()
    hd.add_card_value('filename','alpha_lyr.cohen_1992')
    result = BasicSpectrum(x=data[:,0], x_unit=u.micron, header=hd,
                           y=data[:,1], y_unit=u.W/u.cm**2/u.micron)
    return result

def vega_stis_005():
    """
    Returns the CALSPEC alpha_lyr_stis_005 as a BasicSpectrum

    :return: BasicSpectrum
    """
    dummy = BasicSpectrum()
    classpath = inspect.getfile(dummy.__class__)
    basepath = os.path.dirname(classpath)
    default_vega_dir = os.path.join(basepath, 'data/spectra/vega')
    vegafile = os.path.join(default_vega_dir, 'alpha_lyr_stis_005.fits')
    hd = PhotometryHeader()
    hd.add_card_value('filename','alpha_lyr_stis_005.fits')
    t = Table.read(vegafile)
    result = BasicSpectrum(x=t['WAVELENGTH'].data * u.Angstrom, header=hd,
                           y=t['FLUX'].data * u.erg/u.s/u.cm**2/u.Angstrom)
    return result

def vega_stis_008():
    """
    Returns the CALSPEC alpha_lyr_stis_005 as a BasicSpectrum

    :return: BasicSpectrum
    """
    dummy = BasicSpectrum()
    classpath = inspect.getfile(dummy.__class__)
    basepath = os.path.dirname(classpath)
    default_vega_dir = os.path.join(basepath, 'data/spectra/vega')
    vegafile = os.path.join(default_vega_dir, 'alpha_lyr_stis_008.fits')
    t = Table.read(vegafile)
    hd = PhotometryHeader()
    hd.add_card_value('filename','alpha_lyr_stis_008.fits')
    result = BasicSpectrum(x=t['WAVELENGTH'].data * u.Angstrom, header=hd,
                           y=t['FLUX'].data * u.erg/u.s/u.cm**2/u.Angstrom)
    return result


def vega_ck_1994(norm='Hayes85'):
    """
    Returns the Vega spectrum of Castelli & Kurucz (1994).

    The Vega spectrum is a luminosity, and needs to be converted to a flux
    density at earth.

    fl = 4 pi Hnu c / lambda**2 / (d/R)**2

    where l is the distance to Vega, and (d/R)**2 is the dilution factor:
    d/R =  206265/(theta/2))
    with theta is the stellar angular diameter corrected for limb darkening and
    206265 is the distance of 1pc in AU.

    Different normalisations are provided the norm keyword.

    Parameters:
    -----------

    norm: str
        can be:
            'Hayes85'  : uses the value of 3.44e-9 erg/s/cm**2/A at 5556 A.
            'Bessel98' : uses theta = 3.24 mas

    Returns:
    --------
    BasicSpectrum
    """
    dummy = BasicSpectrum()
    classpath = inspect.getfile(dummy.__class__)
    basepath = os.path.dirname(classpath)
    default_vega_dir = os.path.join(basepath, 'data/spectra/vega')
    vegafile = os.path.join(default_vega_dir, 'vega.sed')
    t = Table.read(vegafile, names=['wavelength','hlam'], comment='#',
                   format='ascii')
    hd = PhotometryHeader()
    hd.add_card_value('filename','vega.sed')
    vega = BasicSpectrum(x=t['wavevelength'], header=hd,
                         y=t['hlam'] * u.W / u.m**2 / u.angstrom)
    if norm == 'Hayes85':
        val = vega.flam_lam(5556 * u.angstrom)
        vega.scale(3.44e-9 * u.erg / u.s / u.cm**2 / u.angstrom / val,
                   inplace=True)
        vega.hd.add_card_value('comment', 'scaled to the Hayes (1985) value '
                                          'Flam(5556A) = {}'.
                    format(3.44e-9 * u.erg / u.s / u.cm**2 / u.angstrom))
    else:
        raise NotImplementedError
    return vega

#### BaSeL 2p2 Library

class BaSeL2p2Match:
    def __init__(self, Teff, logg, met, vturb, xh):
        self.Teff = Teff
        self.logg = logg
        self.met = met
        self.vturb = vturb
        self.xh = xh
        try:
            l = len(self.Teff)
        except TypeError:
            params = np.zeros((1, 3))
            params[0, :] = [Teff, logg, met]
        else:
            params = np.stack([self.Teff, self.logg, self.met], axis=-1)
        self.tree = cKDTree(params)

    def __getitem__(self, item):
        return BaSeL2p2Match(self.Teff[item], self.logg[item],
                             self.met[item], self.vturb[item], self.xh[item])

    def __str__(self):
        result = 'BaSeL 2.2 Library by Lejeune et al. (1998)\n'
        result += 'Library parameters ({} values): \n'.format(len(self.Teff))
        result += 'Teff (matching param) from {} to {}\n'.format(np.min(
            self.Teff), np.max(self.Teff))
        result += 'logg (matching param) from {} to {}\n'.format(np.min(
            self.logg), np.max(self.logg))
        result += 'met (matching param) from {} to {}\n'.format(np.min(
            self.met), np.max(self.met))
        result += 'vturb (not matching param) from {} to {}\n'.format(np.min(
            self.vturb), np.max(self.vturb))
        result += 'xh (not matching param) from {} to {}\n'.format(np.min(
            self.xh), np.max(self.xh))
        return result

    def _closest_match(self, Teff, logg, met, return_distance=False):
        try:
            l = len(Teff)
        except TypeError:
            params = [Teff, logg, met]
        else:
            params = np.stack([Teff, logg, met], axis=-1)
        distance, index = self.tree.query(params)
        if return_distance:
            return index, distance
        else:
            return index

    def _exact_match(self, Teff, logg, met):
        index, distance = self._closest_match(Teff, logg, met,
                                              return_distance=True)
        if np.any(distance > 0):
            raise ValueError('no exact match')
        return index


class BaSeL2p2Reader:

    def __init__(self, libdir=None, asciifileregex=None):
        self.libdir=libdir
        self.regex = re.compile(asciifileregex)

    def __call__(self):
        filenames = glob.glob(os.path.join(self.libdir, '*'))
        istar = 0
        for filename in filenames:
            fname = os.path.basename(filename)
            if self.regex.match(fname):
                with open(filename, 'r') as f:
                    content = f.read()
                    bits = content.split()
                    lam = np.array(bits[0:1221], dtype=float)
                    nstars = int(bits[-1227])
                    if istar == 0:
                        hnu = np.zeros((nstars, 1221))
                        Teff = np.zeros(nstars)
                        logg = np.zeros(nstars)
                        met = np.zeros(nstars)
                        vturb = np.zeros(nstars)
                        xh = np.zeros(nstars)
                    else:
                        hnu = np.append(hnu, np.zeros((nstars, 1221)),
                                        axis=0)
                        Teff = np.append(Teff, np.zeros(nstars))
                        logg = np.append(logg, np.zeros(nstars))
                        met = np.append(met, np.zeros(nstars))
                        vturb = np.append(vturb, np.zeros(nstars))
                        xh = np.append(xh, np.zeros(nstars))
                    ipos = 1221
                    while ipos < len(bits):
                        Teff[istar] = float(bits[ipos+1])
                        logg[istar] = float(bits[ipos+2])
                        met[istar] = float(bits[ipos+3])
                        vturb[istar] = float(bits[ipos + 4])
                        xh[istar] = float(bits[ipos + 5])
                        hnu[istar, :] = np.array(bits[ipos+6:ipos+6+1221],
                                                 dtype=float)
                        istar = istar + 1
                        ipos = ipos + 1227
        matchmaker = BaSeL2p2Match(Teff, logg, met, vturb, xh)
        return lam * u.nm, hnu * u.erg / u.s / u.cm ** 2 / u.Hz, matchmaker


def BaSeL2p2(libdir=os.path.join(STELLAR_LIBRARY_DIR, 'BaSeL2.2'),
             asciifileregex='^lbc96_[mp]\d\d.cor$'):
    """
    Read the BaSeL 2.2 (Lejeune et al. 1998) as a StellarLibrary

    Parameters
    ----------
    libdir : str
        path to the directory where the library can be found.
    asciifileregex: str
        python regular expression
    """
    return StellarLibrary(reader=BaSeL2p2Reader(libdir=libdir,
                                                asciifileregex=asciifileregex),
                          interpolation_method='linear', extrapolate='no',
                          positive=True)



### Pickles 1998 Library
class Pickles1998Match:
    def __init__(self, sptype, lumclass, met):
        self.sptype = self.sptype2typ(sptype)
        self.lumclass = self.lumclass2lum(lumclass)
        self.met = self.met2num(met)
        if type(sptype) is str:
            self.s_sptype = np.array([sptype])
            self.s_lumclass = np.array([lumclass])
            self.s_met = np.array([met])
#            self.params = np.zeros((1, 3))
#            self.params[0,:] = [self.sptype, self.lumclass, self.met]
        else:
            self.s_sptype = np.array(sptype)
            self.s_lumclass = np.array(lumclass)
            self.s_met = np.array(met)
        params = np.stack([self.sptype, self.lumclass, self.met], axis=-1)
        self.tree = cKDTree(params)


    def __getitem__(self, item):
        return Pickles1998Match(self.s_sptype[item], self.s_lumclass[item],
                                self.s_met[item])

    def __str__(self):
        result = 'Pickles (1998) Library \n'
        result += 'Library parameters ({} values): \n'.format(len(
            self.sptype))
        result += 'sptype (matching param): {}\n'.format(self.s_sptype)
        result += 'lumclass (matching param): {} \n'.format(self.s_lumclass)
        result += 'met (matching param): {}\n'.format(self.s_met)
        return result

    def _closest_match(self, sptype, lumclass, met, return_distance=False):

        typ = self.sptype2typ(sptype)
        lum = self.lumclass2lum(lumclass)
        met = self.met2num(met)
        params = np.stack([typ, lum, met], axis=-1)
        distance, index = self.tree.query(params)
        if return_distance:
            return index, distance
        else:
            return index

    def _exact_match(self, sptype, lumclass, met):
        index, distance = self._closest_match(sptype, lumclass, met,
                                              return_distance=True)
        if np.any(distance > 0):
            raise ValueError('no exact match')
        return index

    def lumclass2lum(self, lumclass):
        d = {'I': 0.0, 'II': 1.0, 'III': 2.0, 'IV': 3.0, 'V': 4.0}
        if type(lumclass) is str:
            result = [d[lumclass]]
        else:
            result = [d[x] for x in lumclass]
        return result

    def lum2lumclass(self, lum):
        d = ['I', 'II', 'III', 'IV', 'V']
        result = [d[x] for x in lum]
        return result

    def sptype2typ(self, sptype):
        d = {'O': 0.0, 'B': 1.0, 'A': 2.0, 'F': 3.0, 'G': 4.0, 'K': 5.0,
             'M': 6.0}
        r = re.compile('([OBAFGKM])((\d)(\.?)(/?)(\d?))')
        if type(sptype) is str:
            wstptype = [sptype]
        else:
            wstptype = sptype
        result = []
        for s in wstptype:
            m = r.match(s)
            if m:
                if m.group(4) is '.':
                    digit = float(m.group(2))/10.
                elif m.group(5) is '/':
                    digit = 0.5 * (float(m.group(3))+float(m.group(6)))/10.
                elif m.group(4) is '' and m.group(5) is '':
                    digit = np.mean([float(x)/10. for x in m.group(2)])
                result.append(d[m.group(1)]+digit)
            else:
                raise ValueError('Invalid spectral type: "{}"'.format(s))
        return result

    def met2num(self, met):
        d = {'w':0.0, '':1.0, 'r':2.0}
        if type(met) is str:
            result = [d[met]]
        else:
            result = [d[x] for x in met]
        return result

class Pickles1998Reader:
    def __init__(self, libdir=None):
        self.libdir=libdir

    def __call__(self):
        synphot = Table.read(os.path.join(self.libdir, 'synphot.dat'),
                             format='ascii')

        r = re.compile('([rw]?)([OBAFGKM]\d\.?/?\d?)([IV]+)')
        sptypes = []
        lumclasses = []
        mets = []
        for i, row in enumerate(synphot):
            m = r.match(row['col5'])
            if m:
                mets.append(m.group(1))
                sptypes.append(m.group(2))
                lumclasses.append(m.group(3))
            else:
                raise ValueError('invalid spectral class :"{}"'.
                                 format(row['col5']))
            fname = os.path.join(self.libdir, 'uk' + synphot['col25'][i] +
                                 '.dat')
            t = Table.read(fname, format='ascii.cds', readme=os.path.join(
                self.libdir, 'ReadMe'))
            if i == 0:
                x = t['lambda'].data
                y = t['nflam'].data
            else:
                if np.any((x - t['lambda'].data) != 0):
                    raise ValueError('Inconsistent wavelengths in file {'
                                     '}'.format(fname))
                y = np.append(y, t['nflam'].data)
        x = x * u.Angstrom
        y = y.reshape((len(synphot), len(x))) * u.erg / u.s / u.cm**2 / u.Angstrom
        matchmaker = Pickles1998Match(sptypes, lumclasses, mets)
        return x, y, matchmaker

def Pickles1998(libdir=os.path.join(STELLAR_LIBRARY_DIR, 'Pickles1998')):
    """
    Read the Pickles (1998) library as a StellarLibrary

    Parameters
    ----------
    libdir : str
        path to the directory where the library can be found.
    asciifileregex: str
        python regular expression
    """
    reader = Pickles1998Reader(libdir=libdir)
    return StellarLibrary(reader=reader, interpolation_method='linear',
                          extrapolate='no', positive=True)


## Euclid OU-SIM stellar library

class EuclidOUSIMMatcher:
    def __init__(self, Teff, logg, feh):
        try:
            l = len(Teff)
        except TypeError:
            self.Teff = np.array([Teff])
            self.logg = np.array([logg])
            self.met = np.array([feh])
            params = np.zeros((1, 3))
            params[0, :] = [Teff, logg, feh]
        else:
            self.Teff = Teff
            self.logg = logg
            self.met = feh
            params = np.stack([self.Teff, self.logg, self.met], axis=-1)
        self.tree = cKDTree(params)

    def __getitem__(self, item):
        return EuclidOUSIMMatcher(self.Teff[item], self.logg[item],
                                  self.met[item])

    def __str__(self):
        result = 'BaSeL 2.2 Library by Lejeune et al. (1998) (OU-SIM version)\n'
        result += 'Library parameters ({} values): \n'.format(len(self.Teff))
        result += 'Teff (matching param) from {} to {}\n'.format(np.min(
            self.Teff), np.max(self.Teff))
        result += 'logg (matching param) from {} to {}\n'.format(np.min(
            self.logg), np.max(self.logg))
        result += 'met (matching param) from {} to {}\n'.format(np.min(
            self.met), np.max(self.met))
        return result

    def _closest_match(self, Teff, logg, met, return_distance=False):
        try:
            l = len(Teff)
        except TypeError:
            params = [Teff, logg, met]
        else:
            params = np.stack([Teff, logg, met], axis=-1)
        distance, index = self.tree.query(params)
        if return_distance:
            return index, distance
        else:
            return index

    def _exact_match(self, Teff, logg, met):
        index, distance = self._closest_match(Teff, logg, met,
                                              return_distance=True)
        if np.any(distance > 0):
            raise ValueError('no exact match')
        else:
            return index


class EuclidOUSIMReader:
    def __init__(self, libdir=None, libfitsfile=None):
        self.libdir = libdir
        self.libfitsfile = libfitsfile

    def __call__(self):
        hdul = fits.open(os.path.join(self.libdir, self.libfitsfile))
        x = hdul['LAMBDA'].data * \
            u.Quantity('1 {}'.format(hdul['LAMBDA'].header['UNIT']))
        y = hdul['SED'].data * u.erg / u.s / u.cm**2 / u.Hz
        params = Table(hdul['SED_PARAM'].data)
        matcher = EuclidOUSIMMatcher(params['TEFF'].data, params['LOGG'].data,
                                     params['FEH'].data)
        return x, y, matcher


def EuclidOUSIM(libdir=os.path.join(STELLAR_LIBRARY_DIR, 'OU-SIM'),
                libfitsfile='EUC-TEST-SEDLIB-2013-11-14T152700.000.fits'):
    reader = EuclidOUSIMReader(libdir=libdir, libfitsfile=libfitsfile)
    return StellarLibrary(reader=reader, interpolation_method='linear',
                          extrapolate='no', positive=True)



class OldStellarLibrary(BasicSpectrum):
    """
    Provide support for spectral libraries, such as the BaSeL 2.2 one (Lejeune
    et al., 1998) as BasicSpectrum.

    The library can consist in a set of ascii files, such as the distribution
    off BaSeL 2.2, or a single fits file. In the latter case, it is expected
    that the fits file contains:
    extension 'LAMBDA' containing the common array of wavelengths. Unit of
    wavelengths shall be specified by the 'UNIT' keyword
    extension 'SED' containing the spectra, in units of erg/s/cm**2/Angstrom
    extension 'SED_PARAM' containing a bin table with TEFF, LOGG and FEH

    Beside the spectra, the spectral parameters are provided:
    Teff : effective temperature (K)
    logg : log surface gravity
    met  : metallicity [Fe/H]
    vturb: turbulent velocity (km/s)
    xh   : I will need to get back to the paper to check what this is.
    tree : kDTree to quickly search matching spectra
    """
    def __init__(self, wavelength=None, hnu=None, Teff=None, logg=None,
                 met=None, vturb=None, xh=None,
                 libdir=None, fitsfilename=None,
                 asciifileregex='^lbc96_[mp]\d\d.cor$'):
        """
        Read the spectral files. The distributed files are normally with a name
        corresponding to the regular expression ^lbc96_[mp]\d\d.cor$. This
        can be adjusted with the fileregex keyword.
        The location of the files must be provided using the libdir keyword

        Parameters:
        -----------
        libdir: str
            path to the directory where the library file(s) is located

        fitsfilename: str
            name of the fits file containing the library.

        asciifileregex: str
            Regular expression to find the library files.
        """
        self.Teff = None
        self.logg = None
        self.met = None
        self.vturb = None
        self.xh = None
        if wavelength is not None and hnu is not None and Teff is not None \
                and logg is not None and vturb is not None and xh is not None:
            self.Teff = Teff
            self.logg = logg
            self.met = met
            self.vturb = vturb
            self.xh = xh
        elif libdir is not None:
            if fitsfilename is not None:
                wavelength, hnu = self._read_from_fitsfile(libdir, fitsfilename)
            else:
                wavelength, hnu = self._read_from_asciifile(libdir,
                                                            asciifileregex)
            # get rid of the zeroes in the spectra
            maxi = 0
            for i in range(hnu.shape[0]):
                idx, = np.where(hnu[i, :].value == 0.)
                if len(idx) > 0:
                    locmax = np.max(idx)
                    if locmax > maxi:
                        maxi = locmax
            wavelength = wavelength[(maxi + 1):]
            hnu = hnu[:, (maxi + 1):]

        else:
            raise ValueError('Invalid parameters')
        super().__init__(x=wavelength, y=hnu)
        msg = ndarray_1darray(self.Teff)
        if msg is None:
            params = np.stack([self.Teff, self.logg, self.met], axis=-1)
        else:
            params = np.zeros((1, 3))
            params[0, :] = [Teff, logg, met]
        self.tree = cKDTree(params)


    def _read_from_asciifile(self, libdir, fileregex):
        filenames = glob.glob(os.path.join(libdir, '*'))
        baselname = re.compile(fileregex)
        istar = 0
        for filename in filenames:
            fname = os.path.basename(filename)
            if baselname.match(fname):
                with open(filename, 'r') as f:
                    content = f.read()
                    bits = content.split()
                    lam = np.array(bits[0:1221], dtype=float)
                    nstars = int(bits[-1227])
                    if istar == 0:
                        hnu = np.zeros((nstars, 1221))
                        self.Teff = np.zeros(nstars)
                        self.logg = np.zeros(nstars)
                        self.met = np.zeros(nstars)
                        self.vturb = np.zeros(nstars)
                        self.xh = np.zeros(nstars)
                    else:
                        hnu = np.append(hnu, np.zeros((nstars, 1221)), axis=0)
                        self.Teff = np.append(self.Teff, np.zeros(nstars))
                        self.logg = np.append(self.logg, np.zeros(nstars))
                        self.met = np.append(self.met, np.zeros(nstars))
                        self.vturb = np.append(self.vturb, np.zeros(nstars))
                        self.xh= np.append(self.xh, np.zeros(nstars))
                    ipos = 1221
                    while ipos < len(bits):
                        self.Teff[istar] = float(bits[ipos+1])
                        self.logg[istar] = float(bits[ipos+2])
                        self.met[istar] = float(bits[ipos+3])
                        self.vturb[istar] = float(bits[ipos + 4])
                        self.xh[istar] = float(bits[ipos + 5])
                        hnu[istar, :] = np.array(bits[ipos+6:ipos+6+1221],
                                                 dtype=float)
                        istar = istar + 1
                        ipos = ipos + 1227
        return lam * u.nm, hnu * u.erg / u.s / u.cm ** 2 / u.Hz

    def _read_from_fitsfile(self, libdir, fitsfilename):
        hdul = fits.open(os.path.join(libdir, fitsfilename))
        wavelength = hdul['LAMBDA'].data * \
             u.Quantity('1 {}'.format(hdul['LAMBDA'].header['UNIT']))
        hnu = hdul['SED'].data * u.erg / u.s / u.cm**2 / u.Hz
        nbstars = hnu.shape[0]
        params = Table(hdul['SED_PARAM'].data)
        self.Teff = params['TEFF']
        self.logg = params['LOGG']
        self.met = params['FEH']
        self.vturb = np.zeros(nbstars)
        self.xh = np.zeros(nbstars)
        return wavelength, hnu

    def __getitem__(self, item):
        return StellarLibrary(wavelength=self.lam(unit=lam_unit),
                              hnu=self.fnu(unit=fnu_unit)[item,:],
                              Teff=self.Teff[item], logg=self.logg[item],
                              met=self.met[item], vturb=self.vturb[item],
                              xh=self.xh[item])

    def find_spectra(self, Teff, logg, met):
        """
        Return a spectral library containing the requested Teff, logg and
        metallicity only.

        Parameters:
        -----------
        Teff: float or ndarray
        logg: float or ndarray
        met:  float of ndarray

        Returns:
        --------
        StellarLibray of matching spectra
        """
        msg = ndarray_1darray(Teff)
        if msg is None:
            params = np.stack([Teff, logg, met], axis=-1)
        else:
            params = [Teff, logg, met]
        index, distance = self.tree.query(params)
        return self.__getitem__(index)

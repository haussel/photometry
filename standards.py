__author__ = 'haussel'
"""
This module provides function to obtain standard spectra.

- blackbody
- greybody
- cohen 2003 standards
- vega cohen 1992
- vega stis 005
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
from .spectrum import BasicSpectrum
from scipy.spatial import cKDTree
from .phottools import is_frequency, is_wavelength, is_flam, is_fnu, \
    quantity_scalar, quantity_1darray, ndarray_1darray, lam_unit, fnu_unit


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

    :return:
    """
    dummy = BasicSpectrum()
    classpath = inspect.getfile(dummy.__class__)
    basepath = os.path.dirname(classpath)
    default_vega_dir = os.path.join(basepath, 'data/spectra/vega')
    vegafile = os.path.join(default_vega_dir, 'alp_lyr.cohen_1992')
    data = np.genfromtxt(vegafile)
    result = BasicSpectrum(x=data[:,0], x_unit=u.micron,
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
    t = Table.read(vegafile)
    result = BasicSpectrum(x=t['WAVELENGTH'].data * u.Angstrom,
                           y=t['FLUX'].data * u.erg/u.s/u.cm**2/u.Angstrom)
    return result


class StellarLibrary(BasicSpectrum):
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

__author__ = 'haussel'
"""
This module provides specific classes to Euclid True Universe
"""
import numpy as np
import os
from astropy import units as u
from astropy.table import Table
from astropy.io import fits
from .spectrum import BasicSpectrum, StellarLibrary
from .extinction import ODonnellExtinction
from .passband import Passband
from scipy.spatial import cKDTree
from .phottools import PhotometryInterpolator
import re
from .config import PHOTOMETRY_INSTALL_DIR, STELLAR_LIBRARY_DIR, \
    GALAXY_LIBRARY_DIR, DEBUG


## Euclid OU-SIM stellar library

class EuclidOUSIMStarsMatcher:
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
        return EuclidOUSIMStarsMatcher(self.Teff[item], self.logg[item],
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


class EuclidOUSIMStarsReader:
    def __init__(self, libdir=None, libfitsfile=None):
        self.libdir = libdir
        self.libfitsfile = libfitsfile

    def __call__(self):
        hdul = fits.open(os.path.join(self.libdir, self.libfitsfile))
        x = hdul['LAMBDA'].data * \
            u.Quantity('1 {}'.format(hdul['LAMBDA'].header['UNIT']))
        y = hdul['SED'].data * u.erg / u.s / u.cm**2 / u.Hz
        params = Table(hdul['SED_PARAM'].data)
        matcher = EuclidOUSIMStarsMatcher(params['TEFF'].data,
                                          params['LOGG'].data,
                                          params['FEH'].data)
        return x, y, matcher


def EuclidOUSIMStars(libdir=os.path.join(STELLAR_LIBRARY_DIR, 'OU-SIM'),
                     libfitsfile='EUC-TEST-SEDLIB-2013-11-14T152700.000.fits'):
    reader = EuclidOUSIMStarsReader(libdir=libdir, libfitsfile=libfitsfile)
    return StellarLibrary(reader=reader, interpolation_method='linear',
                          extrapolate='no', positive=True)


EMISSION_LINES = np.array([
    ['logf_halpha_model3_ext',6564.614, 1.0], # H-alpha
    ['logf_hbeta_model3_ext', 4862.721, 1.0], # H-beta
    ['logf_o2_model3_ext',    3727.09,  2.0], # logf_o2_ext contains flux of
                                              #  both 3726 and 3728
    ['logf_o2_model3_ext',    3729.88,  2.0], # OII (secondary - assuming
                                              # ratio 1:1)
    ['logf_o3_model3_ext',   5008.239,  1.0], # OIII (primary)
    ['logf_o3_model3_ext',   4960.295,  3.0], # OIII (secondary - assuming
                                              # ratio 1:3)
    ['logf_n2_model3_ext',    6585.27,  1.0],  # NII (primary)
    ['logf_n2_model3_ext',    6549.86,  3.0],  # NII (secondary - ratio 1:3)
    ['logf_s2_model3_ext',    6718.29,  1.0],  # SII (primary)
    ['logf_s2_model3_ext',    6732.68,  1.0]])  # SII (secondary - assuming
                                                # ratio 1:1)

UNIT_SPECLIB = u.erg / u.s / u.cm**2 / u.Angstrom
REFERENCE_PASSBAND = {'file':'r01_SDSS.Euclid.pb', 'inCol': 'sdss_r01',
                      'band':None}

class EuclidOUSIMGalaxies:
    """

    """
    def __init__(self, libdir=os.path.join(GALAXY_LIBRARY_DIR, 'OU-SIM'),
                 setup=None):
        # Default values for catalog column names
        self.sed_index = 'sed_template'
        self.ext_curve = 'ext_law'
        self.ebv_internal = 'ebv'
        self.redshift = 'z_obs'
        self.r01_abs_mag = 'abs_mag_r01'
        self.reference_mag = 'ref_mag_r01'
        self.logf_halpha = 'logf_halpha_ext'
        self.logf_hbeta = 'logf_hbeta_ext'
        self.logf_o2 = 'logf_o2_ext'
        self.logf_o3 = 'logf_o3_ext'
        self.logf_n2 = 'logf_n2_ext'
        self.logf_s2 = 'logf_s2_ext'
        self.emission_lines = EMISSION_LINES
        self.refpb = REFERENCE_PASSBAND
        self.mw_av = 'Av'
        self.mw_ebv = None
        self.refmag_in_flux = False
        if setup is not None:
            self._read_setup(setup)
            self.emission_lines[0, 0] = self.logf_halpha
            self.emission_lines[1, 0] = self.logf_hbeta
            self.emission_lines[2:4,0] = self.logf_o2
            self.emission_lines[4:6, 0] = self.logf_o3
            self.emission_lines[6:8, 0] = self.logf_n2
            self.emission_lines[8:10, 0] = self.logf_s2
            self.refpb['inCol'] = self.reference_mag
        # load the reference passband
        self.refpb['band'] = Passband(file=self.refpb['file'])
        # final coverage of the spectrum (in Angstrom)
        self.wavelength = np.arange(2000., 50000., 1.)
        self.extinction = ODonnellExtinction()
        # Read the library of spectra
        hdul = fits.open(os.path.join(libdir,
                            'EUC-TEST-GSEDLIB-2015-03-18T171100.000.fits'))
        # wavelengths
        w = hdul['LAMBDA'].data
        # spectra
        specs = hdul['SED'].data
        # Fix the duplicate
        idx, = np.where((w[1:]-w[:-1]) == 0)
        while len(idx) > 0:
            w = np.concatenate((w[0:idx[0]], w[idx[0]+1:]))
            specs = np.concatenate((specs[:,0:idx[0]],
                                    specs[:,idx[0]+1:]), axis=1)
            idx, = np.where((w[1:] - w[:-1]) == 0)
        w = w * u.Quantity('1 {}'.format(hdul['LAMBDA'].header['UNIT']))
        # TODO : fix the interpretation of units so we can use the header one
        specs = specs * UNIT_SPECLIB
        specliborg = BasicSpectrum(x=w, y=specs,
                                   interpolation_method='log-log-linear')
        hdul.close()
        # Read the attenuation library
        hdul = fits.open(os.path.join(libdir,
                            'EUC-TEST-GEXTLAWLIB-2015-03-18T203100.000.fits'))
        wavelength = hdul['LAMBDA'].data * \
                     u.Quantity('1 {}'.format(hdul['LAMBDA'].header['UNIT']))
        # TODO : fix the interpretation of units so we can use the header one
        specs = hdul['SED'].data * UNIT_SPECLIB
        extliborg = BasicSpectrum(x=wavelength, y=specs,
                                  interpolation_method='log-log-linear')
        # interpolate the extinction at the wavelength of the spectra
        extlibinterp = extliborg.flam_lam(specliborg.x * specliborg.x_si_unit)
        # get rid of the NaNs
        ok, = np.where(np.isfinite(extlibinterp[0,:]))
        if len(ok) > 0:
            extlibinterp = extlibinterp[:, ok]
            w = specliborg.x[ok]
            y = specliborg.y[:, ok]
        # store the result
        self.speclib = BasicSpectrum(x=w * specliborg.x_si_unit,
                                     y=y * specliborg.y_si_unit)
        self.extlib = extlibinterp


    def _read_setup(self, setupfile):
        with open(setupfile, 'r') as f:
            lines = f.readlines()
        k = re.compile('^(.+):\s+(\S+.*)$')
        for line in lines:
            if line != '':
                m = k.match(line)
                if m:
                    key = m.group(1)
                    value = m.group(2)
                    if key == 'Halpha':
                        self.logf_halpha = value
                    elif key == 'Hbeta':
                        self.logf_hbeta = value
                    elif key == 'OII':
                        self.logf_o2 = value
                    elif key == 'OIII':
                        self.logf_o3 = value
                    elif key == 'NII':
                        self.logf_n2 = value
                    elif key == 'SII':
                        self.logf_s2 = value
                    elif key == 'SED':
                        self.sed_index = value
                    elif key == 'EXT':
                        self.ext_curve = value
                    elif key == 'EBV':
                        self.ebv_internal = value
                    elif key == 'Z':
                        self.redshift = value
                    elif key == 'R01_ABS_MAG':
                        self.r01_abs_mag = value
                    elif key == 'REFERENCE_MAG':
                        self.reference_mag = value
                    elif key == 'MW_AV':
                        self.mw_av = value
                        self.mw_ebv = None
                    elif key == 'MW_EBV':
                        self.mw_ebv = value
                        self.mw_av = None
                    elif key == 'REFMAG_IS_FLUX':
                        if value == 'True':
                            self.refmag_in_flux = True
                        elif value == 'False':
                            self.refmag_in_flux = False
                        else:
                            raise ValueError("Value {} should be 'True' or "
                                             "'False".format(value))
                    else:
                        raise ValueError("Unknown key '{}' in line '{}'".format(
                            key, line))
            else:
                raise ValueError("Cannot interpret line {}".format(line))

#    def _gaussian_profiles(self, lam, centers, widths):
#        result = 1.0/

    def __call__(self, incat):
        sed_index = incat[self.sed_index].data
        sed_base = sed_index.astype(int)
        sed_frac = sed_index - sed_base
        specslow = self.speclib[sed_base]
        sed_high = sed_base+1
        idx, = np.where(sed_high >= (self.speclib.nb-1))
        if len(idx) > 0:
            sed_high[idx] = self.speclib.nb-1
        specshigh = self.speclib[sed_high]
        specs = (1.-sed_frac[:, np.newaxis]) * specslow.y + \
                sed_frac[:,np.newaxis] * specshigh.y
        ext = (self.extlib[incat[self.ext_curve].data.astype(int),:] /
               self.extlib[0, :])**(incat[self.ebv_internal].data[:,np.newaxis]
                                    / 0.2)
        specs = specs * ext
        restwave = (self.speclib.x * self.speclib.x_si_unit).to(
            u.Angstrom).value
        # compute the spectra in observed wavelength
        obsspec = np.zeros((specslow.nb, len(self.wavelength)))
        for i in range(specslow.nb):
            interp = PhotometryInterpolator(restwave *
                                            (1.+incat[self.redshift][i]),
                                            specs[i, :], 'log-log-linear',
                                            extrapolate='no', positive=True)
            obsspec[i, :] = interp(self.wavelength)

        result = BasicSpectrum(x=self.wavelength * u.Angstrom,
                               y=obsspec*UNIT_SPECLIB)
        # scale the spectrum
        result.in_nu(reinterpolate=False)
        result.in_fnu(reinterpolate=True)
        mags = self.refpb['band'].mag_ab(result)
        if self.refmag_in_flux:
            mref = -2.5*np.log10(incat[self.refpb['inCol']])-48.6
        else:
            mref =incat[self.refpb['inCol']]
        sc = 10. ** (0.4 * (mags - mref))
        result.scale(sc)
        result.in_lam(reinterpolate=False)
        result.in_flam(reinterpolate=True)
        # Add emission lines
        widths = 10.**((-0.1 + 0.01 * incat[self.redshift]) *
                       (incat[self.r01_abs_mag]-3.0) -
                        0.05 * incat[self.redshift])
        idx, = np.where(widths < 50)
        widths[idx] = 50.
        fluxes = np.zeros((specslow.nb, self.emission_lines.shape[0]))
        sigmas = np.zeros((specslow.nb, self.emission_lines.shape[0]))
        center = np.zeros((specslow.nb, self.emission_lines.shape[0]))
        for i in range(self.emission_lines.shape[0]):
            fluxes[:,i] = 10.**(incat[self.emission_lines[i,0]].data)/ \
                          float(self.emission_lines[i, 2])
            center[:, i] = float(self.emission_lines[i,1]) * \
                            (1.+incat[self.redshift])
            sigmas[:,i] = widths * float(self.emission_lines[i,1]) * \
                          1000. / 299792458. * (1.+incat[self.redshift])
        emspec = np.sum(1./np.sqrt(2.*np.pi)/sigmas[:, :, np.newaxis] * \
                        np.exp(-(((self.wavelength[np.newaxis,np.newaxis:] -
                                   center[:, :, np.newaxis])/
                                  sigmas[:,:,np.newaxis])**2)/2) *
                        fluxes[:,:, np.newaxis], axis=1)

        result = BasicSpectrum(x=self.wavelength * u.Angstrom,
                               y=result.flam(unit=UNIT_SPECLIB)
                                 + emspec*UNIT_SPECLIB)
        result.set_extinction(self.extinction, Rv=3.1)
        if self.mw_av is not None:
            result.apply_galactic_extinction(incat[self.mw_av].data)
        elif self.mw_ebv is not None:
            result.apply_galactic_extinction(3.1*incat[self.mw_ebv].data)
        else:
            raise ValueError("No MW extinction set")
        return result


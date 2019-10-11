# photometry

The photometry package is a simple synthetic photometry package.

The emphasis has been put on speed (as much as I could) rather than on
economy of memory. For example, all the data necessary to interpolate
passbands or spectra are computed once (or rather each time the
internal representation changes) and stored in memory.

There was a slight emphasis on (sub)mm wave astronomy as this package
as been written during my participation to the commissioning of NIKA2,
the new 1mm and 2mm camera at the IRAM 30m telescope, but I am using now
to with my work on the Euclid (visible and near-infrared) where it
is used extensively for the Science Perfomance Verification. Yet, it can
be used in any wavelength domain (except the high energy, as I work with
frequencies or wavelengths, not energies).

## Dependencies
- python 3
- numpy
- scipy
- astropy
- BeautifulSoup

## Installation
```
git clone https://github.com/haussel/photometry.git
```
and make sure the `photometry/` directory is in your PYTHONPATH.

All the passband data are provided with the distribution, but some
spectral libraries are not, as they are a bit on the heavy side. If you
need these, you will have to download them yourselves and let
`photometry` know where they are located on your computer. This is done
in the `config.py` file that you have to edit:
```
STELLAR_LIBRARY_DIR = '/path/to/the/top/directory/of/stellar/libraries'
GALAXY_LIBRARY_DIR = '/path/to/the/top/directory/of/galaxy/libraries'
```


## Documentation
the `notebooks/` directory contains a few notebooks to get you started.

## Provides

`photometry` contains the following files, all imported by default:

- `config`:    define global environment variables
- `phottools`: some useful functions and constants used accross various
               submodules, plus the `PhotometryHeader` and
               `PhotometryInterpolator` classes.
- `photcurve`: the `PhotCurve` class that is used to represent passbands
               and spectra.
- `passband`:  the `Passband` class and the `tophat()` function.

- `spectrum`:  the `BasicSpectrum`, `GalaxySpectrum` and `StellarLibrary`
               classes.
- `standards`: some convenience function to obtain classical spectra,
               including various Vega spectra and the BaSeL 2.2 and
               Pickles (1998) stellar libraries.
- `extinction`: Milky Way extinction in the UV-IR range. Provide the
                ExtinctionLaw, CCMExtinction and ODonnellExtinction
                classes.
- `euclidtu`: Euclid specific functions and classes to use the stellar
              and galaxy libraries used in OU-SIM True Universe.
- `atmosphere`:  defines the IramAtmosphere class to handle GILDAS
                 atmosphere models, and the Atmosphere class for
                 atmospheric transmission.
- `irampassband`: defines the IramPassband class, to handle varying
                  opacity and emission of the sky in the sub-mm.
- `planets`: define the GiantPlanet class to compute the flux of
             Uranus and Neptune and the Mars class for... Mars.


## Usage

A typical session would go as this: Read a passband from a file
*.pb. See in the photometry/data/passbands directory for some
examples. They are grouped per instruments in subdirectories.

```
import photometry as pt
w1 = pt.Passband(file='W1.WISE.pb')
```

Read a spectrum. Alternatively, the data directory contains a few
classical standard spectra as well as some libraries.  For example, to
get the Vega spectrum used by Jarrett et al. (2011) for the
calibration of the WISE data:

```
vega = pt.vega_cohen_1992()
```
To get the AB magnitude of Vega in the W1 passband:

```
w1.mag_ab(Vega)
```

The notebook directory provides more examples of usage.
 - nika2_primary_calibrator: run this notebook to provide the primary 
 calibrator fluxes  for a NIKA2 run.
 - passband_creation: how to add a passband to the photometry package.
 - planet_fluxes: how to use the GiantPlanet and Mars class to derive the 
 flux of a planet at (sub-)mm wavelengths
 -  submm-atmosphere: how to work with the IramAtmosphere and IramPassband 
 classes.

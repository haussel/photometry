# photometry

The photometry package is a simple synthetic photometry package.

The emphasis has been put on speed (as much as I could) rather than on economy
of memory. For example, all the data necessary to interpolate passbands or
spectra are computed once (or rather each time the internal representation
changes) and stored in memory.

## Dependencies
- numpy
- scipy
- astropy
- pyephem for the planet spectra

## Usage

A typical session would go as this:
Read a passband from a file *.pb. See in the photometry/data/passbands
directory for some examples. They are grouped per instruments in subdirectories. 

```
import photometry as pt
w1 = pt.Passband(file='W1.WISE.pb')
```
Read a spectrum. Alternatively, the data directory contains a few classical
standard spectra as well as some libraries.
For example, to get the Vega spectrum used by Jarrett et al. (2011) for the
calibration of the WISE data:
```
vega = pt.vega_cohen_1992()
```
To get the AB magnitude of Vega in the W1 passband:

```
w1.mag_ab(Vega)
```

## Provides

It contains the following submodules, all imported by default.
- phottools:      some useful functions and constants used accross various
                  submodules
- passband:       defines the Passband, PassbandHeader and PassbandInterpolator
                  classes
- spectrum:       defines the BasicSpectrum and the SpectrumInterpolator classes
- standards:      define some convenience function to obtain classical spectra.
- atmosphere:     defines the IramAtmosphere class to handle GILDAS atmosphere models
- irampassband :  defines the IramPassband class

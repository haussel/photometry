"""
Provides the PhotCurve class, the class inherited by the Passband,
BasicSpectrum and Atmosphere classes.
"""
__author__ = "Herve Aussel"

import numpy as np
from astropy.table import Table
from astropy import units as u
from scipy import integrate
from .phottools import velc, quantity_1darray, ndarray_1darray, \
    ndarray_2darray, is_frequency, is_wavelength, nu_unit, lam_unit, \
    read_photometry_file, write_photometry_file, PhotometryInterpolator, \
    PhotometryHeader


class PhotCurve:
    """
    A class to hold a set of curves as a function of wavelength or frequency.

    Attributes:
    -----------
    file: str
        the filename containing the curves
    header: PhotometryHeader
        the header of the file.
    x: ndarray - 1D
        The common abscissa of the set of curves.
    x_si_unit: astropy.units.Quantity
        the unit used to represent x internally. SI units are used, i.e. u.m
        for wavelength and u.Hz for frequencies.
    org_x_type: str
        the original type of abscissa: can be 'lam' or 'nu'.
    org_x_unit: astropy.units.Quantity
        the original unit of x.
    y: ndarray 1D or 2D
        the ordinates of the curves. if y has more than one dimension,
        the last axis must have the same number of elements as x.
    org_y_unit: astropy.units.Quantity
        the original unit of y.
    nb: int
        the number of curves: 1 if y is 1D, and y.shape[0] if y is 2D.
    is_lam: bool
        True if x is a wavelength, False if x is a frequency
    is_nu: bool
        True if x is a frequency, False if x is wavelength
    interpolate_method: str
        interpolation method
    interpolate_set: bool
        True if the interpolation is ready.
    extrapolate: str
        'yes': the curves can be extrapolated
        'no': the curves cannot be extrapolated, if an extrapolation is
        requested, the extrapolated values will be set to NaN
        'zero' : extrapolated values are set to zero
    positive: bool
        If True, any interpolated value that would lead to a negative result
        is set to zero
    interpolate: function
        the function that performs the interpolation
    integration_method: str
        the method to integrate the curve
    integrate_set: bool
        True if the integration is setup.
    integrate: function
        the function that performs the integration
    ready: bool
    intialized: bool
    yfactor: float or ndarray
        the scaling factor in y that has been applied to the curve
    xshift: float
        the multiplicative shift in x that has been applied to the curve
    """
    def __init__(self, file=None, x=None, y=None, header=None,
                 table=None, colname_x=None, colnames_y=None,
                 x_unit=None, interpolation=None, extrapolate=None,
                 positive=True, integration=None):
        """
        PhotCurve initialization. Possible initializations are:

        Initialization from a photometry file (understandable by the phottools
        read_photometry_file function)
        >>> curve = PhotCurve(file=file)

        Initialization from a table. In this case, the first column is
        considered as the x values, the remaining columns the y values. In
        this case, the x column MUST have a unit.
        >>> curve = PhotCurve(table=table)

        Initialization from a table, specifying the x and y columns. In these
        cases, the x column MUST have a unit.
        >>> curve = PhotCurve(table=table, colname_x='colx', colnames_y='coly')
        >>> curve = PhotCurve(table=table, colname_x='colx',
                                           colnames_y=['coly1', 'coly2'])

        If the table x column has no unit, and if the header has a unit,
        passing the header will work:
        >>> curve = PhotCurve(table=table, header=header)
        Same passing explictely x_unit
        >>> curve =  PhotCurve(table=table, x_unit = u.micron)

        Initialization from quantities:
        >>> curve = PhotCurve(x=x, y=y)

        Initialisation from ndarray, passing the header to get x_unit
        >>> curve = PhotCurve(x=x, y=y, header=header)
        Initialisation from ndarray, passing x_unit explictely
        >>> curve = PhotCurve(x=x, y=y, x_unit=x_unit)

        Parameters
        ----------
        file: str
            filename containing the curves
        x: astropy.unit.Quantity or numpy.ndarray
            array of frequencies or wavelengths. If x is an ndarray, the header
            must be provided to know what are the units.
        y:  numpy.ndarray
            array of values as a function of x.
        header: PhotometryHeader or dict or array of str
            the header of the curve
        table: astropy.table.Table
            table containing x and y values
        colname_x: str
            name of the table column containing the x values
        colnames_y: str or list of str
            name or names of the column(s) containing the y values
        x_unit: astropy.units.Quantity
            unit of x
        interpolation: str
            defines the interpolation mehod. Defaulted to 'quadratic'. Other
            methods are 'nearest' or 'linear'. See PassbandInterpolator.
        extrapolate: str
            'yes', 'no', or 'zero'
        positive: bool
            If True, prevent negative interpolation.
        integration: str
            defines the integration method. Can be 'trapezoidal' or 'simpson'

        Methods:
        --------
        set_interpolation(interpolation [,extrapolate, positive]):
            sets the interpolation method
        set_integration(method):
            select the integration method
        scale(factor):
            scale the y by factor
        unscale():
            remove all scalings previously applied
        shift(factor):
            shift the x axis by a multiplicative factor.
        unshift():
            remove all previously applied shift
        nu([unit]):
            returns x as frequency in unit (or in Hz if unspecified)
        lam([unit]):
            returns y as a wavelength in unit (or in m if unspecified)
        in_nu([reinterpolate]):
            convert the x to frequency. Do not reinterpolate is requested.
        in_lam([reinterpolate]):
            convert the x to wavelength. Do not reinterpolate is requested.
        interpolate(x):
            interpolate curves at x
        integrate(y, x):
            computes the integral of y over x.

        """
        self.file = None
        self.header = PhotometryHeader()
        self.x = None
        self.x_si_unit = None
        self.org_x_type = None
        self.org_x_unit = None
        self.y = None
        self.org_y_unit = None
        self.nb = None
        self.is_lam = False
        self.is_nu = False
        self.interpolate_method = None
        self.interpolate_set = None
        self.extrapolate = None
        self.positive = None
        self.interpolate = None
        self.integrate_set = None
        self.integration_method = None
        self.integrate = None
        self.intialized = False
        self.ready = False
        self.yfactor = 1.0
        self.xshift = 1.0


        if file is not None:
            if (x is None) and (y is None) and (table is None):
                self.initialized = self._read(file)
            else:
                raise ValueError("Cannot set both file and x, y or table "
                                 "values")
        elif table is not None:
            if (x is None) and (y is None):
                self.initialized = self._init_from_table(table,
                                                         colname_x=colname_x,
                                                         colnames_y=colnames_y,
                                                         header=header,
                                                         x_unit=x_unit)
            else:
                raise ValueError("Cannot set both table and x, y values")
        elif (x is not None) and (y is not None) and (header is not None):
                self.initialized = self._init_from_header(x, y, header)
        elif (x is not None) and (y is not None) and (x_unit is None):
            self.initialized = self._init_from_quantities(x, y, header=header)
        elif (x is not None) and (y is not None) and (x_unit is not None):
            self.initialized = self._init_from_ndarray(x, x_unit, y,
                                                       header=header)
        else:
            self.initialized = False

        if self.initialized:
            self.set_interpolation(interpolation, extrapolate=extrapolate,
                                   positive=positive)
            self.set_integration(integration)
            if self.interpolate_set and self.integrate_set:
                self.ready = True

    def _init_from_quantities(self, x, y, header=None):
        # check x
        msg = quantity_1darray(x)
        if msg is not None:
            raise ValueError("`x`" + msg)
        # check y
        msg = ndarray_2darray(y, length=len(x), other='x')
        if msg is not None:
            raise ValueError("`y`" + msg)
        if is_frequency(x.unit):
            self.is_nu = True
            self.is_lam = False
            self.x_si_unit = nu_unit
            self.org_x_type = 'nu'
            self.org_x_unit = x.unit
        elif is_wavelength(x.unit):
            self.is_lam = True
            self.is_nu = False
            self.x_si_unit = lam_unit
            self.org_x_type = 'lam'
            self.org_x_unit = x.unit
        else:
            raise ValueError('invalid unit {} for input quantity `x`'.
                             format(x.unit))
        xv = x.to(self.x_si_unit).value
        idx = np.argsort(xv)
        self.x = xv[idx]
        if isinstance(y, u.Quantity):
            self.org_y_unit = y.unit
            yv = y.value
        else:
            self.org_y_unit = None
            yv = y
        if y.ndim == 2:
            self.nb = y.shape[0]
            self.y = yv[:, idx]
        else:
            self.nb = 1
            self.y = yv[idx]

        idx = np.where(self.y < 0)
        if len(idx[0]) > 0:
            print("Negative values have been set to zero")
            self.y[idx] = 0.

        # copy the header values
        if header is not None:
            if isinstance(header, PhotometryHeader):
                self.header = header
            elif isinstance(header, dict):
                self.header.import_dict(header)
            elif isinstance(header, list):
                for elem in header:
                    self.header.import_line(elem)
            else:
                raise ValueError('Invalid type for header')
        return True

    def _init_from_header(self, x, y, header):
        # ingest the header
        if not isinstance(header, PhotometryHeader):
            phd = PhotometryHeader()
            if isinstance(header, dict):
                phd.import_dict(header)
            elif isinstance(header, list):
                for elem in header:
                    phd.import_line(elem)
            else:
                raise ValueError('Invalid type for header')
        else:
            phd = header
        # xunit
        if 'xunit' not in phd:
            raise ValueError('Missing keyword xunit in header')
        try:
            x_unit = u.Unit(phd['xunit'])
        except:
            raise ValueError('Invalid `xunit` {}'.format(phd['xunit']))
        # check x
        msg = ndarray_1darray(x)
        if msg is not None:
            raise ValueError("`x`" + msg)
        # check y
        msg = ndarray_1darray(y, length=len(x), other='x')
        if msg is not None:
            raise ValueError("`y`" + msg)

        if isinstance(x, u.Quantity):
            if x.unit is not x_unit:
                raise ValueError('`x` and `xunit` key in header are different')
            xv = x
        else:
            xv = x * x_unit
        return self._init_from_quantities(xv, y, header=phd)

    def _init_from_table(self, data, colname_x=None, colnames_y=None,
                         header=None, x_unit=None):
        """
        Initialize from a table.

        Parameters
        ----------
        data: astropy.table.Table
            table containing the data
        colname_x: str
            name of the column containing the frequencies or wavelengths.
            If not given, try with the first column
        colnames_y: str
            name(s) of the column(s) containing the fluxes.
            If not given, try with all the columns except the first
        header: PhotometryHeader or dict or array of str
            the header of the curve
        x_unit: astropy.units.Unit
            the unit of column x if not present in the table.
        """
        if not isinstance(data, Table):
            raise ValueError('`data` is not a Table')

        if header is not None:
            if not isinstance(header, PhotometryHeader):
                phd = PhotometryHeader()
                if isinstance(header, dict):
                    phd.import_dict(header)
                elif isinstance(header, list):
                    for elem in header:
                        phd.import_line(elem)
                else:
                    raise ValueError('Invalid type for header')
            else:
                phd = header
            hasheader = True
        else:
            hasheader = False

        colnames = data.colnames
        if len(colnames) < 2:
            raise ValueError("data must have at least two columns")
        if colname_x is None:
            colname_x = colnames[0]
        if colnames_y is None:
            colnames_y = colnames[1:]

        if colname_x in colnames:
            if hasattr(data[colname_x], 'unit'):
                x = data[colname_x].data * data[colname_x].unit
            elif hasheader:
                if 'xunit' not in phd:
                    raise ValueError('Missing keyword xunit in header')
                try:
                    x_unit = u.Unit(phd['xunit'])
                except:
                    raise ValueError('Invalid `xunit` in header {}'.
                                     format(phd['xunit']))
                x = data[colname_x].data * x_unit
            elif x_unit is not None:
                try:
                    x = data[colname_x].data * x_unit
                except:
                    raise ValueError('Invalid `x_unit` {}'.format(x_unit))
            else:
                raise ValueError("No unit for column x '{}'".format(colname_x))
        else:
            raise ValueError("Column `{}` not found in Table".format(colname_x))

        foundone = False
        foundunit = None
        for colname_y in colnames_y:
            if colname_y in colnames:
                if hasattr(data[colname_y], 'unit'):
                    if foundone:
                        if data[colname_y].unit == foundunit:
                            y = np.vstack((y, data[colname_y].data))
                        else:
                            raise ValueError("column {} has not the same "
                                             "unit ({}) as the others "
                                             "cols ({}) ".
                                             format(colname_y,
                                                    data[colname_y].unit,
                                                    foundunit))
                    else:
                        foundone = True
                        foundunit = data[colname_y].unit
                        y = data[colname_y].data

        return self._init_from_quantities(x, y, header=header)

    def _init_from_ndarray(self, x, x_unit, y, header=None):
        try:
            xu = x * x_unit
        except:
            raise ValueError('Invalid unit for x: {}'.format(x_unit))
        return self._init_from_quantities(x=xu, y=y, header=header)

    def _read(self, filename):
        (values, sheader) = read_photometry_file(filename)
        self.file = filename
        # TODO: uses more than one column...
        return self._init_from_header(values['x'], values['y'], sheader)

    def write(self, xunit, filename=None, dirname=None, overwrite=False,
              xfmt=':> 0.6f', yfmt=':>0.6f'):
        return write_photometry_file(self, xunit, filename=filename,
                                     dirname=dirname, overwrite=overwrite,
                                     xfmt=xfmt, yfmt=yfmt)

    def __str__(self):
        result = "{}".format(self.header)
        result += "\nInternal settings:\n"
        result += "    interpolation method: {} / ready ? {}\n".\
            format(self.interpolate_method, self.interpolate_set)
        result += "    integration method: {} / ready ? {}\n".format(
            self.integration_method, self.integrate_set)
        return result



    def set_interpolation(self, interpolation, extrapolate='no', positive=True):
        """
        Sets the interpolation method of the curve.


        Parameters
        ----------
        interpolation: str
            the chosen interpolation method. Can be any offered by the
            PhotometryInterpolator class.
        extrapolate: str
            'no', 'yes', 'zero', see PhotometryInterpolator
        positive: bool
            if True, negative values are set to 0

        Returns
        -------
        bool: True if succesful

        Effects
        -------
        Sets the value of the following attributes:
         - interpolate: the function performing the interpolation
         - interpolate_method: the name of the method
         - interpolate_set: flag indicating whether the passband is ready to
           interpolate.
        """
        try:
            self.interpolate = PhotometryInterpolator(self.x, self.y,
                                                      interpolation,
                                                      positive=positive,
                                                      extrapolate=extrapolate)
            result = True
        except:
            print("Problem setting interpolation with {}".format(interpolation))
            result = False
        if result is True:
            self.interpolate_method = interpolation
            self.interpolate_set = True
            self.extrapolate = extrapolate
            self.positive = positive
        return result

    def set_integration(self, method_name):
        """
        select the integration method of the curve

        Parameters
        ----------
        method_name: str
          Can be any of the two methods:
          - trapezoidal : use numpy.trapz
          - simpson     : use numpy.simps
        """
        result = True
        try:
            if method_name == 'trapezoidal':
                self.integrate = self._integrate_trapezoidal
            elif method_name == 'simpson':
                self.integrate = self._integrate_simpson
            else:
#                print("invalid integration method {}".format(method_name))
                result = False
        except:
            print("invalid integration method {}".format(method_name))
            result = False
        if result is True:
            self.integration_method = method_name
            self.integrate_set = True
        else:
            self.integration_method = None
            self.integrate_set = False
        return result

    def _integrate_trapezoidal(self, y, x):
        return np.trapz(y, x=x, axis=-1)

    def _integrate_simpson(self, y, x):
        return integrate.simps(y, x=x, axis=-1)

    def scale(self, factor):
        """
        Scale the curve in the sense y *= factor. If factor is a quantity,
        only the value is used and the unit is dropped.

        Parameter
        ---------
        factor: float or numpy.array of nb curves values
          the scaling factor(s)

        Returns
        -------
        None
        """
        # Ensure factor has no unit
        if isinstance(factor, u.Quantity):
            wfactor = factor.value
        else:
            wfactor = factor
        msg = ndarray_1darray(factor, length=self.nb)
        if msg is None:
            # factor is a 1D array
            self.y = self.y * wfactor[:, np.newaxis]
        else:
            self.y *= wfactor
        self.yfactor *= factor
        self.set_interpolation(self.interpolate_method,
                               extrapolate=self.extrapolate,
                               positive=self.positive)
        return None

    def unscale(self):
        """
        Remove any scaling
        """
        factor = 1./self.yfactor
        self.scale(factor)

    def shift(self, factor):
        """
        Shift the curve in the sense x *= factor

        Parameter
        ---------
        factor: float
            value by which x are multiplied

        Returns
        -------
        None
        """
        # ensure factor has no unit.
        if isinstance(factor, u.Quantity):
            wfactor = factor.value
        else:
            wfactor = factor
        self.x *= wfactor
        self.xshift *= wfactor
        self.set_interpolation(self.interpolate_method,
                               extrapolate=self.extrapolate,
                               positive=self.positive)
        return None

    def unshift(self):
        """
        Remove the shift in x
        """
        factor = 1./self.xshift
        self.shift(factor)

    def nu(self, unit=None):
        """
        Returns the frequency array of the curve in Hz or in requested unit

        Parameters
        ----------
        unit: astropy.unit
            the output unit. If None, returns frequencies in Hz

        Returns
        -------
        astropy.units.Quantity
            array of frequencies
        """
        if self.is_lam:
            if unit is None:
                return velc / self.x
            else:
                return (velc / self.x * nu_unit).to(unit)
        else:
            if unit is None:
                return self.x
            else:
                return (self.x * nu_unit).to(unit)

    def lam(self, unit=None):
        """
        Returns the wavelength array of the curve in m or in requested unit

        Parameters
        ----------
        unit: astropy.unit
            the output unit. If None, returns wavelengths in m

        Returns
        -------
        astropy.units.Quantity
            array of wavelengths

        """
        if self.is_lam:
            if unit is None:
                return self.x
            else:
                return (self.x * lam_unit).to(unit)
        else:
            if unit is None:
                return velc / self.x
            else:
                return (velc / self.x * lam_unit).to(unit)

    def in_nu(self, reinterpolate=True):
        """
        Set the curve to 'nu' type

        Parameters
        ----------
        reinterpolate: bool
            If True, the interpolation is recomputed.
        """
        if self.is_lam:
            self.x = velc / self.x[::-1]
            if self.nb > 1:
                self.y = self.y[:, ::-1]
            else:
                self.y = self.y[::-1]
            self.is_lam = False
            self.is_nu = True
            self.x_si_unit = nu_unit
            self.xshift = 1./self.xshift
            if reinterpolate:
                self.interpolate_set = self.set_interpolation(
                    self.interpolate_method,
                    extrapolate=self.extrapolate,
                    positive=self.positive)
            else:
                self.interpolate = None
                self.interpolate_set = False
        return None

    def in_lam(self, reinterpolate=True):
        """
        Set the curve to 'lam' type

        Parameters
        ----------
        reinterpolate: bool
            If True, the interpolation is recomputed.
        """
        if self.is_nu:
            self.x = velc / self.x[::-1]
            if self.nb > 1:
                self.y = self.y[:, ::-1]
            else:
                self.y = self.y[::-1]
            self.is_lam = True
            self.is_nu = False
            self.x_si_unit = lam_unit
            self.xshift = 1./self.xshift
            if reinterpolate:
                self.interpolate_set = self.set_interpolation(
                    self.interpolate_method,
                    extrapolate=self.extrapolate,
                    positive=self.positive)
            else:
                self.interpolate = None
                self.interpolate_set = False
        return None

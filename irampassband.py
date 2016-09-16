__author__ = 'haussel'
from .atmosphere import IramAtmosphere
from .spectrum import BasicSpectrum
from .passband import Passband, PassbandInterpolator

class IramPassband(Passband,IramAtmosphere):
    def __init__(self, file=None, x=None, y=None, xref=None, ytype=None,
                 header=None, location='both_spectrum_and_passband',
                 interpolation='quadratic', integration='trapezoidal',
                 model=2009, observatory='iram30m', profile='midlatwinter',
                 gildas_atm_file=None):
        IramAtmosphere.__init__(self,model=model, observatory=observatory,
                                profile=profile,
                                gildas_atm_file=gildas_atm_file)
        Passband.__init__(self, file=file, x=x, y=y, xref=xref, ytype=ytype,
                          header=header,
                          location=location, interpolation=interpolation,
                          integration=integration)

        if self.is_lam:
            raise ValueError('IramPassband work in frequency only !')
        self.nu = ((self.nu()).to(self.x_si_unit)).value
        self.x_org = self.x.copy()
        self.y_org = self.y.copy()
        self.interp_org = PassbandInterpolator(self.x_org, self.y_org, kind='quadratic')
        self.elevation = None
        self.currenttrans = None

    def set_p_t(self, pressure, temperature):
        self.select_grid(temperature=temperature, pressure=pressure)

    def set_tau225(self, tau_225):
        self.set_tau_225(tau_225=tau_225)
        self.tau_225 = tau_225

    def set_mmH2O(self, mmH2O):
        self.set_mm_H2O(mmH2O=mmH2O)
        self.mmH2O = mmH2O

    def __str__(self):
        result = "**** Atmosphere: \n" + super(IramAtmosphere, __str__)
        result = result + "**** Passband: \n" + super(Passband.__str__)
        return result 


    def set_elevation(self, elevation):
        if self.is_lam:
            self.in_nu()
        trans = self.transmission(elevation)
        interp = PassbandInterpolator(self.nu, trans, kind=self.interpolate_method)
        self.x = self.location(self.nu, self.x_org)
        self.currenttrans = interp(self.x)
        self.y = self.interp_org(self.x) * self.currenttrans
        self.set_interpolation(self.interpolate_method)
        self.elevation = elevation


import numpy as np
from astropy.table import Table
from astropy import units as u
import photometry as pt

TABLE3TONRYETAL = 'apj425122t3_mrt.txt'
PS1BANDS = ['gp1', 'rp1', 'ip1', 'zp1', 'yp1', 'wp1']
PS1LEFF = np.array([481, 617, 752, 866, 962, 608]) * u.nm

def build(infile=TABLE3TONRYETAL):
    try:
        t = Table.read(infile, format='ascii.cds')
    except:
        raise IOError("Cannot read input passband file: {}\n "
                       "This file is the machine readable form of table 3 of Tonry et al. (2012), ApJ 750, 99\n"
                       "http://adsabs.harvard.edu/abs/2012ApJ...750...99T\n"
                       "and can be obtained from the Astrophysical Journal web site.".format(infile))
    hd = pt.PassbandHeader()
    hd.add_card_value('instrument', 'PS1')
    hd.add_card_value('system', 'AB')
    hd.add_card_value('xtype', 'wavelength')
    hd.add_card_value('xunit', t['Wave'].unit)
    hd.add_card_value('ytype', 'qe')
    hd.add_card_value('description', 'Full system QE')
    hd.add_card_value('url', 'http://adsabs.harvard.edu/abs/2012ApJ...750...99T')
    hd.add_card_value('reference', 'Tonry et al. (2012), ApJ 750, 99')
    hd.add_card_value('comment', 'Passband from table 3 of the reference')
    hd.add_card_value('comment', 'Lambda_eff from table 4 of the reference')
    for i, band in enumerate(PS1BANDS):
        print("Creating passband {}".format(band))
        hd.edit_card_value('filter', band)
        hd.edit_card_value('xref', PS1LEFF[i])
        hd.edit_card_value('file', '{}.PS1.pb'.format(band))
        pb = pt.Passband(x=t['Wave'].data, y=t[band].data, header=hd)
        pb.write(t['Wave'].unit, overwrite=True)
    print('All done')

if __name__ == '__main__':
    build()


import numpy as np
import os
from astropy.table import Table
from astropy import units as u
import photometry as pt

WISEBANDS = ['W1', 'W2', 'W3', 'W4']
WISELAMREF = np.array([3.3526, 4.6028, 11.5608, 22.0883]) * u.micron

def build():
    hd = pt.PassbandHeader()
    hd.add_card_value('instrument', 'WISE')
    hd.add_card_value('xtype', 'wavelength')
    hd.add_card_value('xunit', u.micron)
    hd.add_card_value('ytype', 'rsr')
    hd.add_card_value('description', 'Telescope + Instrument')
    hd.add_card_value('url', 'http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html#WISEZMA')
    hd.add_card_value('reference', 'Jarett et al. (2011), ApJ 735 112')
    hd.add_card_value('comment', 'Reference wavelength from table 1 of Jarrett et al. (2011)')
    for i, band in enumerate(WISEBANDS):
        print("Creating passband {}".format(band))
        infile = "RSR-{}.EE.txt".format(band)
        try:
            t = Table.read(infile, format='ascii', delimiter='\s', \
                           comment='#', names=['lam', 'rsr', 'unc'])
        except:
            raise IOError("Cannot read input passband file: {}\n "
                          "This file can be obtained from IPAC at\n"
                          "http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec4_3g.html#WISEZMA\n".format(infile))
        hd.add_card_value('comment', 'RSR from the file {} available from WISE '
                          'Explanatory Supplement (see url)'.format(infile))
        hd.edit_card_value('filter', band)
        hd.edit_card_value('xref', WISELAMREF[i])
        hd.edit_card_value('file', '{}.WISE.pb'.format(band))
        pb = pt.Passband(x=t['lam'].data * u.micron, y=t['rsr'].data, header=hd)
        pb.write(u.micron, overwrite=True)
    print('All done')

if __name__ == '__main__':
    build()


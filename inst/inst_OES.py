#! /usr/bin/env python3

## viper - Velocity and IP Estimator
## Copyright (C) Mathias Zechmeister and Jana Koehler
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License along
## with this program; if not, write to the Free Software Foundation, Inc.,
## 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import sys
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.constants import c

from .readmultispec import readmultispec
from .airtovac import airtovac

from .FTS_resample import resample, FTSfits

# see https://github.com/mzechmeister/serval/blob/master/src/inst_FIES.py

location = oes = EarthLocation.from_geodetic(lat=49.91056*u.deg, lon=14.78361*u.deg, height=528*u.m)

oset = '1:30'

ip_guess = {'s': 300_000/67_000/ (2*np.sqrt(2*np.log(2))) }   # convert FHWM resolution to sigma

def Spectrum(filename='', order=None, targ=None):
    hdu = fits.open(filename, ignore_blank=True)[0]
    hdr = hdu.header

    dateobs = hdr['DATE-OBS']+ 'T' + hdr['UT']
    exptime = hdr['EXPTIME']

    midtime = Time(dateobs, format='isot', scale='utc') + exptime * u.s
    bjd = midtime.tdb

   # targdrs = SkyCoord(ra=ra*u.hour, dec=de*u.deg)
    if not targ: 
        #targ = targdrs
        berv = 0
    else:
        berv = targ.radial_velocity_correction(obstime=midtime, location=oes)
        berv = berv.to(u.km/u.s).value

    spec = hdu.data
    spec /= np.nanmean(spec)
    gg = readmultispec(filename, reform=True, quiet=True)
    wave = gg['wavelen']
    wave = airtovac(wave)
    if order is not None:
         wave, spec= wave[order], spec[order]

    pixel = np.arange(spec.size) 
    err = np.ones(spec.size)*0.1
    flag_pixel = 1 * np.isnan(spec) # bad pixel map
 #   b[f>1.5] |= 4 # large flux

    return pixel, wave, spec, err, flag_pixel, bjd, berv


def Tpl(tplname, order=None, targ=None):
    '''Tpl should return barycentric corrected wavelengths'''
    if tplname.endswith('_s1d_A.fits'):
        hdu = fits.open(tplname)[0]
        spec= hdu.data
        h = hdu.header
        wave = h['CRVAL1'] +  h['CDELT1'] * (1. + np.arange(spec.size) - h['CRPIX1'])
        wave = airtovac(wave)
    elif tplname.endswith('1d.fits'):
        hdu = fits.open(tplname, ignore_blank=True)[0]
        hdr = hdu.header
        dateobs = hdr['DATE']
        exptime = hdr['EXPTIME']
        midtime = Time(dateobs, format='isot', scale='utc') + exptime * u.s
        bjd = midtime.tdb

        if not targ: 
            #targ = targdrs
            berv = 0
        else:
            berv = targ.radial_velocity_correction(obstime=midtime, location=oes)
            berv = berv.to(u.km/u.s).value

        spec = hdu.data
        spec /= np.nanmean(spec)
        gg = readmultispec(tplname, reform=True, quiet=True)
        wave = gg['wavelen']
        wave = airtovac(wave)
        wave *= 1 + (berv*u.km/u.s/c).to_value('')
    elif tplname.endswith('_tpl.model'):
        pixel, wave, spec, err, flag_pixel, bjd, berv = Spectrum(tplname, order=order, targ=targ)
    else:
        pixel, wave, spec, err, flag_pixel, bjd, berv = Spectrum(tplname, order=order, targ=targ)
        wave *= 1 + (berv*u.km/u.s/c).to_value('')

    return wave, spec


def FTS(ftsname='lib/oes.fits', dv=100):

    return resample(*FTSfits(ftsname), dv=dv)


def write_fits(wtpl_all, tpl_all, e_all, list_files, file_out):

    file_in = list_files[0]

    # copy header from first fits file 
    hdu = fits.open(file_in, ignore_blank=True)[0]
    f = hdu.data

    # write the template data to the file
    for order in range(1,49,1): 
        if order in tpl_all:
            f[order] = tpl_all[order]
        else:
            f[order] = np.ones(len(f[order]))

    hdu.writeto(file_out+'_tpl.model', overwrite=True)  
  #  hdu.close()  


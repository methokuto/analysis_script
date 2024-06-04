#! /usr/bin/env python
#-------------------------------------------------------------------------------------------
# 
#--------10--------20--------30--------40--------50--------60--------70--------80--------90#

import os
import sys

from grads.ganum import GaNum
import numpy as np

import netCDF4
from netCDF4 import Dataset, chartostring

from scipy.ndimage import gaussian_filter
from datetime import date

#---------------------------------------------------------------
#nt = 145
#ni = 456
#nj = 444
#nk = 60

# ctl info
xini = 120.00000
xinc = 0.0300000
yini = 15.50000
yinc = 0.0300000

avoid_internal_error = 0.01 # tentative approach to avoid mismatch of array size between data1 and data2.

tim_start = 5
tim_end   = 217

# for YDK
datadir = '/home1/kitano/cress/work/nanmadol'
indir   = datadir+'/work_N300/out'  # input netcdf file dir

basename_3d = 'nanmadol.dmp'
basename_2d = 'nanmadol.mon'

reso     = 3.0
title_h  = 'CReSS_IV'
case     = 'nanmadol2022.N300'
ver      = 'CReSS-NHM4ICE-AEROSOL-release_20220331_cti'
footer   = '.t'+str(tim_start)+'-'+str(tim_end)

#anal_region_xs = 136.5
#anal_region_xe = 144.0
#anal_region_ys = 28.0
#anal_region_ye = 33.5

GF_sigma = 1

#---------------------------------------------------------------
outdir  = '.'
outfile = 'track.'+case+footer+'.nc'

#-------------------------------------------------------------------------------------------
def land_masking(var, alt): # var: target variable, alt: altitude [m]

   thres = 60 # the minimum threshold for land [m]
   if( thres < np.amin(alt) ):
      sys.exit("ERROR: threshold is less than the minimum of altitude. [land_masking]")

   nj, ni = var.shape
   m_var = np.zeros( (nj,ni), dtype='float32' )

   for j in range( nj ):
      m_var[j,:] = np.where( alt[j,:] > thres, 9.999e9, var[j,:] )

   return( m_var )

#---------------------------------------------------------------

# open grads binary file
#-----------------------------------------------
ga1 = GaNum(Bin='grads -b')
ga1.open(indir+'/'+basename_3d+'.ctl')
#ga1('q file')
ga1('set t '+str(tim_start))
#ga1('set lon '+str(anal_region_xs)+' '+str(anal_region_xe))
#ga1('set lat '+str(anal_region_ys)+' '+str(anal_region_ye))

ga1('set z 1')

izph = ga1.exp('zph')

#del ga1
print( 'complete: load grads binary data (3D)' )

#-----------------------------------------------
ga2 = GaNum(Bin='grads -b')
ga2.open(indir+'/'+basename_2d+'.ctl')
#ga2('q file')
ga2('set t '+str(tim_start)+' '+str(tim_end))
#ga2('set lon '+str(anal_region_xs)+' '+str(anal_region_xe-avoid_internal_error))
#ga2('set lat '+str(anal_region_ys)+' '+str(anal_region_ye))

ps = ga2.exp('ps')

#del ga2
print( 'complete: load grads binary data (2D)' )


# check array shapes
nj, ni = izph.shape
nt, nj2, ni2 = ps.shape
print( nt, nj, nj2, ni, ni2 )
if( ni2 != ni ):
   sys.exit("ni and ni2 are not consistent!")
if( nj2 != nj ):
   sys.exit("nj and nj2 are not consistent!")

# generate 2 2d grids for the x & y bounds
ilon = np.arange(xini, (ni)*xinc + xini, xinc)   # tentative
ilat = np.arange(yini, (nj)*yinc + yini, yinc)   # tentative
print( ilon.shape, ilat.shape )


nc_time   = np.zeros( (nt), dtype='int32' )
nc_idx_cx = np.zeros( (nt), dtype='int32' )
nc_idx_cy = np.zeros( (nt), dtype='int32' )
nc_lon_cx = np.zeros( (nt), dtype='float32' )
nc_lat_cy = np.zeros( (nt), dtype='float32' )
nc_mps    = np.zeros( (nt), dtype='float32' )
nc_mfps   = np.zeros( (nt), dtype='float32' )

# time loop
step = tim_start
for tim in range(nt):

   # Smoothing by Gaussian filter
   fps = gaussian_filter(ps[tim,:,:], GF_sigma)

   # Masking by altitude
   fps = land_masking(fps[:,:], izph[:,:])

   # search a location of the ps-minimum
   cj, ci = np.unravel_index(np.argmin(fps), fps.shape)
   print( 'time step:', tim, 'tc-center:', cj, ci, fps[cj,ci], (fps[cj,ci] - np.amin(fps)), 'min-ps:', ps[tim,cj,ci] )

   nc_time[tim]   = step
   nc_idx_cx[tim] = ci
   nc_idx_cy[tim] = cj
   nc_lon_cx[tim] = ilon[ci]
   nc_lat_cy[tim] = ilat[cj]
   nc_mps[tim]    = ps[tim,cj,ci]
   nc_mfps[tim]   = fps[cj,ci]

   step += 1

#sys.exit("DEBUG")

# ------------------------------------------------------------------------

# dump data
# ---------------------------------------------------------------------
today = date.today()

nc = netCDF4.Dataset( outdir+'/'+outfile, 'w', format='NETCDF4_CLASSIC' )

nc.createDimension('nx', ni)
nc.createDimension('ny', nj)
nc.createDimension('time', None)

nc.description = 'Case: '+case
nc.model = ver
nc.history = "Created " + today.strftime("%d/%m/%y")

time = nc.createVariable('time', 'i4', ('time'))
time.long_name = 'time step number of history output'
time.units = '#'

lon = nc.createVariable('lon', 'f4', ('nx'))
lon.long_name = 'longitude (west to east)'
lon.units = 'deg'

lat = nc.createVariable('lat', 'f4', ('ny'))
lat.long_name = 'latitude (south to north)'
lat.units = 'deg'

zph = nc.createVariable('zph', 'f4', ('ny','nx'))
zph.long_name = 'geopotential height (bottom to top)'
zph.units = 'm'

idx_cx = nc.createVariable('idx_cx', 'i4', ('time'))
idx_cx.long_name = 'index of tc center for x-axis'
idx_cx.units = '-'

idx_cy = nc.createVariable('idx_cy', 'i4', ('time'))
idx_cy.long_name = 'index of tc center for y-axis'
idx_cy.units = '-'

lon_cx = nc.createVariable('lon_cx', 'f4', ('time'))
lon_cx.long_name = 'longitude of tc center'
lon_cx.units = 'deg'

lat_cy = nc.createVariable('lat_cy', 'f4', ('time'))
lat_cy.long_name = 'latitude of tc center'
lat_cy.units = 'deg'

mps = nc.createVariable('mps', 'f4', ('time'))
mps.long_name = 'minimum of surface pressure at tc center'
mps.units = 'Pa'

mfps = nc.createVariable('mfps', 'f4', ('time'))
mfps.long_name = 'minimum of surface pressure at tc center (filtered)'
mfps.units = 'Pa'

time[:]  = nc_time[:]

lon[:]   = ilon[:]
lat[:]   = ilat[:]
zph[:,:] = izph[:,:]

idx_cx[:] = nc_idx_cx[:]
idx_cy[:] = nc_idx_cy[:]
lon_cx[:] = nc_lon_cx[:]
lat_cy[:] = nc_lat_cy[:]

mps[:]    = nc_mps[:]
mfps[:]   = nc_mfps[:]

nc.close()

print( "finish." )

#--------10--------20--------30--------40--------50--------60--------70--------80--------90#
#EOF

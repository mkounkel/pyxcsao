import numpy as np
from astropy.table import Table
from astropy.io import fits
import warnings
import glob
from PyAstronomy import pyasl
warnings.filterwarnings('ignore')



def data_loader(data_path,i=0,data_class='boss',laname=None,meta=None):
	if data_class=='boss':
		return load_boss(data_path)
	elif data_class=='boss_merged':
		return load_boss_merged(data_path,i=i)
	elif data_class=='boss_raw':
		return load_boss_raw(data_path,laname,meta,i=i)
	elif data_class=='boss_frame':
		return load_boss_frame(data_path,meta,i=i)
	elif data_class=='boss_cframe':
		return load_boss_cframe(data_path,meta,i=i)
	elif data_class=='lamost':
		return load_lamost(data_path)
	elif data_class=='user':
		return load_user(data_path,laname,meta)
	else:
		raise RuntimeError('Data loader not yet implemented, please supply your own.')
		
	


def load_boss(name):
	hdul = fits.open(name)
	spec = hdul[1].data
	y = hdul[2].data
	x=hdul[0].header
	hdul.close()
	try:
		flux=spec['flux']
		la=10**spec['loglam']
	except:
		flux=spec['FLUX']
		la=10**spec['LOGLAM']

	try:
		meta={'ra'    :x['PLUG_RA'],
	          'dec'   :x['PLUG_DEC'],
	          'objid' :y['CATALOGID'][0],
	          'plate' :x['PLATEID'],
	          'mjd'   :x['MJD'],
	          'fiber' :y['FIBERID'][0],
	          'snr'   :y['SN_MEDIAN_ALL'][0]}
	except:
		meta={'ra'    :x['PLUG_RA'],
	          'dec'   :x['PLUG_DEC'],
	          'objid' :y['OBJID'][0],
	          'plate' :x['PLATEID'],
	          'mjd'   :x['MJD'],
	          'fiber' :y['FIBERID'][0],
	          'snr'   :y['SN_MEDIAN_ALL'][0]}

	return flux,la,meta
	
def load_lamost(name):
	hdul = fits.open(name)
	spec = hdul[0].data
	x=hdul[0].header
	hdul.close()
	flux=spec[0,:]
	la=spec[2,:]
	
	meta={'ra'    :x['RA'],
	      'dec'   :x['DEC'],
	      'objid' :x['OBSID'],
	      'plate' :x['PLANID'],
	      'mjd'   :x['MJD'],
	      'fiber' :str(x['SPID'])+'-'+str(x['FIBERID']),
	      'snr'   :x['SNRR']}

	return flux,la,meta
	
def load_boss_merged(name,i=0):
	flux=Table.read(name)
	la=Table.read('boss_spectra_la.fits')['la']
	
	meta={'ra'    :flux['ra'][i],
	      'dec'   :flux['dec'][i],
	      'objid' :flux['catalogid'][i],
	      'plate' :flux['plate'][i],
	      'mjd'   :flux['mjd'][i],
	      'fiber' :flux['fiberid'][i],
	      'snr'   :flux['snr'][i]}
		
	return flux['flux'][i],la,meta
	
def load_boss_raw(name,laname,meta,i=0):
	hdul = fits.open(name)
	flux = hdul[0].data
	x=hdul[0].header
	hdul.close()
	
	
	hdul = fits.open(laname)
	la = hdul[0].data
	hdul.close()
	
	meta=makemeta(meta)

	return flux[i,:],la[i,:],meta
	
def load_boss_frame(name,meta,i=0):
	hdul = fits.open(name)
	flux = hdul[0].data
	x=hdul[0].header
	hdul.close()
	la = 10**(np.arange(x['NAXIS1'])*x['CD1_1']+x['CRVAL1'])
	
	meta=makemeta(meta)
		
	return flux[i,:],la,meta
	
	
def load_boss_cframe(name,meta,i=0):
	hdul = fits.open(name)
	flux = hdul[0].data
	la=10**(hdul[3].data)
	x=hdul[0].header
	hdul.close()
	
	meta=makemeta(meta)
	meta['mjd']=x['TAI-BEG']/86400.
	meta['fiber']=i+1
	meta['plate']=x['PLATEID']
	meta['alt']=x['ALT']
	meta['az']=x['AZ']
	meta['airmass']=x['AIRMASS']
		
	return flux[i,:],la[i,:],meta
	
def load_user(name,la,meta):

	meta=makemeta(meta)
	      
	return name,la,meta
	
def makemeta(meta):
	if meta==None:
		meta={'ra'    :np.nan,
	      	  'dec'   :np.nan,
	      	  'objid' :np.nan,
	      	  'plate' :np.nan,
	      	  'mjd'   :np.nan,
	      	  'fiber' :np.nan,
	      	  'snr'   :np.nan}
	else:
		if 'ra' not in meta:    meta['ra']=np.nan
		if 'dec' not in meta:   meta['dec']=np.nan
		if 'objid' not in meta: meta['objid']=np.nan
		if 'plate' not in meta: meta['plate']=np.nan
		if 'mjd' not in meta:   meta['mjd']=np.nan
		if 'fiber' not in meta: meta['fiber']=np.nan
		if 'snr' not in meta:   meta['snr']=np.nan
	return meta
	
def correct_blue_lambda(la):
	return la+10**(-5.5888976E-4*la+3.6036503)*la/299792.458
	
	
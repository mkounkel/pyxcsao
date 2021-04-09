import numpy as np
from astropy.io import fits
from PyAstronomy import pyasl
import warnings
import glob
import pickle
warnings.filterwarnings('ignore')


def grid_loader(grid_path,grid_class,laname=None):
	if grid_class=='coelho':
		return load_template_coelho(grid_path)
	elif grid_class=='phoenix':
		return load_template_phoenix(grid_path)
	elif grid_class=='phoenixhires':
		return load_template_phoenix_hires(grid_path,laname)
	elif callable(grid_class):
		return grid_class(grid_path)
	
	else:
		raise RuntimeError('Grid loader not yet implemented, please supply your own.')

def load_template_coelho(name):
	hdul = fits.open(name)
	spec = hdul[0].data
	header=hdul[0].header
	hdul.close()
	la=pyasl.airtovac2(np.arange(50001)*0.2+3000)
	return spec,la,header['TEFF'],header['LOG_G'],header['FEH'],header['AFE']
	
def load_template_phoenix(name,la=None):
	hdul = fits.open(name)
	spec = hdul[0].data
	header=hdul[0].header
	hdul.close()
	la=np.exp(np.arange(212027)*1e-5+8.006368)
	a=np.where((la>3000) & (la<11000))[0]
	return spec[a],la[a],header['PHXTEFF'],header['PHXLOGG'],header['PHXM_H'],header['PHXALPHA']
	
def load_template_phoenix_hires(name,laname):
	hdul = fits.open(name)
	spec = hdul[0].data
	header=hdul[0].header
	hdul.close()
	hdul = fits.open(laname)
	la=hdul[0].data
	a=np.where((la>3000) & (la<11000))[0]
	hdul.close()
	return spec[a],la[a],header['PHXTEFF'],header['PHXLOGG'],header['PHXM_H'],header['PHXALPHA']
	

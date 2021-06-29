import numpy as np
from astropy.table import Table
from astropy.io import fits
import astropy.units as u
from scipy import interpolate
from scipy.fft import fft,ifft,fftshift,fftfreq
from specutils import Spectrum1D
from specutils.fitting import fit_continuum
from specutils.manipulation import SplineInterpolatedResampler
import warnings
import glob
import pickle

from .data import data_loader
from .grid import grid_loader

warnings.filterwarnings('ignore')


class PyXCSAO:
	def __init__(self,st_lambda=None,end_lambda=None,ncols=8192,low_bin=5,top_low=10,top_nrun=125,nrun=255,bell_window=0.05,minvel=-500,maxvel=500,from_spectrum=None,spectrum_num=0,data_class='boss'):
		
		self.ncols=ncols
		self.bell_window=bell_window
		self.low_bin=low_bin
		self.top_low=top_low
		self.top_nrun=top_nrun
		self.nrun=nrun
		self.spline = SplineInterpolatedResampler()
		self.taper=self.taper_spectrum()
		self.taperf=self.taper_FFT()
		self.minvel=minvel
		self.maxvel=maxvel
		
		
		if st_lambda is not None:
			self.st_lambda=st_lambda
		else:
			self.st_lambda=None
		if end_lambda is not None:
			self.end_lambda=end_lambda
		else:
			self.end_lambda=None
		if (self.st_lambda is not None) & (self.end_lambda is not None):
			self.la,self.lag=self.set_lambda()
		if (from_spectrum is not None):
			self.add_spectrum(from_spectrum,i=spectrum_num,data_class=data_class)
		if (self.st_lambda is None) | (self.end_lambda is None):
			raise RuntimeError('Please specify st_lambda & end_lambda, or provide a data spectrum to automatically determine the range.')


	def add_spectrum(self,name,i=0,laname=None,data_class='boss',meta=None,emission=False,clip=True):
		flux,la,meta=data_loader(name,i=i,data_class=data_class,laname=laname,meta=meta)
		if self.st_lambda is None:
			self.st_lambda=np.ceil(min(la))
			if self.end_lambda is not None:
				self.la,self.lag=self.set_lambda()
		if self.end_lambda is None:
			self.end_lambda=np.floor(max(la))
			self.la,self.lag=self.set_lambda()
	
		
		self.data=self.format_spectrum(flux,la,clip=clip,emission=emission)
		self.meta=meta
		self.best_r=None
		self.best_grid_index=None
		self.best_ccf=None
		self.best_rv=None
		self.grid_r=None
		self.best_teff=None
		self.best_logg=None
		self.best_feh=None
		self.best_alpha=None
		return
		
	def add_grid(self,grid_pickle=None,grid_path=None,grid_class=None,laname=None,silent=False):
		if grid_path is None and grid_pickle is not None:
			try:
				self.grid,self.grid_teff,self.grid_logg,self.grid_feh,self.grid_alpha,grid_la=pickle.load( open( grid_pickle, "rb" ) )
			except ValueError:
				print("Cannot load this grid, recompute and make sure it has been properly formated.")
			if (np.abs(grid_la[0].value-self.st_lambda)>0.1) | (np.abs(grid_la[-1].value-self.end_lambda)>0.1) | (len(grid_la)!=self.ncols):
				raise RuntimeError('The grid has an incompatible wavelength range of [{st},{ed},{bn}] with the data. Specify grid_path and recompute again.'.format(st=str(grid_la[0]),ed=str(grid_la[-1]),bn=str(len(grid_la))))
		elif grid_pickle is None:
			raise RuntimeError('Please provide a path to a pickle file to either load or save the grid/templates.')
		else:
			if grid_class is None:
				raise RuntimeError('Please provide the grid type or an appropriate data loader')
			self.grid,self.grid_teff,self.grid_logg,self.grid_feh,self.grid_alpha=self.add_new_grid(grid_pickle,grid_path,grid_class,laname=laname,silent=silent)
			
			
		self.grid_teff_num=len(np.unique(self.grid_teff))
		self.grid_logg_num=len(np.unique(self.grid_logg))
		self.grid_feh_num=len(np.unique(self.grid_feh))
		self.grid_alpha_num=len(np.unique(self.grid_alpha))
		self.grid_class=grid_class
		return
		
	
	def add_new_grid(self,grid_pickle,grid_path,grid_class,laname=None,silent=False):
		
		path= glob.glob(grid_path)
		if len(path)>1:
			a=np.argsort(path)
			path=np.array(path)[a]
		else:
			path=np.array(path)
		
		teffs=[]
		loggs=[]
		fehs=[]
		alphas=[]
		temps=[]
		for i in path:
			if not silent: print(i)
			#try:
			temp,la,teff,logg,feh,alpha=grid_loader(i,grid_class,laname=laname)
			temps.append(self.format_spectrum(temp,la))
			teffs.append(teff)
			loggs.append(logg)
			fehs.append(feh)
			alphas.append(alpha)
			#except:
			#	pass
		
		pickle.dump( [np.array(temps),np.array(teffs),np.array(loggs),np.array(fehs),np.array(alphas),self.la], open(grid_pickle, "wb" ) )
		return np.array(temps),np.array(teffs),np.array(loggs),np.array(fehs),np.array(alphas)
		
	def format_spectrum(self,flux,la,clip=True,emission=False):
	
		mx=np.nanmax(flux)
		if mx==0.: mx=1.	
		spec = Spectrum1D(spectral_axis=la*u.AA, flux=np.nan_to_num(flux)/mx*u.Jy)
		
		#rebin
		if (min(la)>self.st_lambda) | (max(la)<self.end_lambda):
			raise RuntimeError('st_lambda {st} or end_lambda {ed} are outside of the input spectrum range of {mn} to {mx}'.format(st=str(self.st_lambda),ed=str(self.end_lambda),mn=str(min(la)),mx=str(max(la))))
		spec=self.spline(spec,self.la)
		
		#continuum correct
		spec_fit = fit_continuum(spec)
		spec_cont = spec_fit(self.la)
		spec=spec/spec_cont
		
		spec=spec.flux.value
		
		if clip:
			a=np.where((spec>2) | (spec<-0.5))[0]
			spec[a]=1
		
		if emission:
			w=15
			a=np.where(((self.la.value>6562.79-w) & (self.la.value<6562.79+w)) | ((self.la.value>4861.35-w) & (self.la.value<4861.35+w)) | ((self.la.value>4340.472-w) & (self.la.value<4340.472+w)) | ((self.la.value>4101.734-w) & (self.la.value<4101.734+w)))[0]
			spec[a]=1
		
		return spec
		
	def small_spectrum(self):
		a=np.where((self.la>6400*u.AA) & (self.la<6800*u.AA))[0]
		if len(a)>0:
			spec = Spectrum1D(spectral_axis=self.la[a], flux=self.data[a]*u.Jy)
			spec_fit = fit_continuum(spec)
			spec_cont = spec_fit(self.la[a])
			spec=spec/spec_cont
			spec = Spectrum1D(spectral_axis=self.la[a], flux=spec.flux.value*u.Jy)
			return spec
		else:
			return None

	#taper function to bring ends of the rebinned spectra & template to zero within bell_window fraction
	def taper_spectrum(self):
		taper=np.ones(self.ncols)
		off=int(np.around(self.ncols*self.bell_window))
		taper[:off]=taper[:off]*np.sin(np.arange(off)*np.pi/2/off)
		taper[-off:]=taper[-off:]*np.cos(np.arange(off)*np.pi/2/off)
		return taper
	
	#taper function for the cross correlation in FFT space
	def taper_FFT(self):
		k=fftfreq(self.ncols)*self.ncols/2/np.pi
		taperf=np.ones(self.ncols)
		a=np.where((np.abs(k)>=self.nrun) | (np.abs(k)<=self.low_bin))[0]
		taperf[a]=0
		a=np.where((np.abs(k)>self.low_bin) & (np.abs(k)<=self.top_low))[0]
		taperf[a]=np.sin((np.abs(k[a])-self.low_bin)*np.pi/2/(self.top_low-self.low_bin))
		a=np.where((np.abs(k)>=self.top_nrun) & (np.abs(k)<self.nrun))[0]
		taperf[a]=np.cos((np.abs(k[a])-self.top_nrun)*np.pi/2/(self.nrun-self.top_nrun))
		return taperf
	
	#sets up the rebinned loglam & creates lag in km/s
	def set_lambda(self):
		i=int(self.ncols/2)
		new_la=10**(np.linspace(np.log10(self.st_lambda),np.log10(self.end_lambda),self.ncols))*u.AA
		lagmult=(new_la[i+1]-new_la[i-1])/new_la[i]/2*299792.458
		lag=np.arange(-self.ncols/2,self.ncols/2)*lagmult
		return new_la,lag
	
	#calculates r value of cross correlation
	def calcR(self,x,pm=None):
		if pm is None:
			pm=int(self.ncols/2)
		a=np.where((self.lag>self.minvel) & (self.lag<self.maxvel))[0]
		peak_loc=a[np.argmax(x[a])]
		if peak_loc<pm: pm=peak_loc
		if peak_loc>len(x)-pm: pm=len(x)-peak_loc
		if peak_loc==0:
			return -1000
		endpoint=peak_loc+pm
		startpoint=peak_loc-pm
		mirror=np.flip(x[peak_loc:endpoint])
		sigmaA=np.sqrt(1./2/len(mirror)*np.sum((x[startpoint:peak_loc]-mirror)**2))
		return x[peak_loc]/sigmaA/np.sqrt(2)
	
	
	def run_XCSAO(self,run_subgrid=True):
		if self.data is None:
			raise RuntimeError('Please add a data spectrum.')
		if self.grid is None:
			raise RuntimeError('Please add a template grid/spectrum.')
		
		rr=self.get_r_for_grid(self.grid)
		try:
			a=np.where(rr==max(rr))[0][0]
		except:
			a=0
		
		self.grid_r=np.array(rr)
		self.best_r=rr[a]
		self.best_grid_index=a
		self.best_ccf=self.getCCF(self.data,self.grid[a])
		
		if np.isfinite(self.best_r):
			self.get_rv()
		else:
			self.best_rv=[np.nan,np.nan]
		
		
		if (self.grid_teff_num>2) & (run_subgrid):
			self.best_teff=self.get_par(self.best_teff_subgrid)
		if (self.grid_logg_num>2) & (run_subgrid):
			self.best_logg=self.get_par(self.best_logg_subgrid)
		if (self.grid_feh_num>2) & (run_subgrid):
			self.best_feh=self.get_par(self.best_feh_subgrid)
		if (self.grid_alpha_num>2) & (run_subgrid):
			self.best_feh=self.get_par(self.best_feh_subgrid)
		
		
		return self.best_template()
		
	def compare_sparse(self):
		print(self.get_par(self.best_teff_sparse))
		print(self.get_par(self.best_logg_sparse))
		print(self.get_par(self.best_feh_sparse))
		return
	
	def get_r_for_grid(self,grid):
		rr=[]
		for g in grid:
			out=self.getCCF(self.data,g)
			rr.append(self.calcR(out))
		return np.array(rr)
	
	def get_rv(self):
		a=np.where((self.lag>self.minvel) & (self.lag<self.maxvel))[0]

		peak_loc=a[np.argmax(self.best_ccf[a])]
		
		left,right=peak_loc,peak_loc
		
		while self.best_ccf[peak_loc]<self.best_ccf[left]*2:
			left=left-1
		while self.best_ccf[peak_loc]<self.best_ccf[right]*2:
			right=right+1
		
		
		z=np.polyfit(self.lag[left:right],self.best_ccf[left:right], 2)
		rv=(-z[1]/2/z[0])
		rve=3*(self.lag[right]-self.lag[left])/8/(1+self.best_r)
		
		self.best_rv=[rv,rve.value]
		return self.best_rv
		
		

    
	def get_par(self,func):
	
		par,rr=func()
		weight=np.exp(rr)
		weight=10**(rr)
		
		average=np.average(par,weights=weight)
		variance = np.average((par-average)**2, weights=weight)
		
		return average, np.sqrt(variance)
	
	def best_teff_sparse(self):
		a=np.where((self.grid_logg==self.grid_logg[self.best_grid_index]) & (self.grid_feh==self.grid_feh[self.best_grid_index]) & (self.grid_alpha==self.grid_alpha[self.best_grid_index]))[0]
		return self.grid_teff[a],self.grid_r[a]
		
	def best_logg_sparse(self):
		a=np.where((self.grid_teff==self.grid_teff[self.best_grid_index]) & (self.grid_feh==self.grid_feh[self.best_grid_index]) & (self.grid_alpha==self.grid_alpha[self.best_grid_index]))[0]
		return self.grid_logg[a],self.grid_r[a]
		
	def best_feh_sparse(self):
		a=np.where((self.grid_logg==self.grid_logg[self.best_grid_index]) & (self.grid_teff==self.grid_teff[self.best_grid_index]) & (self.grid_alpha==self.grid_alpha[self.best_grid_index]))[0]
		return self.grid_feh[a],self.grid_r[a]
		
	def best_alpha_sparse(self):
		a=np.where((self.grid_logg==self.grid_logg[self.best_grid_index]) & (self.grid_feh==self.grid_feh[self.best_grid_index]) & (self.grid_teff==self.grid_teff[self.best_grid_index]))[0]
		return self.grid_alpha[a],self.grid_r[a]
	
	def best_teff_subgrid(self,subscale=25):
		a=np.where((self.grid_logg==self.grid_logg[self.best_grid_index]) & (self.grid_feh==self.grid_feh[self.best_grid_index]) & (self.grid_alpha==self.grid_alpha[self.best_grid_index]))[0]
		f=interpolate.interp2d(self.la,self.grid_teff[a], self.grid[a], kind='quintic')
		newteff=np.arange(min(self.grid_teff[a]),max(self.grid_teff[a]),subscale)
		newgrid = f(self.la,newteff)
		rr=self.get_r_for_grid(newgrid)
		return newteff,rr
		
	def best_logg_subgrid(self,subscale=0.05):
		a=np.where((self.grid_teff==self.grid_teff[self.best_grid_index]) & (self.grid_feh==self.grid_feh[self.best_grid_index]) & (self.grid_alpha==self.grid_alpha[self.best_grid_index]))[0]
		f=interpolate.interp2d(self.la,self.grid_logg[a], self.grid[a], kind='cubic')
		newlogg=np.arange(min(self.grid_logg[a]),max(self.grid_logg[a]),subscale)
		newgrid = f(self.la,newlogg)
		rr=self.get_r_for_grid(newgrid)
		return newlogg,rr
		
	def best_feh_subgrid(self,subscale=0.05):
		a=np.where((self.grid_logg==self.grid_logg[self.best_grid_index]) & (self.grid_teff==self.grid_teff[self.best_grid_index]) & (self.grid_alpha==self.grid_alpha[self.best_grid_index]))[0]
		f=interpolate.interp2d(self.la,self.grid_feh[a], self.grid[a], kind='linear')
		newfeh=np.arange(min(self.grid_feh[a]),max(self.grid_feh[a]),subscale)
		newgrid = f(self.la,newfeh)
		rr=self.get_r_for_grid(newgrid)
		return newfeh,rr
		
	def best_alpha_subgrid(self,subscale=0.05):
		a=np.where((self.grid_logg==self.grid_logg[self.best_grid_index]) & (self.grid_feh==self.grid_feh[self.best_grid_index]) & (self.grid_teff==self.grid_teff[self.best_grid_index]))[0]
		f=interpolate.interp2d(self.la,self.grid_alpha[a], self.grid[a], kind='linear')
		newalpha=np.arange(min(self.grid_alpha[a]),max(self.grid_alpha[a]),subscale)
		newgrid = f(self.la,newalpha)
		rr=self.get_r_for_grid(newgrid)
		return newalpha,rr
		
	def getCCF(self,data,temp):
		out=fft(data*self.taper)*fft(temp).conj()
		out=out*self.taperf
		out=fftshift(ifft(out)).real
		return out
		
	def best_template(self):
		st=self.meta
		st['r']=self.best_r
		st['rv']=self.best_rv[0]
		st['erv']=self.best_rv[1]
		st['grid']=self.grid_class
		st['st_lambda']=self.st_lambda
		st['end_lambda']=self.end_lambda
		
		try:
			st['teff']=self.best_teff[0]
			st['eteff']=self.best_teff[1]
		except:
			st['teff']=self.grid_teff[self.best_grid_index]
			st['eteff']=np.nan
		try:
			st['logg']=self.best_logg[0]
			st['elogg']=self.best_logg[1]
		except:
			st['logg']=self.grid_logg[self.best_grid_index]
			st['elogg']=np.nan
		try:
			st['feh']=self.best_feh[0]
			st['efeh']=self.best_feh[1]
		except:
			st['feh']=self.grid_feh[self.best_grid_index]
			st['efeh']=np.nan
		try:
			st['alpha']=self.best_alpha[0]
			st['ealpha']=self.best_alpha[1]
		except:
			st['alpha']=self.grid_alpha[self.best_grid_index]
			st['ealpha']=np.nan
			
		return st
	
		
	def lag(self):
		return self.lag
	def wavelength(self):
		return self.la
	def spectrum(self):
		if self.data is not None:
			return self.data
		else:
			raise RuntimeError('No spectra have been added yet.')
	def CCF(self):
		if self.best_ccf is not None:
			return self.best_ccf
		else:
			raise RuntimeError('CCF is not computed yet, run run_XCSAO first.')
		

import numpy as np
import astropy.units as u
from specutils import Spectrum1D
from specutils.fitting import fit_continuum
from specutils.manipulation import FluxConservingResampler, LinearInterpolatedResampler, SplineInterpolatedResampler
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import data_loader
class getEqW():
	def __init__(self):

		self.la=np.append(np.linspace(6564.6-100,6564.6+100,128)*u.AA,np.linspace(6709.8-20,6709.8+20,128)*u.AA)
		self.st_lambda=np.min(self.la.value)
		self.end_lambda=np.max(self.la.value)
		self.ncols=128
		self.spline = SplineInterpolatedResampler()
		
		self.model_li=Net()
		self.model_ha=Net()
		
		self.model_li.load_state_dict(torch.load('pyxcsao/getli.pt', map_location='cpu'))
		self.model_ha.load_state_dict(torch.load('pyxcsao/getha.pt', map_location='cpu'))
		
		self.model_li.eval()
		self.model_ha.eval()
		
	def add_spectrum(self,name,i=0,laname=None,data_class='boss'):
		flux,la,meta=data_loader(name,i=i,data_class=data_class,laname=laname)		
		
		cont=self.measureEQW(la,flux,7035-15,7035+15)
		self.cah=cont/self.measureEQW(la,flux,6975-15,6975+15)
		self.tio1=cont/self.measureEQW(la,flux,7140-15,7140+15)
		self.tio2=self.measureEQW(la,flux,7500-15,7500+15)/self.measureEQW(la,flux,7800-15,7800+15)
		self.nai=self.measureEQW(la,flux,8150-15,8150+15)/self.measureEQW(la,flux,8189-15,8189+15)
		
		self.data=torch.Tensor([[self.format_spectrum(flux,la)]])
		self.meta=meta
		return
		
	def format_spectrum(self,flux,la):
	
		mx=np.nanmax(flux)
		if mx==0.: mx=1.	
		spec = Spectrum1D(spectral_axis=la*u.AA, flux=np.nan_to_num(flux)/mx*u.Jy)
		if (min(la)>self.st_lambda) | (max(la)<self.end_lambda):
			raise RuntimeError('st_lambda {st} or end_lambda {ed} are outside of the input spectrum range of {mn} to {mx}'.format(st=str(self.st_lambda),ed=str(self.end_lambda),mn=str(min(la)),mx=str(max(la))))
		spec=self.spline(spec,self.la)
		
		
		region = [(6400 * u.AA, 6500 * u.AA), (6600 * u.AA, 6800 * u.AA)]
		spec_fit = fit_continuum(spec,window=region)
		spec_cont = spec_fit(self.la)
		if len(np.where(spec_cont<=0)[0])==0:
			spec=spec/spec_cont-1
		
		return spec.flux.value
	
	def eqw(self):
		clusterx_li=self.data[:,:,128:]
		clusterx_ha=np.log10(250+self.data[:,:,:128])-2.4
		with torch.no_grad():
			#self.ha=model(clusterx_ha).cpu().detach().numpy()[:,0]
			self.ha=25-10**(self.model_ha(clusterx_ha).cpu().detach().numpy()[:,0]+1.4)[0]
			self.li=self.model_li(clusterx_li).cpu().detach().numpy()[:,0][0]
		return self.ha,self.li
		
		
	def measureEQW(self,x1,y1,minx,maxx):
	    x=np.linspace(minx,maxx,50)
	    y=np.interp(x, x1, y1)
	    eqw=-(np.trapz(y,x=x)-np.trapz(y*0+1,x=x))
	    
	    return eqw		
		
		
class Net(nn.Module):

    def __init__(self,out=1,lin=360):
        super(Net, self).__init__()
        self.feats = nn.Sequential(
            nn.Conv1d(1, 32, 5, 1, 2),
            nn.MaxPool1d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 64, 3,  1, 2),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 64, 3,  1, 2),
            nn.MaxPool1d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm1d(64),

            nn.Conv1d(64, 128, 3, 1, 2),
            nn.ReLU(True),
            nn.BatchNorm1d(128)
        )

        self.classifier = nn.Conv1d(128, 10, 1)
        self.avgpool = nn.AvgPool1d(2,2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(lin, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512,out)

    def forward(self, inputs):
        out = self.feats(inputs)
        out = self.dropout(out)
        out = self.classifier(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)
        return out

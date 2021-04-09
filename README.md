# PyXCSAO
Replicates functionality of IRAF XCSAO

To run:

### Import
from pyxcsao.crosscorrelate import PyXCSAO

### Initiates instance:
b=PyXCSAO(st_lambda=4000,end_lambda=6000,ncols=16000)

---optional parameters: low_bin=5,top_low=10,top_nrun=125,nrun=255,bell_window=0.05,minvel=-500,maxvel=500

### Adds Synthetic grid

First time running:
b.add_grid(grid_pickle='phoenix.p',grid_path='phoenix/*0.0/*4.5*.fits',grid_class='phoenix') 

---options: phoenix, phoenixhires, coelho

From a precompiled pickle file:

b.add_grid(grid_pickle='phoenix.p')

### Adds data

b.add_spectrum('file.fits',data_class='boss')

---options: boss,lamost,user

### Run XCSAO and get parameters

print(b.run_XCSAO(run_subgrid=False))

### Plot CCF:

plt.plot(b.lag,b.best_ccf)
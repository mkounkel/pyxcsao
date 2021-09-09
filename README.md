# PyXCSAO
Replicates functionality of IRAF XCSAO

To run:

### Import
from pyxcsao.crosscorrelate import PyXCSAO

### Initiates instance:
b=PyXCSAO(st_lambda=5000,end_lambda=10000)

---optional parameters: ncols=8192,low_bin=0,top_low=10,top_nrun=125,nrun=255,bell_window=0.05,minvel=-500,maxvel=500

### Adds Synthetic grid

First time running:
b.add_grid(grid_pickle='phoenix.p',grid_path='phoenix/*0.0/*4.5*.fits',grid_class='phoenix') 

---options: phoenix, phoenixhires, coelho

From a precompiled pickle file:

b.add_grid(grid_pickle='phoenix.p')

### Adds data

b.add_spectrum('file.fits',data_class='boss')

---options: boss,lamost,segue,user

### Run XCSAO and get parameters

print(b.run_XCSAO())

### Optimized for large grids:

print(b.run_XCSAO_optimized())

### Plot CCF:

plt.plot(b.lag,b.best_ccf)

### Example Code
```python
import glob
import pandas as pd
from pyxcsao.crosscorrelate import PyXCSAO
from astropy.table import Table
import time

cat=Table.read('path.fits')

best=[]
b=PyXCSAO(st_lambda=5000,end_lambda=10000)
b.add_grid(grid_pickle='phoenix_full1.p')


batchsize=500
for j in range(0,len(cat),batchsize):
    cat1=cat[j:j+batchsize]
    print(j)
    for i in range(len(cat1)):
        path=cat1['path'][i]
        try:
            b.add_spectrum(path)
            x=b.run_XCSAO_optimized()
            best.append(x.copy())
        except:
            print(path)
            

df = pd.DataFrame(best)
df.to_csv('batch.csv')
```

import numpy as np
from scipy import fft, io
import os
import math

n_train=5
n=0
num=[180,550,1000,2000,5200]

data_motai=np.zeros((n_train, 200, 173,47))
data_motai_vv=np.zeros((n_train, 200, 173,47))
data_motai_ww=np.zeros((n_train, 200, 173,47))
data_motai_uv=np.zeros((n_train, 200, 173,47))
data_motai_3=np.zeros((n_train, 600, 173,47))
data_kzs=np.zeros((n_train, 173))
data_cps=np.zeros((n_train, 47))
data_yy=np.zeros((n_train, 200))
data_um=np.zeros((n_train, 200,173))
data_vm=np.zeros((n_train, 200,173))
data_wm=np.zeros((n_train, 200,173))
data_uvm=np.zeros((n_train, 200,173))
data_4m=np.zeros((n_train, 800,173))


for i in num:
    print(i)
    data=io.loadmat('y_jiami_data/Euukc_Re'+str(i)+'.mat')
    data_motai[n]=data['Euu_k'][0:200]
    data_motai_vv[n]=data['Evv_k'][0:200]
    data_motai_ww[n]=data['Eww_k'][0:200]
    # data_motai_uv[n]=data['Euv_k'][0:100]
    data_motai_3[n]=np.concatenate((data_motai[n],data_motai_vv[n],data_motai_ww[n]),axis=0)
    data_cps[n]=data['cPs']
    data_kzs[n]=data['kzs']
    data_yy[n]=io.loadmat('y_jiami_data/N400/spectra1d_Re'+str(i)+'.mat')['yd_1d']
    
    data_spe=io.loadmat('y_jiami_data/N400/spectra1d_Re'+str(i)+'.mat')
    data_um[n]=io.loadmat('y_jiami_data/N400/spectra1d_Re'+str(i)+'.mat')['Euu_kz_1d']
    data_vm[n]=io.loadmat('y_jiami_data/N400/spectra1d_Re'+str(i)+'.mat')['Evv_kz_1d']
    data_wm[n]=io.loadmat('y_jiami_data/N400/spectra1d_Re'+str(i)+'.mat')['Eww_kz_1d']
    data_uvm[n]=io.loadmat('y_jiami_data/N400/spectra1d_Re'+str(i)+'.mat')['Euv_kz_1d']
    data_4m[n]=np.concatenate((data_um[n],data_vm[n],data_wm[n],data_uvm[n]),axis=0)
    
    


    n=n+1
data_dcps=data_cps[:,1:47]-data_cps[:,0:46]
data_dkzs=data_kzs[:,0:172]-data_kzs[:,1:173]

np.save('data_motai.npy', data_motai)
np.save('data_motai_vv.npy', data_motai_vv)
np.save('data_motai_ww.npy', data_motai_ww)
#np.save('data_motai_uv.npy', data_motai_uv)
np.save('data_motai_4.npy', data_motai_3)
np.save('data_cps.npy', data_cps)
np.save('data_kzs.npy', data_kzs)
np.save('data_yy.npy', data_yy)
np.save('data_dcps.npy', data_dcps)
np.save('data_dkzs.npy', data_dkzs)
np.save('data_um.npy', data_um)
np.save('data_vm.npy', data_vm)
np.save('data_wm.npy', data_wm)
np.save('data_uvm.npy', data_uvm)
np.save('data_4m.npy', data_4m)

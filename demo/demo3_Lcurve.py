# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 18:29:09 2022
Gravity inversion in wavenumber domain 
Select the regularization parameter 
@author: chens
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# local imports
import freqinv.su as su
#from pyevtk.hl import gridToVTK
import datetime
import pathlib

if __name__ == '__main__':
    # define model volume and grid shape

    work_dir = '.'
        #geoist.TEMP_PATH #pathlib.Path('/home/zhangb/work/people/zhangbei/freq')
    nzyx = [2,16,16] # The grid setting of field source, [z,y,x] are the number of meshes in the three directions, respectively
    source_volume = [-3000, 3000, -3000, 3000, 100, 1000] # Field source distribution range, [xmin,xmax,ymin,ymax,zmin,zmax]
    obs_area = [-3000,3000,-3000,3000] # Observation area,[xmin,xmax,ymin,ymax]
    nobsyx = [30,30] # The number of lattice points of the observation field, which are the number of lattice points in the y and x directions, respectively

    model_density = np.zeros(tuple(nzyx)) # Density model used for forward
    model_density[1:2,4:8,5:8] = 1000
    model_density[1:2,10:12,2:5] = 1500
    model_density[1:2,11:12,13:15] = 2000

    refer_densities = [] # A list of inverted reference models, each member of which is a reference density model.
    
        
    weights = {'refers':[0.01]} # trade-off for reference model
    refer_density = np.zeros(tuple(nzyx)) #  Set a reference model with density equals zero, which is equivalent to the minimum model constraint
    refer_density = refer_density.ravel()
    refer_densities.append(refer_density) # Add the reference model to reference list

    small_model = su.FreqInvModel(nzyx=nzyx,
                                       source_volume=source_volume,
                                       nobsyx=nobsyx,
                                       obs_area=obs_area,
                                       model_density=model_density,
                                       weights=weights,
                                       refer_densities=refer_densities)
        
        
    small_model.gen_mesh(height=1.)  # Grid generation
    small_model.gen_kernel() # kernel function generation
    small_model.forward(update=True) # Forward
                                     # If the density source is not given, use small_model.model_density as the density source.
                                     # If update=True,update small_model.freq and small_model.obs_field as forward result.
                                     # If update=False,return the result of the calculation，small_model.freq and small_model.obs_field will not change.
    true_freq = small_model.freq     # Back up true frequency domain gravity
    small_model.obs_freq = small_model.freq # small_model.obs_freq,The observed gravity in frequency domain.
    # add noise
    #real_noise = 0.1*np.max(np.abs(small_model.obs_freq.real))
    #imag_noise = 0.1*np.max(np.abs(small_model.obs_freq.imag))
    real_noise = 0.00*(np.max(small_model.obs_freq.real)-np.min(small_model.obs_freq.real))
    imag_noise = 0.00*(np.max(small_model.obs_freq.imag)-np.min(small_model.obs_freq.imag)) 
    mshapex, mshapey = small_model.obs_freq.shape    
    small_model.obs_freq += real_noise*np.random.rand(mshapex, mshapey) + imag_noise*np.random.rand(mshapex, mshapey)*1j
    
    fieldt= np.real(np.fft.ifft2(small_model.obs_freq))
    fieldn = fieldt[small_model.padx: small_model.padx + nobsyx[1], small_model.pady: small_model.pady + nobsyx[0]].ravel()
    print(np.max(small_model._model_density))
    print(len(small_model._model_density))
    print(np.conj(small_model.FX).shape,small_model.FY.shape)
    st = datetime.datetime.now()
    
    inv_models = []
    model_n2l = []
    data_n2l = []
    freq_n2l = []
    nz,ny,nx = nzyx
    for i in range(-6, 1, 1):
        weights = {'refers':[10**i]} 
        small_model._weights = weights
        small_model.do_linear_solve_quiet() # Inversion, The results save to small_model.solution.
        arr = small_model.solution.real.copy()
        modelinv = arr.reshape(nz,ny,nx)
        model_n2 = np.linalg.norm(modelinv)        
        freq,recover = small_model.forward(small_model.solution.real) # Predict gravity using the inversion result.        
        data_n2 = np.linalg.norm(recover - small_model.obs_field)
        freq_n2 = np.linalg.norm(freq - true_freq)
        print('norm model and data: ',model_n2,data_n2,freq_n2)  
        model_n2l.append(model_n2)
        data_n2l.append(data_n2)
        freq_n2l.append(freq_n2)
        inv_models.append(small_model)
    ed = datetime.datetime.now()
    print("inversion use time: ",ed-st)

    df1 = pd.DataFrame(columns=['model','data','freq','beta'], dtype= object)
    df1['model'] = model_n2l
    df1['data'] = data_n2l
    df1['freq'] = freq_n2l
    df1['beta'] = [10**i for i in range(-6, 1, 1)]
    df1.to_csv('.\\lcurve.txt')


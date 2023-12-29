#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:09:50 2019
data_load loads Multi coil data with kernwls for T1/T2 images

@author: salman
"""

def data_load(directory,directory_kernel,batch_size=5,subs=1,slices=10,tot_slices=10,shuffle=1):
    #total slices means total numbber of slices per subject, and slices means the selected number of slicces per subjeccts
    import numpy as np, h5py
    f = h5py.File(directory,  "r")
    f_kernel=h5py.File(directory_kernel,  "r")
    samples=np.split(np.array(range(subs*tot_slices)),subs,0)
    #np.random.seed(52)     
    #loading samples and concatenating
    for ii in range(subs):       
        if slices==tot_slices:
            samples[ii]=samples[ii][range(0,slices)]
        else:
            np.random.seed(52) 
            np.random.shuffle(samples[ii])
            samples[ii]=samples[ii][range(0,slices)]
    samples=np.concatenate(samples) 
    if shuffle==1:    
        np.random.seed(42)
        np.random.shuffle(samples)
        print(samples)
    
    #Load undersampled images
    images_us=f.get('images_us')
    images_us=np.array(images_us)
    images_us=images_us['real']+ 1j *images_us['imag']
    normalizer=(abs(images_us).max())
    images_us=images_us/normalizer     
    images_us=np.transpose(images_us,axes=(1,0,3,2))[samples,:,:,:]
    data_instances=images_us.shape[0]
    images_us=np.split(images_us,round(data_instances/batch_size),0)
    #load undersampling masks
    masks=f.get('map')
    masks=np.array(masks)           
    masks=np.transpose(masks,axes=(0,2,1))[samples,:,:]
    #masks=np.fft.ifftshift(masks,axes=(1,2))
    masks=np.split(masks,round(data_instances/batch_size),0)
    #load fully sampled data
    images_fs=f.get('images_fs')
    images_fs=np.array(images_fs)
    images_fs=images_fs['real']+ 1j *images_fs['imag']
    images_fs=images_fs/normalizer
    images_fs=np.transpose(images_fs,axes=(0,2,1))[samples,:,:]
    images_fs=np.split(images_fs,round(data_instances/batch_size),0)
    #load coil sensitivity masks
    coil_maps=f.get('coil_maps')
    coil_maps=np.array(coil_maps)
    coil_maps=coil_maps['real']+ 1j *coil_maps['imag']  
    normalizer=abs(coil_maps).max()
    coil_maps=np.transpose(coil_maps,axes=(1,0,3,2))[samples,:,:,:]
    coil_maps=np.split(coil_maps,round(data_instances/batch_size),0) 
    #load_kernel
    kernel=f_kernel.get('kernel')
    kernel=np.array(kernel)
    kernel=kernel['real']+ 1j *kernel['imag']   
    kernel=np.transpose(kernel,axes=(0,1,2,4,3))[samples,:,:,:,:]#loop should across dim=1
    kernel=np.split(kernel,round(data_instances/batch_size),0) 
    #kernel=np.pad(kernel, [(0, ), (0, ),(0, ),((images_us.shape[2]-kernel.shape[3])/2, ),((images_us.shape[3]-kernel.shape[4])/2, ) ],mode='constant')

    
    
    return (images_us, images_fs, masks, coil_maps,kernel,samples)

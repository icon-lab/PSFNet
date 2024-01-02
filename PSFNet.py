# -*- coding: utf-8 -*-
"""


* 
* This code trains and test PSFNet
"""
    
# imports
import tensorflow as tf
import time
import h5py
import numpy as np
from transform import ifft2c
from model_hybrid_fused import network
import argparse
from data import data_load

parser = argparse.ArgumentParser()
parser.add_argument('--subs', type= int, default=1, help='No. of subjects in fastMRI')
parser.add_argument('--slices', type= int, default=10, help='No. of slices in fastMRI')
parser.add_argument('--us', type= int, default=4, help='Acceleration rate')
parser.add_argument('--data_dir', type = str,default='data/')
parser.add_argument('--ckpt_dir', type = str,default='ckpt/')
parser.add_argument('--results_dir', type = str,default='results/')
parser.add_argument('--Epochs', type = int,default=10)
parser.add_argument('--batch_size', type = int,default=2)
parser.add_argument('--cascades', type = int,default=5, help= 'No. of cascades')
parser.add_argument('--n_spirit_itr', type = int,default=5, help= 'number of spirit blocks per cascade')
parser.add_argument('--learning_rate', type = float,default=1e-4)
parser.add_argument('--recursive', action='store_false', help= 'if same weights are shared across cacscades' )
parser.add_argument('--joint_optimization', action='store_false')
#Arguments
args = parser.parse_args()
subs = args.subs
slices = args.slices
us = args.us
data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
results_dir = args.results_dir
Epochs = args.Epochs
batch_size = args.batch_size
batch_size_val = 2
cascades = args.cascades
total_cascades=cascades+1
n_spirit_itr = args.n_spirit_itr
learning_rate = args.learning_rate
recursive = args.recursive
joint_optimization = args.joint_optimization 

data_loader = data_load

#load training data
dataset=data_dir + 'images_train.mat'
dataset_kernel=data_dir + 'kernel_train.mat'
(images_us, images_fs, masks, coil_maps,int_kernel,samples)=data_loader(dataset,dataset_kernel,batch_size=batch_size,subs=subs,slices=slices)
#load validation data
dataset=data_dir + 'images_val.mat'
dataset_kernel=data_dir + 'kernel_val.mat'
(images_us_val, images_fs_val, masks_val, coil_maps_val,int_kernel_val,samples)=data_loader(dataset,dataset_kernel,batch_size=batch_size_val,subs=10,slices=10,shuffle=0)   
#Get shapes of images
batch_size, coil_n,  patch_size_x, patch_size_y = images_us[0].shape
kernel_size_train = int_kernel[0].shape[-1];kernel_train_x = int(np.floor(kernel_size_train/2));kernel_train_y = int(np.ceil(kernel_size_train/2))
kernel_size_val = int_kernel_val[0].shape[-1];kernel_val_x = int(np.floor(kernel_size_train/2));kernel_val_y = int(np.ceil(kernel_size_train/2))
# Begin session
tf.compat.v1.disable_eager_execution() 
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
graph=tf.compat.v1.reset_default_graph()  

#Set place holders
patch_size=None
patch_size=None
x = tf.compat.v1.placeholder(tf.complex64, shape=[None,coil_n, patch_size,patch_size], name="x") # undersampled  
y = tf.compat.v1.placeholder(tf.complex64, shape=[None, patch_size,patch_size], name="y") # groundtruth   
us_masks=tf.compat.v1.placeholder(tf.bool, shape=[None,patch_size,patch_size], name="us_masks") #masks
coil_sens_maps=tf.compat.v1.placeholder(tf.complex64, shape=[None, coil_n,patch_size,patch_size], name="coil_sens_maps") #coil sensitivity maps
kernel=tf.compat.v1.placeholder(tf.complex64, shape=[None, coil_n,coil_n,patch_size,patch_size], name="kernel")#interpolation kernel
l_r=tf.compat.v1.placeholder(tf.float32, shape=[], name="l_r")#learning rate
sess = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
#define network
dnn=network( x=x,y=y,us_masks=us_masks,coil_sens_maps=coil_sens_maps,kernel=kernel,l_r=l_r,cascades=cascades, n_spirit_itr=n_spirit_itr, coil_n=coil_n,recursive=recursive)    
sess.run(tf.compat.v1.global_variables_initializer())
model_saver = tf.compat.v1.train.Saver()
#define validation variables for monitoring the training
validation_samples = batch_size_val * len(images_us_val)
val_error_mse_pre,val_error_mae_pre,val_error_mse,val_error_mae,val_image_dc,val_image_spirit=np.zeros([Epochs,
                                                                                                        len(images_us_val)]),np.zeros([Epochs,
                                                                                                        len(images_us_val)]),np.zeros([Epochs,
                                                                                                        len(images_us_val)]),np.zeros([Epochs,
                                                                                                        len(images_us_val)]),np.zeros([
                                                                                                        len(images_us_val),patch_size_x,patch_size_y],dtype='complex64'),np.zeros([
                                                                                                        len(images_us_val),patch_size_x,patch_size_y],dtype='complex64')  

  
model_name='model'
total_time_train=np.zeros([Epochs,len(images_us)])  
val_range=list(range(50))    

#cascade loop
for cas in range(total_cascades,total_cascades+1):
    #epoch loop
    for i in range(Epochs):
        #training batch loop
        for j in range(len(images_us)):  
            start=time.time() 
            batch_x,batch_y,batch_coil_maps,batch_mask,batch_kernel = images_us[j],images_fs[j],coil_maps[j],masks[j],int_kernel[j]
            #In the tensofrlow part ifft and fftshift are not implemented, tehrefore the masks is ifftshifted for Data consistency
            batch_mask=np.fft.ifftshift(batch_mask,axes=(1,2))
            batch_kernel=np.pad(batch_kernel,((0,0),(0,0),(0,0),(int(patch_size_x/2-kernel_train_x),int(patch_size_x/2-kernel_train_y)),(int(patch_size_y/2-kernel_train_x),int(patch_size_y/2-kernel_train_y))),mode='constant')
            for kk in range(batch_kernel.shape[0]):                  
                batch_kernel[kk,:,:,:,:]=ifft2c(batch_kernel[kk,:,:,:,:],dim=(2,3))                    
            if joint_optimization:
                sess.run(dnn.optimize_joint(),feed_dict={ x: batch_x, y: batch_y,coil_sens_maps: batch_coil_maps, us_masks:batch_mask, kernel:batch_kernel, l_r:learning_rate})
                total_time_train[i,j]=time.time()-start
        #validation batch loop
        for j in val_range:
            batch_x,batch_y,batch_coil_maps,batch_mask,batch_kernel = images_us_val[j],images_fs_val[j],coil_maps_val[j],masks_val[j],int_kernel_val[j] 
            batch_kernel=np.pad(batch_kernel,((0,0),(0,0),(0,0),(int(patch_size_x/2-kernel_val_x),int(patch_size_x/2-kernel_val_y)),(int(patch_size_y/2-kernel_val_x),int(patch_size_y/2-kernel_val_y))),mode='constant')
            batch_mask=np.fft.ifftshift(batch_mask,axes=(1,2))
            for kk in range(batch_kernel.shape[0]):                  
                batch_kernel[kk,:,:,:,:]=ifft2c(batch_kernel[kk,:,:,:,:],dim=(2,3))           
            
            
            val_error_mse_pre[i,j],val_error_mae_pre[i,j],val_error_mse[i,j],val_error_mae[i,j]=sess.run(dnn.losses(cascade_n=np.min((cas,cascades))-1),feed_dict={x: batch_x, y: batch_y,coil_sens_maps: batch_coil_maps, us_masks:batch_mask, kernel:batch_kernel})
            val_image_dc[j,:,:],val_image_spirit[j,:,:]=sess.run(dnn.predictions(cascade_n=np.min((cas,cascades))-1),feed_dict={x: np.expand_dims(batch_x[1,:,:,:],0),coil_sens_maps: np.expand_dims(batch_coil_maps[1,:,:,:],0), us_masks:np.expand_dims(batch_mask[1,:,:],0), kernel:np.expand_dims(batch_kernel[1,:,:,:,:],0)})
        #save the best model           
        if i>1:
            if ((val_error_mse[i,:].mean()+val_error_mae[i,:].mean())<np.min(val_error_mse[0:i,:].mean(axis=1)+val_error_mae[0:i,:].mean(axis=1))): 
                model_saver.save(sess, ckpt_dir + '/model_best.ckpt')               

        #print Errors
        print ('cascade ' + str(cas) +',Epoch ' + str(i) +',validation error pre DC (MSE) = {}'.format(val_error_mse_pre[i,:].mean()))
        print ('cascade ' + str(cas) +',Epoch ' + str(i) +',validation error pre DC (MAE) = {}'.format(val_error_mae_pre[i,:].mean()))                
        print ('cascade ' + str(cas) +',Epoch ' + str(i) +',validation error (MSE) = {}'.format(val_error_mse[i,:].mean()))
        print ('cascade ' + str(cas) +',Epoch ' + str(i) +',validation error (MAE) = {}'.format(val_error_mae[i,:].mean())) 
    f = h5py.File(ckpt_dir + 'val_loss.mat',  "w")
    f.create_dataset('val_error_mse_pre', data=val_error_mse_pre)
    f.create_dataset('val_error_mae_pre', data=val_error_mae_pre)                
    f.create_dataset('val_error_mse', data=val_error_mse)
    f.create_dataset('val_error_mae', data=val_error_mae)
    f.create_dataset('val_image_dc', data=val_image_dc)
    f.create_dataset('val_image_spirit', data=val_image_spirit)
    f.close()      
            
print('mean time'+str(total_time_train.mean()))
print('var time'+str(total_time_train.var()))    
model_saver.save(sess, ckpt_dir + '/model.ckpt')

#testing
#load test data
batch_size=1
dataset=data_dir + 'images_test.mat'
dataset_kernel=data_dir + 'kernel_test.mat'
(images_us, images_fs, masks, coil_maps,int_kernel,samples)=data_loader(dataset,dataset_kernel,batch_size=batch_size,subs=40,slices=10,shuffle=0)
kernel_size_test = int_kernel[0].shape[-1];kernel_test_x = int(np.floor(kernel_size_train/2));kernel_test_y = int(np.ceil(kernel_size_train/2))


val_error_mse=np.zeros([len(images_us)])
val_error_mae=np.zeros([len(images_us)])
val_image_dc=np.zeros([len(images_us),patch_size_x,patch_size_y],dtype='complex64')#images

total_time=np.zeros([len(images_us)]) 
for j in range(len(images_us)):
    batch_x,batch_y,batch_coil_maps,batch_mask,batch_kernel = images_us[j],images_fs[j],coil_maps[j],masks[j],int_kernel[j]
    batch_mask=np.fft.ifftshift(batch_mask,axes=(1,2))
    batch_kernel=np.pad(batch_kernel,((0,0),(0,0),(0,0),(int(patch_size_x/2-kernel_test_x),int(patch_size_x/2-kernel_test_y)),(int(patch_size_y/2-kernel_test_x),int(patch_size_y/2-kernel_test_y))),mode='constant')
    for kk in range(batch_kernel.shape[0]):                  
        batch_kernel[kk,:,:,:,:]=ifft2c(batch_kernel[kk,:,:,:,:],dim=(2,3))                        
    _,_,val_error_mse[j],val_error_mae[j]=sess.run(dnn.losses(cascade_n=cascades-1),feed_dict={x: batch_x, y: batch_y,coil_sens_maps: batch_coil_maps, us_masks:batch_mask, kernel:batch_kernel})
    start=time.time()
    val_image_dc[j,:,:],_=sess.run(dnn.predictions(cascade_n=cascades-1),feed_dict={x: batch_x, y: batch_y,coil_sens_maps: batch_coil_maps, us_masks:batch_mask, kernel:batch_kernel})
    total_time[j]=time.time()-start
print('mean time'+str(total_time.mean()))
print('var time'+str(total_time.var()))
f = h5py.File(results_dir + '/results_test.mat',  "w")

f.create_dataset('val_error_mse', data=val_error_mse)
f.create_dataset('val_error_mae', data=val_error_mae)
f.create_dataset('val_image_dc', data=val_image_dc)
f.close() 
    


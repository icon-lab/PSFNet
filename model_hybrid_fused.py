#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import sys

class network():    
    def __init__(self, x,y,us_masks,coil_sens_maps,kernel,l_r=1e-4,cascades=5, n_spirit_itr=5, coil_n=5,recursive=True):
        self.x=x
        self.y=y
        self.us_masks=us_masks
        self.coil_sens_maps=coil_sens_maps      
        self.kernel=kernel
        self.l_r=l_r
        self.coil_n=coil_n
        self.cascades = cascades
        self.n_spirit_itr = n_spirit_itr 
        self.recursive=recursive
        self.spirit,self.spirit_dc,self.rec,self.nn,self.nn_dc,self.nn_dc_rec,self.fusion,self.fusion_dc,self.spirit_dc_rec,self.cas,self.cas_fusion,self.loss_l1,self.loss_l2,self.loss_l1_fusion,self.loss_l2_fusion,self.reg,self.train_step,self.train_step_fusion,self.train_step_joint,self.grads_and_vars=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
        for ii in range(self.cascades):
            for jj in range(self.n_spirit_itr):
                if jj==0:#if first ker nel consistency layer 
                    if ii==0:#if first casde directly receive input
                        self.spirit.append(self.kernel_consistency(self.x,self.kernel,self.coil_n,self.coil_sens_maps))
                    else:#else receive from the previous cascade
                        self.spirit.append(self.kernel_consistency(self.fusion_dc[ii-1],self.kernel,self.coil_n,self.coil_sens_maps))#spirit
                    self.spirit_dc.append(self.data_consistency(self.x,self.spirit[ii],self.us_masks,self.coil_sens_maps,self.coil_n))
                else:#else receive from previous consistency layer
                    self.spirit[ii]=self.kernel_consistency(self.spirit_dc[ii],self.kernel,self.coil_n,self.coil_sens_maps)#spirit
                    self.spirit_dc[ii]=self.data_consistency(self.x,self.spirit[ii],self.us_masks,self.coil_sens_maps,self.coil_n)#dc               
            self.spirit_dc_rec.append(self.adj_proj(self.spirit_dc[ii],self.coil_sens_maps))#adjoint projection
            if ii==0:#if first casde directly receive input
                cascade_no=0
                self.nn.append(self.cascade(self.adj_proj(self.x,self.coil_sens_maps),'cascade_'+str(ii+1)))# nn 
            else:
                if self.recursive:
                    cascade_no=0
                    self.nn.append(self.cascade_forward(self.adj_proj(self.fusion_dc[ii-1],self.coil_sens_maps),self.grads_and_vars[0],'cascade_'+str(cascade_no+1)))# nn
                else:   
                    cascade_no=ii
                    self.nn.append(self.cascade(self.adj_proj(self.fusion_dc[ii-1],self.coil_sens_maps),'cascade_'+str(ii+1)))# nn
            self.nn_dc.append(self.data_consistency(self.x,self.forw_proj(self.nn[ii],self.coil_sens_maps,self.coil_n),self.us_masks,self.coil_sens_maps,self.coil_n))# dc
            self.nn_dc_rec.append(self.adj_proj(self.nn_dc[ii],self.coil_sens_maps))
            self.fusion.append((self.fusion_layer(self.nn_dc_rec[ii],self.spirit_dc_rec[ii],'cascade_fusion_'+str(ii+1)))) 
            self.fusion_dc.append(self.data_consistency(self.x,self.forw_proj(self.fusion[ii],self.coil_sens_maps,self.coil_n),self.us_masks,self.coil_sens_maps,self.coil_n))
            self.rec.append(self.adj_proj(self.fusion_dc[ii],self.coil_sens_maps))
            self.loss_l1_fusion.append(tf.reduce_mean(tf.abs(tf.squeeze(tf.math.real(self.fusion[ii]))-tf.math.real(self.y))+tf.abs(tf.squeeze(tf.math.imag(self.fusion[ii]))-tf.math.imag(self.y)) ))
            self.loss_l2_fusion.append(tf.reduce_mean((tf.squeeze(tf.math.real(self.fusion[ii]))-tf.math.real(self.y))**2+(tf.squeeze(tf.math.imag(self.fusion[ii]))-tf.math.imag(self.y))**2))        
            self.loss_l1.append(tf.reduce_mean(tf.abs(tf.squeeze(tf.math.real(self.nn[ii]))-tf.math.real(self.y))+tf.abs(tf.squeeze(tf.math.imag(self.nn[ii]))-tf.math.imag(self.y)) ))
            self.loss_l2.append(tf.reduce_mean((tf.squeeze(tf.math.real(self.nn[ii]))-tf.math.real(self.y))**2+tf.abs(tf.squeeze(tf.math.imag(self.nn[ii]))-tf.math.imag(self.y)))**2)
            self.reg.append(tf.add_n(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES,scope='cascade_'+str(cascade_no+1))))
            self.grads_and_vars.append(tf.compat.v1.train.AdamOptimizer(1e-4).compute_gradients(self.loss_l1[ii]+self.loss_l2[ii]+self.reg[ii],var_list=tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope='cascade_'+str(cascade_no+1))))
        self.grads_and_vars.append(tf.compat.v1.train.AdamOptimizer(1e-4).compute_gradients(self.loss_l1[ii]+self.loss_l2[ii]+self.reg[ii]))

        self.train_step_joint=tf.compat.v1.train.AdamOptimizer(self.l_r).minimize(1e3*self.loss_l1_fusion[-1]+1e3*self.loss_l2_fusion[-1]+self.loss_l1[-1]+self.loss_l2[-1]+self.reg[:])
        self.variables=tf.compat.v1.trainable_variables()

    def dummy_initialize(self):
      
        self.dummy=[tf.compat.v1.constant(var.eval(),name='fix_'+var.name[0:-2]) for var in self.variables] 
        return(self.variables)  
        
    def dummy_save(self):
        self.inter_vars=[tf.compat.v1.Variable(var.eval(),name='new_'+var.name[0:-2]) for var in self.dummy] 
        return(self.variables)    

    def save_checkpoint(self):    
        update_weights = [tf.compat.v1.assign(dummy, main) for (dummy, main) in zip(self.inter_vars, self.variables)]
        return (update_weights[:])
        
    def reset_weights(self,cascade_n=0):
        update_weights = [tf.compat.v1.assign(dummy, main) for (dummy, main) in zip(self.variables, self.inter_vars)]
        return (update_weights[:])
        
    def predictions(self,cascade_n=0):
        return (self.rec[cascade_n],self.nn[cascade_n])
        
    def losses(self,cascade_n=0):
        return (self.loss_l2[cascade_n],self.loss_l1[cascade_n],self.loss_l2_fusion[cascade_n],self.loss_l1_fusion[cascade_n])
                
    def optimize_nn(self,cascade_n=0):
        return (self.train_step_nn)

    def optimize_fusion(self,cascade_n=0):
        return (self.train_step_fusion)        

    def optimize_joint(self,cascade_n=0):
        return (self.train_step_joint)  
              
        
    def weight_variable(self,shape,initializer):
        return tf.compat.v1.get_variable(name="weights", shape=shape, regularizer=tf.keras.regularizers.l2(1e-6))

    def weight_variable_constraint(self,shape,initializer):
        return tf.compat.v1.get_variable(name="weights", shape=shape, constraint=lambda x: tf.clip_by_value(x, 0, 1), initializer=initializer,regularizer=tf.keras.regularizers.l2(1e-6))        
        
        
    def bias_variable(self,shape,initializer):
        return tf.compat.v1.get_variable(name="biases", shape=shape,initializer=initializer)


    def conv_layer(self,input, shape, layer_name,initializer,relu=1):
        with tf.compat.v1.variable_scope(layer_name):
            W = self.weight_variable(shape,initializer)
            b = tf.Variable(tf.zeros([1]),name="biases") 
            linear=tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME') + b
            if relu==1:
                output = tf.nn.relu(linear)
            else:
                output= linear
            return output 


    def conv_layer_forward(self,input,layer_name,weights,relu=1):
        with tf.compat.v1.variable_scope(layer_name):      
            for g,w in weights:
                if w.name==layer_name+"weights:0":
                    W=w
                if w.name==layer_name+"biases:0":
                    b=w                  
            linear=tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME') + b
            if relu==1:
                output = tf.nn.relu(linear)
            else:
                output= linear
            return output 
            
    def conv_layer_constraint_forward(self,input, shape, layer_name,weights,relu=1):
        with tf.compat.v1.variable_scope(layer_name):
            for g,w in weights:
                if w.name==layer_name+"weights:0":
                    W=w

            linear=tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME') 
            if relu==1:
                output = tf.nn.relu(linear)
            else:
                output= linear
            return output     

    def conv_layer_constraint(self,input, shape, layer_name,initializer,relu=1):
        with tf.compat.v1.variable_scope(layer_name):
            W = self.weight_variable_constraint(shape,initializer)
            linear=tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding='SAME') 
            if relu==1:
                output = tf.nn.relu(linear)
            else:
                output= linear
            return output 
            
    def cascade_forward(self,x,weights,name):
        with tf.compat.v1.variable_scope(name):
            
            conv_1_real = self.conv_layer_forward(tf.math.real(tf.expand_dims(x,3)), layer_name=name+"/conv1_real/",weights=weights)
            conv_2_real = self.conv_layer_forward(conv_1_real, layer_name=name+"/conv2_real/",weights=weights)
            conv_3_real = self.conv_layer_forward(conv_2_real,layer_name=name+"/conv3_real/",weights=weights) 
            conv_4_real = self.conv_layer_forward(conv_3_real,layer_name=name+"/conv4_real/",weights=weights) 
            y_rec_real = self.conv_layer_forward(conv_4_real,layer_name=name+"/conv5_real/",weights=weights,relu=0)
        
            conv_1_imag = self.conv_layer_forward(tf.math.imag(tf.expand_dims(x,3)), layer_name=name+"/conv1_imag/",weights=weights)
            conv_2_imag = self.conv_layer_forward(conv_1_imag, layer_name=name+"/conv2_imag/",weights=weights)
            conv_3_imag = self.conv_layer_forward(conv_2_imag,layer_name=name+"/conv3_imag/",weights=weights) 
            conv_4_imag = self.conv_layer_forward(conv_3_imag,layer_name=name+"/conv4_imag/",weights=weights) 
            y_rec_imag = self.conv_layer_forward(conv_4_imag,layer_name=name+"/conv5_imag/",weights=weights,relu=0)  
            y_rec=tf.squeeze(tf.complex(y_rec_real, y_rec_imag),axis=-1)
            return y_rec    
    

    def cascade(self,x,name):
        with tf.compat.v1.variable_scope(name):
            initiate=tf.initializers.glorot_uniform(seed=2000)
            conv_1_real = self.conv_layer(tf.math.real(tf.expand_dims(x,3)),shape=[3,3,1,64], layer_name="conv1_real",initializer=initiate)
            conv_2_real = self.conv_layer(conv_1_real,shape=[3,3,64,64], layer_name="conv2_real",initializer=initiate)
            conv_3_real = self.conv_layer(conv_2_real,shape=[3,3,64,64],layer_name="conv3_real",initializer=initiate) 
            conv_4_real = self.conv_layer(conv_3_real,shape=[3,3,64,64],layer_name="conv4_real",initializer=initiate) 
            y_rec_real = self.conv_layer(conv_4_real,shape=[3,3,64,1],layer_name="conv5_real",initializer=initiate,relu=0)
        
            conv_1_imag = self.conv_layer(tf.math.imag(tf.expand_dims(x,3)), shape=[3,3,1,64],layer_name="conv1_imag",initializer=initiate)
            conv_2_imag = self.conv_layer(conv_1_imag,shape=[3,3,64,64], layer_name="conv2_imag",initializer=initiate)
            conv_3_imag = self.conv_layer(conv_2_imag,shape=[3,3,64,64],layer_name="conv3_imag",initializer=initiate) 
            conv_4_imag = self.conv_layer(conv_3_imag,shape=[3,3,64,64],layer_name="conv4_imag",initializer=initiate) 
            y_rec_imag = self.conv_layer(conv_4_imag,shape=[3,3,64,1],layer_name="conv5_imag",initializer=initiate,relu=0)
            y_rec=tf.squeeze(tf.complex(y_rec_real, y_rec_imag),axis=-1)
            return y_rec  
            
    def fusion_layer(self,x,y,name):
        with tf.compat.v1.variable_scope(name):
            y_rec_real = self.conv_layer_constraint(tf.compat.v1.real(tf.concat([tf.expand_dims(x,3),tf.expand_dims(y,3)],3)),shape=[1,1,2,1], layer_name="fusion_real",initializer=tf.compat.v1.constant_initializer(value=0.5),relu=0)

            y_rec_imag = self.conv_layer_constraint(tf.compat.v1.imag(tf.concat([tf.expand_dims(x,3),tf.expand_dims(y,3)],3)),shape=[1,1,2,1], layer_name="fusion_imag",initializer=tf.compat.v1.constant_initializer(value=0.5),relu=0)
            y_rec=tf.squeeze(tf.complex(y_rec_real, y_rec_imag),axis=-1)
            return y_rec    

    def fusion_layer_forward(self,x,y,weights,name):
        with tf.compat.v1.variable_scope(name):
            y_rec_real = self.conv_layer_constraint_forward(tf.compat.v1.real(tf.concat([tf.expand_dims(x,3),tf.expand_dims(y,3)],3)),shape=[1,1,2,1], layer_name=name+"/fusion_real/",weights=weights,relu=0)
            y_rec_imag = self.conv_layer_constraint_forward(tf.compat.v1.imag(tf.concat([tf.expand_dims(x,3),tf.expand_dims(y,3)],3)),shape=[1,1,2,1], layer_name=name+"/fusion_imag/",weights=weights,relu=0)
            y_rec=tf.squeeze(tf.complex(y_rec_real, y_rec_imag),axis=-1)
            return y_rec               
     
    def data_consistency(self,x,y,us_masks,coil_sens_maps,coil_n):       
        x_k_space=tf.signal.fft2d(x)    
        y_kspace=tf.signal.fft2d(y)
        indices=tf.tile(tf.expand_dims(us_masks,axis=1),[1,coil_n,1,1])
        y_kspace=tf.where(indices,x_k_space,y_kspace)
    #   #IFFT    
        y=tf.signal.ifft2d(y_kspace)   
        return y
    

    def kernel_consistency(self,x,kernel,coil_n,coil_sens_maps):       
        x=tf.tile(tf.expand_dims(x,axis=1),[1,coil_n,1,1,1])
        x=tf.reduce_sum(tf.multiply(x,kernel),2)     
        return x        
    
  
    def adj_proj(self,x,coil_sens_maps):       
        x=tf.reduce_sum(tf.multiply(x,tf.math.conj(coil_sens_maps)),1)
        return x     
        

    def forw_proj(self,x,coil_sens_maps,coil_n): 
        x=tf.tile(tf.expand_dims(x,axis=1),[1,coil_n,1,1])
        x=tf.multiply(x,coil_sens_maps)
        return x             
             

  

import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn


      
def transform_image_to_complex(k):
    """ Compu"""
    us_complex=np.zeros((k.shape[0],k.shape[1],k.shape[2],1),dtype=np.complex_)
    for ii in range(k.shape[0]):   
      us_complex[ii,:,:,0].real=k[ii,:,:,0]
      us_complex[ii,:,:,0].imag=k[ii,:,:,1]
    return us_complex
    
    
def centered_crop(img, new_height, new_width):
   width =  np.size(img,1)
   height =  np.size(img,0)
   left = int(np.ceil((width - new_width)/2.))
   top = int(np.ceil((height - new_height)/2.))
   right = int(np.floor((width + new_width)/2.))
   bottom = int(np.floor((height + new_height)/2.))
   cImg = img[top:bottom, left:right,:]
   return cImg
   
 
def sum_of_square(im,axis=0):
    im = np.sqrt(np.sum((im*im.conjugate()),axis=axis))
    return im
def ifft2c(k, dim=None, img_shape=None):

    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def fft2c(img, dim=None, k_shape=None):

    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k

# PSFNet

The technique (PSFNet) is described in the [following](https://www.sciencedirect.com/science/article/pii/S0010482523010752) paper:

Dar SUH, Öztürk Ş, Özbey M, Oguz KK, Çukur T. Parallel-stream fusion of scan-specific and scan-general priors for learning deep MRI reconstruction in low-data regimes. 2023.
![PSFNet](PSFNet.png)

## Demo

The code trains and tests PFSNet models for reconstruction of post contrast T1-weighted images from the fastMRI dataset. Files containing  training, validation and testing data can be obtained by contacting us at (salmanulhassan.dar[at]med.uni-heidelberg.de) and proving that fastMRI data sharing agrrement has been signed.   <br />

<br />
To run the code on other datasets, modify data_loader function in the code. The code expects data_loder to return the following variables:
images_us, images_fs, masks, coil_maps,int_kernel,samples
<br />
images_us - a list of k elements, where each element is a batch containing zero-filled reconstructions of undersampled data represented by 4D variables and dimensions (1, 2, 3, 4) correspond to (samples, coils, x-size, y-size) <br />
images_fs - a list of k elements, where each element is a batch containing reference images represented by 3D variables and dimensions (1, 2, 3) correspond to (samples,  x-size, y-size) <br />
masks - a list of k elements, where each element is a batch containing undersampling masks represented by 3D variables and dimensions (1, 2, 3) correspond to (samples,  number of samples, x-size, y-size) <br />
coil_maps - a list of k elements, where each element is a batch containing coil sensitivity maps represented by 4D variables and dimensions (1, 2, 3, 4) correspond to (samples, coils, number of samples, x-size, y-size) <br />
int_kernels - a list of k elements, where elementis is a batch containing interpolation kernels represented by 5D variables and dimensions (1, 2, 3, 4, 5) correspond to (samples, coils, coils, kernel-x, kernel-x) <br />
The interpolation kernels and coil sensitivity maps can be obtained via the ESPIRiT (https://people.eecs.berkeley.edu/~mlustig/Software.html) tool box.

### Training and testing
python PSFNet.py --dat_dir data/ --ckpt_dir ckpt/ --results_dir results/
 <br />
 <br />
data_dir - data directory  <br />
ckpt_dir - checkpoints directory <br />
results_dir - results directory

### Dependencies
The code has been tested in the following settings:<br />
Ubuntu [18.04] 
Python [3.6.9]  <br />
CUDA [11.2]  <br />
Tensorflow [2.6.2] <br />


## Citation
You are encouraged to modify/distribute this code. However, please acknowledge this code and cite the paper appropriately.
```
@article{DAR_PSFNet,
title = {Parallel-stream fusion of scan-specific and scan-general priors for learning deep MRI reconstruction in low-data regimes},
journal = {Computers in Biology and Medicine},
volume = {167},
pages = {107610},
year = {2023},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2023.107610},
url = {https://www.sciencedirect.com/science/article/pii/S0010482523010752},
author = {Salman Ul Hassan Dar and Şaban Öztürk and Muzaffer Özbey and Kader Karli Oguz and Tolga Çukur},
keywords = {Image reconstruction, Deep learning, Scan specific, Scan general, Low data, Supervised, Unsupervised}
}
```
For any questions, comments and contributions, please contact Salman Dar (salmanulhassan.dar[at]med.uni-heidelberg.de) <br />

(c) ICON Lab 2023





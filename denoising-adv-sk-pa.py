
# coding: utf-8

# Exploring deep priors of adversarial images by Sourabh Kulkarni and Pradeep Ambati.
# 
# - Import clean image examples from imagenet
# - Apply adversarial preturbations
# - Pass through deep image prior based denoiser
# - Compare adversarial and denoised images
# - Use pretrained alexnet classifier to predict classes
# 
# Note: To see overfitting set `num_iter` to a large value.
# Original code from: https://github.com/DmitryUlyanov/deep-image-prior

# # Import libs

from __future__ import print_function
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from models import *

import foolbox
import torch
import torch.optim
import torchvision.models as t_models
import requests
from scipy.misc import imsave
from skimage.measure import compare_psnr
from utils.denoising_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.

print ('GPU(s) detected:', torch.cuda.get_device_name(0))


# # enter filename 

# path = 'C:/Users/Sourabh Kulkarni/Documents/CS682/Project/deep-image-prior/data/denoising/'

# fname = path + 'ga_adv_class_543.jpg' # bird
# #fname = path +'untargeted_adv_img_from_390_to_397.jpg' # eel


LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'

# Let's get our class labels.
response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
labels = {int(key): value for key, value in response.json().items()}

resnet18 = t_models.resnet18(pretrained=True).eval()
resnet18.double()
if torch.cuda.is_available():
    resnet18 = resnet18.cuda()
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
fmodel = foolbox.models.PyTorchModel(
    resnet18, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))

# get source image and label
image, label = foolbox.utils.imagenet_example(data_format='channels_first')
image = image / 255.  # because our model expects values in [0, 1]
image = np.array(image, dtype=np.double)
print('label', label, labels[label])
prediction = np.argmax(fmodel.predictions(image))
print('predicted class', prediction, labels[prediction])
# apply attack on source image

label = np.int64(label)

attack = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image, label)

print('adversarial class', labels[np.argmax(fmodel.predictions(adversarial))])

#imsave('C:/Users/Sourabh Kulkarni/Documents/CS682/Project/deep-image-prior/data/denoising/adversarial.png',np.transpose(adversarial*255,(1, 2, 0)))

# pass adversarial image to deep prior framework
img_noisy_pil = np.transpose(adversarial*255,(1, 2, 0))
img_noisy_np = pil_to_np(img_noisy_pil)

# As we don't have ground truth
img_pil = img_noisy_pil
img_np = img_noisy_np
img_pil = np.transpose(adversarial*255,(1, 2, 0))
img_np = pil_to_np(img_pil)
    
#Add Extra Noise
#img_pil, img_np = get_noisy_image(img_np, sigma_)
if PLOT:
        plot_image_grid([img_np, img_noisy_np], 4, 6)


# # Setup


INPUT = 'noise' # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net' # 'net,input'

reg_noise_std = 1./30. # set to 1./20. for sigma=50
LR = 0.01 # defalut 0.01

OPTIMIZER='adam' # 'LBFGS'
show_every = 100
exp_weight=0.99


num_iter = 2400
input_depth = 3
figsize = 5 

net = skip(
            input_depth, 3, 
            num_channels_down = [8, 16, 32, 64, 64], # original [8, 16, 32, 64, 128]
            num_channels_up   = [8, 16, 32, 64, 64], # original [8, 16, 32, 64, 128]
            num_channels_skip = [0, 0, 0, 4, 4], 
            upsample_mode='bilinear',
            need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU')

net = net.type(dtype)


#net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
net_input = get_noise(input_depth, INPUT, (img_pil.shape[1], img_pil.shape[0])).type(dtype).detach()

print (net_input)

# Compute number of parameters
s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
print ('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)


# # Optimize

# alexnet = t_models.alexnet(pretrained=True)
# resnet = t_models.resnet101(pretrained=True)

# alexnet.eval()
# resnet.eval()
# alexnet = alexnet.double()
# resnet = resnet.double()

#img_np_normal = (img_np - np.reshape([0.485, 0.456, 0.406],(3,1,1))/np.reshape([0.229, 0.224, 0.225],(3,1,1))).astype(float)
#img_np_normal /= 255
#print (img_np.shape)
img_np_changed = (img_np) / 255
img_np_changed = np.array(image, dtype=np.double)
prediction = np.argmax(fmodel.predictions(img_np_changed))
print ('Initial adversarial image prediction for resnet18:', prediction, labels[prediction])

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0

i = 0
def closure():
    
    global i, out_avg, psrn_noisy_last, last_net, net_input
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
    
    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
            
    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()
        
    
    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0]) 
   

    print ('Iteration %05d    Loss %f   PSNR_noisy: %f   ' % (i, total_loss.item(), psrn_noisy), '\r', end='')
    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        # plot_image_grid([np.clip(out_np, 0, 1), 
        #                  np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)
        #out_np_normal = (out_np - np.reshape([0.485, 0.456, 0.406],(3,1,1))/np.reshape([0.229, 0.224, 0.225],(3,1,1)))
        out_np_changed = (out_np) / 255
        out_np_changed = np.array(image, dtype=np.double)
        predictions = np.argmax(fmodel.predictions(out_np_changed))
        print ('resnet18 prediction for current iteration:', predictions, labels[predictions])
        
    
    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5: 
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.detach().copy_(new_param.cuda())

            return total_loss*0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy
            
    i += 1

    return total_loss




p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)


out_np = torch_to_np(net(net_input))
q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13);


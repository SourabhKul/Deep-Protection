
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

import time
import foolbox
import torch
import torch.optim
import torchvision.models as t_models
import requests
from os import listdir
from os.path import isfile, join
from scipy.misc import imsave
from skimage.measure import compare_psnr
from utils.denoising_utils import *
from utils.data_utils import load_imagenet_val

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

imsize =-1
PLOT = True
sigma = 25
sigma_ = sigma/255.

print ('GPU(s) detected:', torch.cuda.get_device_name(0))

# Get ImageNet labels

LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
response = requests.get(LABELS_URL)  # Make an HTTP GET request and store the response.
labels = {int(key): value for key, value in response.json().items()}

# Get 25 ImageNet validation images, preprocess them for foolbox

X, y, class_names = load_imagenet_val(num=25)
X = X / 255
X_reshaped = np.moveaxis(X,3,1)

for img in y:
    print (img, ',', labels[img])

# Set up a classifer (resnet 18), and a foolbox model to attck that classifier.

resnet18 = t_models.resnet18(pretrained=True).eval()
resnet18.double()
if torch.cuda.is_available():
    resnet18 = resnet18.cuda()
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
fmodel = foolbox.models.PyTorchModel(
    resnet18, bounds=(0, 1), num_classes=1000, preprocessing=(mean, std))


#attack = foolbox.attacks.FGSM(fmodel)
# attack = foolbox.attacks.AdditiveGaussianNoiseAttack(fmodel)
# attack = foolbox.attacks.BlendedUniformNoiseAttack(fmodel)
# attack = foolbox.attacks.GaussianBlurAttack(fmodel)
# attack = foolbox.attacks.NewtonFoolAttack(model=fmodel)



def closure():
    
    global i, out_avg, psrn_noisy_last, last_net, net_input, correct, flag
    
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
   

    # print ('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
    if  PLOT and i % show_every == 0:
        out_np = torch_to_np(out)
        # plot_image_grid([np.clip(out_np, 0, 1), 
        #                  np.clip(torch_to_np(out_avg), 0, 1)], factor=figsize, nrow=1)
        #out_np_normal = (out_np - np.reshape([0.485, 0.456, 0.406],(3,1,1))/np.reshape([0.229, 0.224, 0.225],(3,1,1)))
        predictions = np.argsort(fmodel.predictions(np.array(out_np, dtype=np.double)))[-5:][::-1]
        print ('resnet18 prediction for current iteration:', predictions[0], labels[predictions[0]])
        print ('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')
        if (true_label in predictions):
            if ((adversarial_label not in predictions) or ((adversarial_label in predictions) and (predictions.tolist().index(adversarial_label) >= predictions.tolist().index(true_label)))):
                print ('Non-adversarial prior obtained at iteration', i, 'top', predictions.tolist().index(true_label)+1,'/5')
                correct += 1
                flag = True
            else:
                flag = False

    
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

    if not flag: 
        return total_loss 
    else: 
        flag = False
        return 0
begin = time.time()

# Create a list of foolbox attack models

attacks = [foolbox.attacks.FGSM(fmodel), foolbox.attacks.AdditiveGaussianNoiseAttack(fmodel), foolbox.attacks.BlendedUniformNoiseAttack(fmodel), foolbox.attacks.GaussianBlurAttack(fmodel), foolbox.attacks.NewtonFoolAttack(model=fmodel)]

for attack in attacks:
    print (attack.name)
    
    skipped = 0
    correct = 0
    i = 0
    flag = False

    for img in range(25):
        true_label = y[img]
        prediction = np.argmax(fmodel.predictions(X_reshaped[img]))
        #prediction2 = np.argmax(fmodel.predictions(np.flip(X_reshaped[img],axis=1)))
        print('predicted class', prediction, labels[prediction], true_label)

        if prediction != true_label:
            print ('Prediction falied, skipping this one')
            correct += 1
            pass
        
        # apply different attacks on source image

        # Generate adversarial images, check if they are indeed adversarial

        adversarial_image = attack(X_reshaped[img], y[img])
        prediction3 = np.argmax(fmodel.predictions(np.flip(adversarial_image,axis=2)))
        adversarial_label = np.argmax(fmodel.predictions(adversarial_image))
        print('adversarial class', labels[adversarial_label], 'flipped', prediction3, labels[prediction3])

        # Save adversatial image
        #imsave('C:/Users/Sourabh Kulkarni/Documents/CS682/Project/deep-image-prior/data/denoising/adversarial.png',np.transpose(adversarial*255,(1, 2, 0)))

        # pass adversarial image to deep prior framework

        img_noisy_pil = np.transpose(adversarial_image*255,(1, 2, 0))
        img_noisy_np = pil_to_np(img_noisy_pil)

        # As we don't have ground truth

        img_pil = img_noisy_pil
        img_np = img_noisy_np
        img_pil = np.transpose(adversarial_image*255,(1, 2, 0))
        img_np = pil_to_np(img_pil)
        

        # Add Extra Noise, if needed
        #img_pil, img_np = get_noisy_image(img_np, sigma_)

        # if PLOT:
        #         plot_image_grid([img_np, img_noisy_np], 4, 6)


        # Deep image prior setup


        INPUT = 'noise' # 'meshgrid'
        pad = 'reflection'
        OPT_OVER = 'net' # 'net,input'

        reg_noise_std = 1./30. # set to 1./20. for sigma=50
        LR = 0.01 # defalut 0.01

        OPTIMIZER='adam' # 'LBFGS'
        show_every = 50 # 100
        exp_weight=0.99


        num_iter = 3000
        input_depth = 3
        figsize = 5 

        net = skip(
                    input_depth, 3, 
                    num_channels_down = [8, 16, 32, 64, 128], # original [8, 16, 32, 64, 128]
                    num_channels_up   = [8, 16, 32, 64, 128], # original [8, 16, 32, 64, 128]
                    num_channels_skip = [0, 0, 0, 4, 4], 
                    upsample_mode='bilinear',
                    need_sigmoid=True, need_bias=True, pad=pad, act_fun='Swish')

        net = net.type(dtype)


        #net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
        net_input = get_noise(input_depth, INPUT, (img_pil.shape[1], img_pil.shape[0])).type(dtype).detach()

        # Compute number of parameters
        s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
        print ('Number of params: %d' % s)

        # Loss
        mse = torch.nn.MSELoss().type(dtype)
        img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
        # prediction = np.argmax(fmodel.predictions(np.array(img_np/255, dtype=np.double)))
        # print ('Initial adversarial image prediction for resnet18:', prediction, labels[prediction])

        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()
        out_avg = None
        last_net = None
        psrn_noisy_last = 0

        i = 0

        p = get_params(OPT_OVER, net, net_input)
        optimize(OPTIMIZER, p, closure, LR, num_iter)
        print (correct, 'out of ', img+1)
        print ('Total time elapsed: ', int((time.time()-begin)/60), 'mins')

    print ('total correct ', correct, 'skipped', skipped)


# out_np = torch_to_np(net(net_input))
# q = plot_image_grid([np.clip(out_np, 0, 1), img_np], factor=13);


# CustomSegmentation
Instance/Semantic/Panoptic Segmentation on Custom Dataset

Training SMP model with Catalyst (high-level framework for PyTorch), TTAch (TTA library for PyTorch) and Albumentations (fast image augmentation library) 

# Dataset
https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset

# Good Structure for main()

Using Wand

use lr scheduler, using multiple optimizer
adding time - time.time() - check when to use it
# Good code
use zip for train_test_split
remaining_set, train_set = train_test_split(zip(image_paths,mask_paths), test_size=0.2, random_state=1) # no cannot do it

good way to split dataset
oganized training, testing evaluation steps
no need to convert to tensor in trsnaform anymore, since the image is now from torchvision.io.read_image now

only need to have train loader and eval loader since for testing we test on single image not the batch

Using 3 losses at the same time + SOTA optimizer (not Adam but Lookahead+RAdam)
## Semantic segmentation models
RefineNet
DeepLabV3+
more models from https://github.com/yassouali/pytorch-segmentation/tree/master
## Instance and panoptic segmentation models taking from lectures of CV3

## what to remmember
Calculating accuracies should always detach and convert to cpu cos it will consumes lots of memory otherwise

For segmentation - have to appy the transform for both image and mask (s.t rotating, fllipping) and cannot use normalization (cos the mask value is the class number not the value) (maybe can still apply to input images but not really sure how to do it)

Cross entropy loss (input,target)

where taget should have the shape of (B, H,W) not (B, 1, H,W) -should squeeeze it 
as well as the value of target should be in long
also input should have less 1 dimension than target- zB taget size (M,N) then input should have (M) only,etc
so basically input should be as input.squeeze(dim=-1).long().to(device)

Don't forget convert model, input to cuda

## What are left
using Dataparalle to utilize multiple GPUs
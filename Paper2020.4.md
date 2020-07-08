# **Detection**

## 1.Feature Pyramid Grids

https://arxiv.org/abs/2004.03580

文章提出了一种FPN设计范式Feature Pyramid Grids,由multi-scale backbone pathway和多级平行的pyramid pathway构成,(FPN可以看作只有一条pyramid pathway).不同scale的feature map之间由4种不同的lateral connections 连接构建出grids网格阵列结构.不属于网络搜索,该结构为人工设计.为优化速度和性能之间的trade-off,文章总结了两条实验性的设计规则，在不太影响性能情况下减小网络复杂度：1. 平行pathway之间的AcrossUp连接可以去除,层（hi-res）特征之间的"三角形"连接可以减少.以此得到
contracted FPG,FPG在多种检测网络架构上相较于原版FPN均在不引入明显计算量的情况下提升2+%AP(COCO),而相比NAS-FPN在two-stage网络上也有1-2%AP提升.

## 2.Detection in Crowded Scenes: One Proposal, Multiple Predictions

https://arxiv.org/pdf/2003.09163.pdf

本文提出了一种简单的解决密集场景检测的的方法,主要思路是使每一个proposal不只是预测一个instance，而是同时预测多个correlated instances,本文同时提出了Earth Mover's Distance (EMD)Loss，和Set NMS来增强训练和预测的过程在CrowdHuman数据集上，本文的方法可以提升4.9% AP。在CItyPersons数据集上， 提升了1%MR-2,在并不是非常密集的COCO数据及上，效果也有略微提升.

## 3.YOLOv4: Optimal Speed and Accuracy of Object Detection

https://arxiv.org/pdf/2004.10934.pdf

https://github.com/AlexeyAB/darknet

本文主要目的是设计一类可以部署在GPU上的高效的实时检测模型,本文尝试了很多detection的提升性能的方法（bag of freebies, bag of specials）,做了很多实验一一验证每种方法的作用本文同时改进了几个SOTA方法（CBN,PAN,SAM），使得模型在GPU上更高效,最后得出的YOLOv4，在同样AP的情况下，速度远高于EfficientDet和ASFF.

# **Segmentation/Action Recognition/Pose Estimation/Video**

## 1.X3D: Expanding Architectures for Efficient Video Recognition

https://arxiv.org/pdf/2004.04730.pdf

https://github.com/facebookresearch/SlowFast

本文提出了X3D，一系列由逐步放大6个不同维度（对应图中六个gamma）而形成的video recognition模型,本文用MobileNetV2的inverted bottleneck block搭建出来的网络作为基础结构,每个步骤，尝试从6个维度放大网络，然后选择其中一个最好的。这样重复十几次，得到越来越大的模型,X3D中的不同大小的模型，在kinetics上达到SOTA精度的情况下，计算量比slowfast小5倍左右,作者发现higher spatiotemporal resolution相比于wider model更容易提升video recognition的性能.


# **Regularization/Distillation and Network Structure**

## 1.MUXConv: Information Multiplexing in Convolutional Neural Networks

https://arxiv.org/pdf/2003.13880.pdf

https://github.com/human-analysis/MUXConv


文章提出MUXConv，把spatial information flow分解到不同的大小再合并,得到一个高效的可以替代普通Conv的操作,以MUXConv作为building block，提出了一种NAS算法来同时优化网络的compactness,efficiency,得到一组网络叫MUXNets,在CIFAR10/100,ImageNet的分类问题上，MUXNets在小模型上取得了SOTA的效果.

## 2.ResNeSt: Split-Attention Networks

https://hangzhang.org/files/resnest.pdf

本文提出一种通用BackBone网络架构ResNeSt，相比于参数量相当的ResNet-50网络，ResNeSt在分类、检测、分割等多种任务上都取得了很大的性能提升；与同复杂度的NAS网络EfficientNet相比，也有微小优势。ResNeSt 中的S表示 Split-Attention，该网络基础Block的设计综合了 Group Conv (ResNeXt) Channel-Attention (SE-NET) Multi-Path (GoogleNet) or Feature-Map Attention (SKNET)的思想，主要可以分为三个步骤：按channel划分K个cardinal group，每个Group的输入channel数为C/K, 每个cardinal group内部分出R个同构分支（类似ResNeXt，但不同于SKNET和GoogleNet里的不同size的kernel）,R个分支做Split-Attention，即每个分支均为SE-NET的基础结构，最后做elementwise加在一起.本文可算做一个“集大成”的工作，除网络设计之外，还比较全面地总结了很多训练trick，如Large Minibatch,Label Smoothing,AutoAug等。

## 3.AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty

https://hangzhang.org/files/resnest.pdf

https://github.com/google-research/augmix

本文提出了一种叫作AugMix的新的data augmentation方法，能提升模型的accuray，并且同时提升robustness和在data shift情况下的uncertainty estimates.AugMix把一张输入图片，经过多组，每组多个不同的augmentation，然后再和原始图片mix到一起,本文同时提出用JS Divergence Consistency Loss来minimize原图和经过augmix的图的prediction，使得模型的输出更加smooth,在CIFAR-10/100上，AugMix效果好于CutMix,AutoAugment等同类方法取得了SOTA的accuracy,在CIFAR-10/100的corrupted version上，AugMix的accuracy效果远超Mixup, Adversarial Training等方法（15-20%）.

# **GAN and NAS**

## 1.Evolving Normalization Activation Layers

https://arxiv.org/pdf/2004.02967.pdf

提出把Normalization和Acitvation当做一个整体,同时优化.提出了一个Layer Search的方法，在搜索过程中,要对于各种不同的网络结构进行优化,搜索到了几个新的Normalization+Acitvation的组合,在ResNets,MobileNets,EfficientNets上面都达到了比BN,GN更好的效果.搜索到的组合在Mask R-CNN,BigGan等其他任务上表现也由于现有工作.

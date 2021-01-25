# **Detection**

## 1.PP-YOLO: An Effective and Efficient Implementation of Object Detector
https://arxiv.org/pdf/2007.12099.pdf

https://github.com/PaddlePaddle/PaddleDetection

本文基于Yolo-v3的detector结构和resnet backbone，尝试加多种detection tricks组合，提升检测的性能和效率，加入的tricks有large batch size, EMA, DropBlock, IoU Loss, Grid Sensitive, Matirx NMS, CoordConv, SPP等，本文故意没有用到的tricks有，换其他backbone结构，更多更好的data augmentation和hyperparamter search。作者认为加上这些性能可以进一步提升。

## 2.Corner Proposal Network for Anchor-free, Two-stage Object Detection
https://arxiv.org/pdf/2007.13816.pdf

https://github.com/Duankaiwen/CPNDet

ECCV2020的文章，本文对比了two-stage, one-stage, center-based和corner-based几种不同的detection方法，指出corner based方法的recall会更高一些，本文指出anchor-free的方法虽然recall比较高，但是precision会偏低，这个问题可以用twostage来解决，作者提出了Corner Proposal Network (CPN)，1ststage基于CornerNet的方法，找出一些proposal，然后再由第二个stage做更精确的classification，CPN在COCO detection任务上AP和AR都超过了其他anchor-free的方法。

## 3.RepPoints v2: Verification Meets Regression for Object Detection
https://arxiv.org/pdf/2007.08508.pdf

https://github.com/Scalsol/RepPointsV2

本文指出传统的Object detection方法一般有verification 和 regression 两个步骤，其中
verification 是coarse localization using anchors.而近期的一些anchor free的方法就只有
localization，并没有verification，本文提出在localization的基础上加上一个auxiliary Verification Branch，来提升objectdetection的效果，在RepPoints的基础上，加入Verification Branch,提出RepPoints v2， 在COCO上有2.0 AP左右的提升，Verification还可以加在其他方法中，例如FCOS，同样能提升AP。

## 4.Feature Pyramid Transformer
https://arxiv.org/pdf/2007.09451.pdf

https://github.com/ZHANGDONG-NJUST/FPT

ECCV2020，本文指出spatial context在各种视觉任务里都很重要，而普通的CNN并没有刻意利用context信息，即使是non-local也只是在某一个scale上面的操作，没有Multi-scale的信息，作者提出Feature Pyramid Transformer(FPT)来增强不同scale和space的interaction，本文提出三种transformer: Self-Transformer(ST), Grounding Transformer (GT), Rendering Transformer (RT)分别处理三种scale 变化的情况下的feature transformation，FPT是一个通用的结构， 在object detection,instance segmentation和sementic segmentation任务中都可以明显提升性能。

## 5.Dive Deeper Into Box for Object Detection
https://arxiv.org/pdf/2007.14350.pdf

ECCV2020，本文指出在anchor-free detection问题中，一般confidence score较高的box作为输出的
prediction，可是这些box可能并不够准确作者提出了 decomposition and recombination(D&R)module 和 semantic consistency module，来提升box localization的准度加入了两个module，能使得detection AP从33.6% 提升到38%。基于FCOS, 本文提出了DDBNet, 在COCOdetection上取得了近似SOTA的效果。

## 6.Semi-Anchored Detector for One-Stage Object Detection
https://arxiv.org/pdf/2009.04989.pdf

作者指出class-imbalance是one-stage detector不
够准的主要原因质疑
最近提出的一些anchor-free方法在classification
上一般更准确，但是regression相对不准
本文提出一种semi-anchored的架构，通过
classification找出正例的位置，然后匹配多个
anchor来做regression
在同样Backbone和FPN情况下，在COCO上AP比
RetinaNet高了3.7%
用ResNet-101-FPN在COCO上达到了43.6的AP, 高
于RetinaNet的39.1和FCOS的41.0


# **Segmentation/Action Recognition/Pose Estimation/Video**


# **Regularization/Distillation and Network Structure**

## 1.Learning Connectivity of Neural Networks from a Topological Perspective
https://arxiv.org/pdf/2008.08261.pdf

ECCV2020，本文指出之前的网络设计大多是简单重复模块的叠加，本文试图从connectivity learning的角,度，找到更加高效的结构本文把网络不同层之间的连接看成graph中的edge，然后在训练的过程中学习每个edge的权重，最后去除掉不必要的连接。本文提出的方法可以用在不同的网络上，例如ResNet, MobileNet作者设计并提出了TopoNet，在跟ResNet计算量相似的情况下，可以在ImageNet上提升2.1%的accuracy，在COCO上AP提升了超过5%

## 2.Visual Concept Reasoning Networks
https://arxiv.org/pdf/2008.11783.pdf

本文指出之前的一些network design，虽然有用
到一些split-transform-merge来用不同的branch
学习不同的feature/concepts，但是这些结构大
多是local的，缺少更高层面的reasoning
作者提出了Visual Concept Reasoning Networks
(VCRNet)来提升high-level concept中间的
reasoning
作者提出了concept sampler, concept reasoner 和
concept modulator几个模块，整个结构流程是
split-transform-attend-interact-modulatemerge，
可以直接加在ResNeXt 里面
在classification, segmentation, detection等任务
中，用VCRNet都能在增加很少计算量的情况
下，取得比其他同类方法(SE, CBAM, GC)更好的
效果。

## 3.Activate or Not: Learning Customized Activation
https://arxiv.org/pdf/2009.04759.pdf

本文提出了一种新的activation function: ACON
作者指出Swish是ReLU的一种smooth
approximation，都是更general的ACON的special
case
提出了meta-ACON, 学习activate neurons的程度
用ACON作为activation function, 增加很少计算量
的情况下，在各种backbone上效果都由于ReLU
和Swish


# **GAN && NAS && Others**


## 1.What Should Not Be Contrastive in Contrastive Learning
https://arxiv.org/pdf/2008.05659.pdf

本文指出近期的一些contrastive learning的方法会假设一些invariance, 例如color, rotation,
texture等。但是这些假设在后面的任务中并不一定成立，作者提出一种新的 Leave-one-out Contrastive Learning (LooC) 方法，不需要知道关于任务相关的invariance，LooC方法提出用多个embedding space, 每一个对于多种（除了特定一种）data augmentation invariant。LooC方法在各种下游任务中（ coarse-grained,fine-grained, and few-shot downstream classification）都得到了更好的表现



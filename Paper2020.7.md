# **Detection**

## 1.LabelEnc: A New Intermediate Supervision Method for Object Detection

https://arxiv.org/pdf/2007.03282.pdf

提出了 LabelEnc方法， 把类别label先encode到一个embedding space，并且在训练detector的过程中加一个auxiliary intermediate supervision，提出了相应的两步训练方法，先用一个autoencoder的结构训练label encoding， 然后再训练一个带有auxiliary loss的object detector，这个方法可以用在各种其他detection方法上，
包括2-stage, 1-stage, anchor-free等方法，都可以提高1~2%的AP，由于这个方法只是针对training过程，在
inference的时候就去掉了，所以完全不会影响inference速度

## 2.Point-Set Anchors for Object Detection, Instance Segmentation and Pose Estimation

https://arxiv.org/pdf/2007.02846.pdf

作者指出近期的一些工作(CenterNet)，利用中心点做keypoint regression的方法是有缺陷的，主要原因是很多关键点离中心点可能很远，所以中心店的feature并不能够很好的用来判断其他点的信息,为了解决上述问题，作者提出了根据任务的数据,设计Point-set Anchor来做regression的目标，这样anchor和gt的距离更近，更容易学习,这个方法可以很用来替代普通detection模型中的anchor，来解决instance segementation和pose estimation等问题,Point-set Anchor在detection和instance segmentation任务上都接近one-stage方法的SOTA水平，并且在pose estimation任务上超过了其他的regression based方法.

## 3.AutoAssign: Differentiable Label Assignment for Dense Object Detection

https://arxiv.org/pdf/2007.03496.pdf

本文提出了一个新的anchor-free的检测方法，主要提出了一个新的label assignment策略AutoAssign,AutoAssign可以动态的生成positive和negative weight maps,提出了两个weighting modules, center weighting和confidence weighting来根据不同的class和instance来调整distribution,在相同backbone的情况下，在COCO上的AP明显优于其他anchor-free方法

## 4.BorderDet: Border Feature for Dense Object Detection

https://arxiv.org/pdf/2007.11056.pdf

https://github.com/Megvii-BaseDetection/BorderDet

本文指出一般的object detector只利用一个位置的feature来预测classification和box regression，这样信息不足。作者提出了BorderAlign操作，补充加入边界特征（border feature）增强关于物体的信息，使得预测更准确。作者相应的提出了Border Alignment Modules (BAM)和BorderDet检测网络,这个方法可以应用在single-stage(RetinaNet,FCOS)或者two-stage的方法上，可以提升约2~3%的AP.

# **Segmentation/Action Recognition/Pose Estimation/Video**


# **Regularization/Distillation and Network Structure**

## 1.ReXNet: Diminishing Representational Bottleneck on Convolutional Neural Network

https://arxiv.org/pdf/2007.00992.pdf

https://github.com/clovaai/rexnet

本文指出传统的network design会造成一些representational bottleneck，影响整个模型的性能，作者做了大量的实验，尝试不同的configuration并观察不同层生成feature的matrix rank，观察到了模型的性能与rank非常相关，提出了三个design principle:增大输入channel数；用一个合适的nonlinearity (swish-1/ReLU6)；在网络中加入很多的expland layers；通过上面的几条准则，提出了Rank eXpansion Networks (ReXNets)，效果超过了NAS搜索出的EfficientNet模型，ReXNets在transfer learning 和detection任务上也有很好的表现

## 2.WeightNet: Revisiting the Design Space of Weight Networks

https://arxiv.org/pdf/2007.11823.pdf

提出了一种新的简洁，高效的weight generating方法WeightNet (WN)，结合了Squeeze and Excitation和CondConv,WeightNet在SE和CondConv的基础上，在attention activation中加了一个Grouped FC 层，
并通过WN来输出convolution的weight,WN可以加在其他backbone中，在几乎不增加FLOPs的情况下，提升image classification, object detection的性能.


## 3.Funnel Activation for Visual Recognition

https://arxiv.org/pdf/2007.11824.pdf

https://github.com/megvii-model/FunnelAct

提出了Funnel activation (FReLU), y = max(x, T(x)),其中T(x) 是受周围影响的spatial contextual
feature extractor,T(x)在文中的默认实现方式就是一个3x3的depthwise convolution，这样计算量和参数都很
少，同时输出了一个和输入size相同的feature map,用FReLU代替ReLU/Swish，在不同任务不同模型上，都可以明显提高性能.

# **GAN and NAS**


## 1.Smooth Adversarial Training

https://arxiv.org/pdf/2006.14536.pdf

本文指出ReLU activation的non-smooth性质造成了adversarial training之后的效果不够好,提出了Smooth Adversarial Training (SAT), 把relu换成其他相似的smooth activation，使其在backward propogation时的gradient 更好,使用smooth activation不会降低模型的accuracy, 不会提升计算量，同时在不同大小模型上可以提升10%左右的robustness.
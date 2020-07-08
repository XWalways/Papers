# **Detection**
## 1.Computation Reallocation for Object Detection

https://arxiv.org/pdf/1912.11234.pdf

文章首先提出了直接用分类网络做检测backbone是sub-optimal,不同scale FPN层上有效感受野和anchor size存在mismatch可以验证这一观点,提出了两个阶段网络search方案Computation Reallocation NAS,确定每个stage中的block输入输出shape和总block数的前提下,先搜每个stage最优的block数目，再用greedy algorithm搜每个block里conv的dilation rate,以上两个阶段分别被称为stage allocation in different resolution, convolution allocation in spatial position,得到的CR-Resnet50 faster-rcnn 相较baseline在COCO上提升1.9%.Maskrcnn在COCO分割任务上也有1.3%的提升.

## 2.Soft Anchor-Point Object Detection

https://arxiv.org/pdf/1911.12448.pdf

基于FSAF33改进得到一种 anchor free 的detection 网络——Soft Anchor-Point Detector,anchor free 的detector存在 positive anchor points 的选择（文中叫how to make head better）和feature layer的选择两个问题,主要通过以下两种soft weights的操作缓解以上两类问题：a.soft-weighted anchor points: 给原本positive anchor points不同的weights,离valid bbox中心更近的会assign更高的权重.b. soft-selected pyramid levels: 类似FSAF引入feature layer选择网络,输出一个概率分布而不是某一层，给每一层不同的权重,多层同时对一个instance进行预测；用FSAF
中定义的选layer标准得到one-hot gt作supervise,相比其他anchor-free以及two stage的网络,在性能和
速度上均有明显优势。SAPD-R50在COCO上保持41.7AP可以达到14.9FPS(1080ti batch size=1),SAPDX101-DCN 可以达到47.4%AP.

# **Segmentation/Action Recognition/Pose Estimation/Video**

## 1.More Is Less: Learning Efficient Video Representations by Big-Little Network and Depthwise Temporal Aggregation

https://arxiv.org/abs/1912.00869

https://github.com/IBM/bLVNet-TAM

基于Big-Little-Net提出了一个针对video recongnition的轻量级网络bLVNet,提出了一个Temporal Aggregation Module (TAM),可以有效的处理temporal relation信息，并且不需要3D Conv,TAM由简单的depthwise convolution以及temporal shifting组成，计算量很小,在Kinectics,Something-Something数据集上,在同样计算量限制的情况下,得到了比其他方法更高的准确率,把同类方法Temporal shift module7换成TAM,在计算量增加很小的情况下,accuracy 可以提升至少2%.

## 2.EmbedMask: Embedding Coupling for One-stage Instance Segmentation

https://arxiv.org/pdf/1912.01954.pdf

https://github.com/yinghdb/EmbedMask

提出了porposal embedding和pixel embedding来把pixels通过相似的embedding来assign给相对应的instance结合了proposal-based和segmentation-based的方法,作为一个one-stage instance segmentation方法,达到了接近two-stage SOTA的mask AP, 取得了更好的efficiency

## 3.STAGE: Spatio-Temporal Attention on Graph Entities for Video Action Detection

https://arxiv.org/pdf/1912.04316.pdf

https://github.com/aimagelab/STAGE_action_detection

提出了一个新的video action detection 模型结构,利用spatio-temporal graph来处理视频中人物和物品的关系,这个方法可以接在各种video backbone后面，来完成action detection,在AVA数据集上取得了SOTA的mAP.


## 4.YOLACT++: Better Real-time Instance Segmentation

https://arxiv.org/pdf/1912.06218.pdf

https://github.com/dbolya/yolact

提出了YOLACT, 一种single-stage的instance segementation方法，速度可以达到real time,在YOLACT的基础上,加入了deformable convolution,优化了prediction head里面的anchor scale和ratio,提
出了一个快速高效的mask re-scoring branch,可以在计算量增加很小的情况下提升准确率,相比YOLOACT在同样速度下,AP提升了4个点

## 5.BlendMask: Top-Down Meets Bottom-Up for Instance Segmentation

https://arxiv.org/pdf/2001.00309.pdf

提出了一个blender模块，可以结合instancelevel和dense pixel level的信息,来输出准确的
instance mask,提出了BlendMask，在FCOS的基础上加较少计算量就可以完成instance segmentation任务,BlendMask也可以直接用来做panoptic segmentation任务,在COCO instance segmentation任务上，在AP略微超越Mask R-CNN的模型，速度上也快了大约20%.

## 6.Deep Snake for Real-Time Instance Segmentation

https://arxiv.org/pdf/2001.01629.pdf


提出了一种新颖的基于contour来实现instance segmentation的方法,提出用learning based snake algorithm来实现real-time的contour预测,此方法首先做contour proposal,然后再这个基础上做contour deformation,在cityscapes的instance segmentation任务上达到了SOTA的AP和速度.

# **Regularization and Network Structure**

## 1.Dynamic Convolution: Attention over Convolution Kernels

https://arxiv.org/pdf/1912.03458.pdf


提出了Dynamic Convolution操作来代替普通的Convolution,Dynamic Convolution利用K个平行的kernels, 并且根据输入图片计算出每个的重要性,然后做weighted sum,再以这个结果进行convolution
operation,Dynamic Convolution相对于普通的convolution计算量增加很小（约4%）,用Dynamic Convolution替换掉MobileNetV2/V3里面的普通convolution，在imagenet上面top-1accuracy可以提升2.3~4%,在COCO keypoint上面，也可以提升1.6~4.9的AP.

## 2.Local Context Normalization: Revisiting Local Normalization

https://arxiv.org/pdf/1912.05845.pdf

提出了Local Context Normalization (LCN), 在Group Normalization的基础上，只在spatial neighborhood做normalization LCN与GN类似，不受batch size影响,在object detection, semantic segmentation, instance segmentation任务上，用LCN的效果略高于BN,GN.

## 3.AdderNet: Do We Really Need Multiplications in Deep Learning?

https://arxiv.org/pdf/1912.13200.pdf

提出了AdderNet，用加法取代乘法,用l1 loss取代卷积操作，以此来加速模型运算,给AdderNet的加法设计了一个叫SignSGD的新optimizer,以LeNet-5为样板的AdderNet有1.7M延迟,普通CNN有2.6M延迟,以ResNet为样板的AdderNet在ImageNet上的分类成绩接近CNN,但是据查阅的资料,GPU上乘法有优化，应该和加法速度接近，所以有一定争议

## 4.GridMask Data Augmentation

https://arxiv.org/pdf/2001.04086.pdf

提出了一个新的data augmentation方法GridMask,GridMask相比于同类型的cutout,high-and-seek
都更加有效的完成随机的information removal,GridMask可以加在各种不同任务中,在imagenet classification上面，用resnet50,GridMask达到了比AutoAugment更好的accuracy(77.9 vs 77.6),在COCO object detection 和 instance segmentation任务上,使用GridMask也能带来明显提升


## 5.Compounding the Performance Improvements of Assembled Techniques in a Convolutional Neural Network

https://arxiv.org/pdf/2001.06268.pdf

https://github.com/clovaai/assembled-cnn

注意到近些年来在Image classification问题上,有很多提高准确率的方法被提出,但是几乎没有
尝试过把各种方法都混在一起使用,这篇文章做了很多实验,加入了很多能够提升image classification的方法,包括ResNet1D, SK, Anti-alias, DropBlock, BigLittleNet, AutoAugment,Mixup, Label Smoothing, Knowledge Distillation.最终训练出的模型，相对于之前的SOTA efficientNet, 在相同精度的情况下,速度快了3倍.
# **NAS and Others**

## 1.SpineNet: Learning Scale-Permuted Backbone for Recognition and Localization

https://arxiv.org/pdf/1912.05027.pdf

提出scale-permuted network,不同于一般的backbone模型不断缩小spatial dimension,
SpineNet打乱不同feature map的顺序，试图提升multi-scale recognition and localization的性能,具体permutation经过NAS搜索得到几个不同的模型,在同样One-stage/two-stage detection方法下,
只改变backbone, 几个SpineNet都以更小的计算量达到了更高的AP.
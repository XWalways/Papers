# **Useful Trick**
Bag of Tricks for Image Classification with Convolutional Neural Networks

https://arxiv.org/abs/1812.01187

——列举在ImageNet数据集上提高分类性能的训练技巧

——以ResNet-50为例，将top-1准确率从75.3%提高的79.29%


有源码,挺有用

# **Detection and Counting**

## 1.Towards Adversarially Robust Object Detection 

https://arxiv.org/pdf/1907.10786.pdf

将detection 看作多任务学习,分析了各类对抗攻击方式对task losses的影响,提出了面向对抗攻击的鲁棒detection.

## 2.A Survey of Deep Learning-based Object Detection

https://arxiv.org/pdf/1907.09408.pdf

对深度学习做Object detection的综述,涵盖多种主流架构及不同configuration的在COCO 2017上的比较


## 3.DR Loss: Improving Object Detection by Distributional Ranking

https://arxiv.org/abs/1907.10156

修改loss函数处理样本不平衡问题,通过只修改loss在COCO数据集上把mAP从39.1%提高到41.1%

## 4.BASNet: Boundary-Aware Salient Object Detection

http://openaccess.thecvf.com/content_CVPR_2019/papers/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.pdf

https://github.com/NathanUA/BASNet

关注边界的显著性检测,Loss设计上,使用了交叉熵,结构相似性损失,IoU 损失这三种的混合损失,使网络更关注于边界质量,而不是像以前那样只关注区域精度

## 5. It’s All About The Scale - Efficient Text Detection Using Adaptive Scaling

https://arxiv.org/pdf/1907.12122.pdf

针对text detection任务中,text在图片里面scale不同的问题,以往的工作都是用放大的图片当input/多张不同尺度缩放的图片当input,造成计算量大,这篇工作先在缩小的照片预测出text segmentation和scale,再将不同scale的text region缩放到适合尺寸送到Text extractor中,实验表明此方法效果提升显著（特别是在小图input上）

## 6.Grid R-CNN Plus: Faster and Better

Grid RCNN: https://arxiv.org/abs/1811.12030

Grid RCNN Plus: https://arxiv.org/abs/1906.05688

https://github.com/STVIR/Grid-R-CNN

原版Grid RCNN收录在CVPR 2019; 商汤/港中文/北航原版的 Grid R-CNN 对 Faster RCNN 做了很多精度上的优化,但是速度却慢于 Faster R-CNN,于是 Grid R-CNN Plus就速度优化在四个方面进行了改进：1.网格点特定表示区域;2.轻量网格分支;3.跨图片采样策略;4.一次性 NMS

## 7.Counting with Focus for Free 

https://arxiv.org/pdf/1903.12206.pdf

https://github.com/shizenglin/Counting-with-Focus-for-Free

解决单张图片多个object的计数问题,数量由几十到几千个annotation是每个object所在的点,以往的方法是构建density map当作学习目标,文章将annotation转化为density map + segmentation map + global density map, 训练三个network branch来更好拟合density map. 认为构建的segmentation map / global density map为最后的预测提供了更好的聚焦,在多个object counting数据集上均达到SOTA


## 8.MatrixNets: A New Deep Architecture for Object Detection

https://arxiv.org/pdf/1908.04646.pdf

利用 MatrixNet 加强基于关键点的目标检测,并且在 MS COCO 数据集上获得了 47.8 的 mAP,这比其它任何一步（single-shot）检测器效果都要好.而且参数量减半,重要的是，相比效果第二好的架构,在训练上要快了 3 倍.
针对目标检测中物体有着不同的宽高比(aspect ratio),提出matrix networks 如下图所示：对角线上斜向下的为FPN中的下采样层,而向下或向右为对高度或宽度下采样的层,从而使方形卷积核捕捉不同宽高比的信息.

## 9.ThunderNet: Towards Real-time Generic Object Detection

https://arxiv.org/pdf/1903.11752.pdf

从backbone和detection两个方面都提出了几个提升效率的方法：

Backbone方面:
基于shufflenetV2, 使用5x5 conv来替代3x3来提升receptive field,
提出 the input resolution should match the capability of the backbone（而不是在任何resolution都用同样的backbone）,
Detector方面:
压缩RPN:用5x5 depthwise separable conv来替代3x3 conv,
压缩detection head: 在light-head RCNN的基础上减小channel,
Context Enhancement Module (CEM):把不同大小feature map以较低的计算量进行整合,
Spatial Attention Module (SAM):整合RPN和CEM的feature来做attention,
最终效果，其中较大的一个模型,在COCO上mmAP可以达到28.0,速度方面在ARM, CIP, GPU上分别可以达到5.8, 15.3, 214的FPS

## 10.Propose-and-Attend Single Shot Detector

https://arxiv.org/pdf/1907.12736v1.pdf

提出了Propose-and-Attend module， 用来模仿2-stage detector中的ROI-Pooling操作,使得detection module的输入更加贴合需预测的object,Proposing: 模型需要预测出bounding box offset,Atteding: 通过之前预测出的offset, 修改convolution的sampling locations,(似于deformable conv, 但是这里的offset是由proposing module预测出来的)达到类似于ROI-pooling的效果,再把convolution的结果给到后面做常规的detection prediction (box offsite prediction, classification),最终模型PASSD-768 与主体结构类似以及速度类似的RetinaNet500相比,AP由34.4提升到了40.3,Paper中提到的SOTA是M2Det的41.0,但是结构不同,可以与本文的Propose-and-Attend互补

## 11.Revisiting Feature Alignment for One-stage Object Detection

https://arxiv.org/pdf/1908.01570.pdf

本文与Propose-and-Attend Single Shot Detector的核心思想完全一样,RepPoints也提出了非常相似的操作,注意到一般one-stage detector不够准确是因为进入detector的feature和anchor并没有对齐.而two-stage中,进入detector branch的feature map是通过(align)ROIPooling 来和object box对齐的,提出ROIConv来模仿ROIPooling,实际上就是先predict GT box相对于anchor的offset, 然后利用offset做deformable conv.在RetinaNet模型基础上,加上ROIConv, 用COCO数据集进行训练和测试.ResNet-101 FPN 作为backbone, AP从39.1上升到42.0, inference time 104ms → 110ms.ResNeXt-101 (32x8d) FPN 作为backbone, AP从40.8上升到44.1, inference time 177ms → 180ms

## 12.PosNeg-Balanced Anchors with Aligned Features for Single-Shot Object Detection

https://arxiv.org/pdf/1908.03295.pdf

https://github.com/zxhr2793/PADet

提出了anchor promotion module (APM) 和 feature alignment module (FAM)
APM可以通过调整anchor,使得更多的anchor和gt box有更大的IOU, 从而增加positive anchor, 减少 negative anchor,
FAM通过APM给出的offset, 经过一个小的network做一些额外的学习处理得到最终offset, （这是与AlignDet不同的地方）来提取更准确的feature,
在COCO object detection问题上:
VGG16作为backbone, AP达到了35.2, FPS有62.5 (之前的SOTA PFPNet 在同样AP的FPS是22.2)
ResNet101作为backbone, AP达到了40.0, FPS有28.6 (之前的RetinaNet 在AP 39.1的FPS是~10)

# **Segmentation and Pose Estimation**

## 1.ET-Net: A Generic Edge-aTtention Guidance Network for Medical Image Segmentation

https://arxiv.org/pdf/1907.10936.pdf

MICCAI 2019,天津大学的文章,值得参考的地方是用了edge-attention提升segmentation性能,通用在各个医疗图片分割任务上,作者申称会公布代码放在 https://github.com/ZzzJzzZ/ETNet

## 2.Interpretability Beyond Classification Output: Semantic Bottleneck Networks

https://arxiv.org/pdf/1907.10882.pdf

旨在为deep segmentation 架构提供可解释性,文章在传统segmentation network中加入Semantic Bottleneck(SB),,输出低维的semantic concept. 比较亮点的是加入SB结构后仍然能达到SOTA performance, 同时SB结构使得error case study变得容易,利用SB结构输出,也可以对segmentation结果区间进行置信度打分


## 3.High-Resolution Representations for Labeling Pixels and Regions
https://arxiv.org/abs/1902.09212

## 4.Deep High-Resolution Representation Learning for Human Pose Estimation
https://arxiv.org/abs/1904.04514


提出HRNet:持续保留high resolution feature map,repeated multi-scale fusion,
在多个与pixel位置有关的任务上,都达到了很好的效果,部分达到了SOTA
可以在多种任务上代替ResNet作为backbone, 在不增加计算量的同时,提升accuracy/precision

## 5.Instance Segmentation by Jointly Optimizing Spatial Embeddings and Clustering Bandwidth

https://arxiv.org/pdf/1906.11109.pdf

https://github.com/davyneven/SpatialEmbeddings

提出一种proposal-free的instance segmenation方法,同时兼顾速度与准确率

对于每个pixel,预测出一个Offset vector, 指向其所在instance 的centorid,
centroid并非一个点,而是一个margin以内的范围,
有一个branch用来学习一个class-specific sedd maps, 来确认cluster center,
在Cityscapes上面,达到比Mask R-CNN略高的AP,只需要11FPS, 而Mask R-CNN的FPS只有2.

# **Convolution and Structure**

## 1.MixNet: Mixed Depthwise Convolutional Kernels

https://arxiv.org/abs/1907.09595v1

https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet/mixnet

提出新的convolution机制MDConv, 可与普通cnn替换,用于多种任务,使用autoML和MDConv, 提出新模型MixNets,MixNets 在COCO object detection超过基线模型2%, ImageNet under mobile settings做到SOTA

## 2.Compact Global Descriptor for Neural Networks

https://arxiv.org/pdf/1907.09665.pdf

https://github.com/HolmesShuan/Compact-Global-Descriptor

提出了Compact Global Descriptor (CGD) module,是一种功能类似于non-local block,而且计算量小到可以忽略的module,在video classification task上面，在附加计算量是non-local 的1/100的情况下，达到了同样/更高的accuracy
在imagenet classification task上面,
加在mobilenet上,效果优于其他所有类似SOTA方法，且附加计算量只是SE的1/16
加在resnet50上,效果和其他SOTA方法相当,并且附加计算量是SE的1/100.(效果最好的是CBAM, 附加计算量是CGD的270倍，不过总计算量只多了1.7%)
在COCO detection/segmentation上,加了CGD之后AP 有 1.2~1.5的提升,附加计算量远小于NL,可以忽略

## 3.Attentive Normalization

https://arxiv.org/pdf/1908.01259.pdf

提出了Attentive Normalization, 把attention的机制直接加在normalization layer里面,节省计算量,原本normalization后面只学习一组affine transform, 变成学习K组affine transform,然后通过另一个轻量级branch来学习不同组的权重,权重是instance-specific,也就是同一个batch里面每个instance都会有不一样的结果,在ImageNet Classification 问题上,在几乎不增加计算量的情况下,加在resnet, densenet, mobilenet上之后的accuracy都有约1%的提升,在COCO detection/segmentation上,加在网络里面AP也有> 1.5%的提升

## 4.HBONet: Harmonious Bottleneck on Two Orthogonal Dimensions

https://arxiv.org/pdf/1908.03888.pdf

https://github.com/d-li14/HBONet

提出一种新的bottleneck unit结构,
基于MobileNet-V2中的inverted bottleneck, 在spatial dimension上做类似的contraction-expansion操作,
最后的concat操作,其实是把input feature map的后一半channels 叠加在前面,这样保证channel数量不变,
在ImageNet 上面,和MobileNetV2进行对比:在相似的计算量的前提下,HBONet 比MobileNetV2的top-1 accuracy 提高了0.9 ~ 6.6,计算量越小,提升越明显,作为backbone, 在VOC07 detection问题上,相对于同等级的mobilenetV2, mAP提升了0.6 ~ 6.3,在Market1501的ReID问题上,相对于同等级的MobileNetV2,mAP提升了3.6 ~ 3.9, rank-1 提升了1.7 ~ 5.0


# **Optimizer**

## 1.Lookahead Optimizer: k steps forward, 1 step back

https://arxiv.org/abs/1907.08610v1

https://github.com/alphadl/lookahead.pytorch

多伦多大学向量实验室论文,adam原作和hinton为2,3作以前的optimizer论文,着重于利historical gradients自适应学习率 如Adam/AdamGradaccelerated scheme 如 Nesterov momentum/ Polyak heavy-ball作者提出方法与上述方法正交维护两套参数.一次更新:快参数<--慢参数,快参数多次更新后,慢参数朝快参数方向更新一次,作者称此方法超参数少,训练robust. 在多数据集上接近或是超过Adam/SGD+momentum




# **Detection**

## 1.EfficientDet: Scalable and Efficient Object Detection

https://arxiv.org/pdf/1911.09070.pdf

在COCO object detection任务上远超SOTA.
EfficientDet-D3可以达到44.3的AP,同时GPU inference time只有42ms.

## 2.You Only Watch Once: A Unified CNN Architecture for Real-Time Spatiotemporal Action Localization

https://arxiv.org/pdf/1911.06644.pdf

https://github.com/wei-tim/YOWO

提出You Only Watch Once (YOWO) 模型,来处理spatio-temporal action localization的问题
YOWO是一个single-stage的方法（对应YOLO）可以高效的检测到一个视频中的action的时间以及空间位置
速度可以在16帧视频上达到34FPS, 在8帧视频上达到62FPS,在J-HMDB-21和UCF101-24上达到了SOTA的frame-mAP.

## 3.Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression

https://arxiv.org/pdf/1911.08287.pdf

https://github.com/Zzh-tju/DIoU

提出Distance-IoU Loss (DIoU),更好的描述一个prediction box与ground-truth box的相关性,来帮助bounding box regression 的训练,convergence比GIoU loss要快
在DIoU loss基础上提出Complete IoU loss (CIoU), 考虑了overlap area, central point distance和aspect ratio
DIoU可以用在NMS过程中,效果优于普通NMS
用CIoU loss加上DIoU-NMS可以在COCO detection上得到2%的AP提升,在较大object上面提升更明显.

## 4.Real-time Text Detection with Differentiable Binarization

https://arxiv.org/pdf/1911.08947.pdf

https://github.com/MhLiao/DB

使用ResNet-18作为基础网络,速度快
方法仅在传统的分割网络后加入一个thresholdmap以及一个DB模块,几乎不影响速度,但提升了精度
后续将会代码开源,所po出的性能均为目前SOTA,超越了CRAFT


## 5.Learning Spatial Fusion for Single-Shot Object Detectio

http://xxx.itp.ac.cn/pdf/1911.09516

https://github.com/ruinmessi/ASFF

提出一种data driven的FPN,以很小的计算代价大幅提升YOLOv3的精度,coco上38.1% AP at 60 FPS,
以3个level的特征融合为例,分别在spatial维度上学习3个level特征融合时的weight: alpha beta gama,对应位置相乘后相加进行融合.

## 6.Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection

https://arxiv.org/pdf/1912.02424

https://github.com/sfzhang15/ATSS

首先用几乎具有完全相同架构,参数的RetinaNet（一个location只设置一个方形anchor）和FCOS做实验验证了本文主要观点：anchor-based 和 anchor-free的方法关键区别仅在于在训练时如何选择positive和negtive的samples,这一点都做到一致,performance几乎没有区别,
提出Adaptive Training Sample Selection (ATSS)选择pos和neg的算法,即每层feature map每个gt取最近的9个anchor作为候选集,算这些候选anchor与gt IoU的 mean (u) 和standard deviation (v),取阈值为t = u + v, 大于t且中心落在gt里的选为pos,其余为neg,
此方法对原RetinaNet提升2.x个点,且通过实验发现,采用这个算法default anchor的ratio、scale以及数量都不影响最终的performance,
加上Deformable Conv等一些操作后在COCO能达到mAP=50.7,准SOTA结果.但未有inference 速度和FLOPS报告.

# **Segmentation/Action Recognition/Pose Estimation/Video**

## 1.Video Representation Learning by Dense Predictive Coding

https://arxiv.org/pdf/1909.04656.pdf


提出了Dense Predictive Coding (DPC), 通过预测未来feature representation来做self-supervised represnetation learning on videos
提出了一种curriculum training的方法,训练过程中不断减少temporal context, 来达到更好的semantic representation
在Kinetics-400上面做self-supervised pre-training之后,再用来做其他的video task,在只用RGB输入的情况下,达到了比其他self-supervised方法更好的效果,在UCF101上面，accuracy 达到了75.7,超过了imangenet pre-trained的73.0.

## 2.Grouped Spatial-Temporal Aggregation for Efficient Action Recognition

https://arxiv.org/pdf/1909.13130.pdf

提出了Grouped Spatial-Temporal (GST) Aggregation 模块,分别model Spatial, temporal 信息，然后进行整合
在Something-Something数据集上,在参数量是C3D一半的情况下,可以达到相似的accuracy
在只利用RGB输入的情况下，达到了与2-stream输入方法相似或更好的效果.

## 3.Learning Temporal Action Proposals With Fewer Labels

https://arxiv.org/pdf/1910.01286.pdf

提出了一种训练temporal action proposal的semi-supervised方法
效果远好于其他semi-supervised的方法,并且达到（甚至超越）了fully-supervised方法BSN的Average Recall
利用generated proposal来做action localization, 效果也达到甚至超越了BSN.


## 4.CenterMask:Real-Time Anchor-Free Instance Segmentation

https://arxiv.org/pdf/1911.06667.pdf

提出了一个快速高效的Instance segmentation方法CenterMask
CenterMask在 FCOS detector后面加了一个 Spatial Attention-Guided Mask (SAG-Mask) 来做mask输出
提出了VoVNetV2, 一个基于VoVNet增强的的高效backbone
在COCO segmentation任务上，速度不变的情况下,超过了之前single-stage的SOTA （YOLACT）
较大的一个单模型,速度和精度都略好于Mask R-CNN.

## 5.TEINet: Towards an Efficient Architecture for Video Recognition

https://arxiv.org/pdf/1911.09435.pdf

提出了Temporal Enhancement-and-Interaction (TEI) Modulem,可以加到其他2D CNN中,
TEI 包括了一个Motion Enhanced Module和一个Temporal Interaction Module来增强motion-related和temporal contextual的信息,
在video classification任务上,在相似的计算量/latency的情况下,比其他方法提高了准确率.

# **Regularization and Network Structure**

## 1.A closer look at network resolution for efficient network design

https://arxiv.org/pdf/1909.12978.pdf

指明了input resolution对于efficiency 的重要性
提出一个同时包括input resolution和network width的训练框架
这个训练框架可以用来训练任何模型结构
在同样计算量限制的情况下,可以比mobilenets, EfficientNet等结构达到更高的classification accuracy
与训练的模型在tranfer learning, object detection, instance segmentation上面也达到了很高的效率.

## 2.Deformable Kernels: Adapting Effective Receptive Fields for Object Deformation

https://arxiv.org/pdf/1910.02940.pdf

提出Deformable Kernel (DK),一种新的基础operation来调整effiective receiptive field
Defromable Conv是改变sampled data position, deformable kernel是改变sampled kernel weights
把DK加在不同的backbone上做imagenet classification,可以提升大概1%的accuracy
在COCO object detection任务上,DK可以提升2%的AP.

## 3.ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks

https://arxiv.org/pdf/1910.03151.pdf

https://github.com/BangguWu/ECANet

提出了一种非常高效的channel attention module, 利用简单的1D conv来实现
参数量增加远小于其他同类方法,可以忽略不计
计算量增加跟同类方法相比也很小,并且实际运行速度比SE要快
准确率上达到了同类方法SOTA相似水准
在COCO detection/segmentation任务上对于AP提升略好于SE,同时计算量更小


## 4.FlatteNet: A Simple Versatile Framework for Dense Pixelwise Prediction

https://arxiv.org/pdf/1909.09961.pdf

提出了 Flattening Module,可以代替很多任务（object detection, segmentation, keypoint localization...）中的decoder (upsampling) 部分。
可以加在各种backbone后面,在节省很多计算量的情况下,保持准确率几乎不变.


## 5.Hybrid Composition with IdleBlock: More Efficient Networks for Image Recognition

https://arxiv.org/pdf/1911.08609.pdf

提出了IdleBlock,将一部分channels不经过任何操作直接传到下一层， 可以减少很多计算量
提出Hybrid Composition (HC) with IdleBlock,通过混合不同的类型的Block,来传递/融合不同层的信息，并且控制计算量
把HC IdleBlock加在MobileNetV3以及EfficientNet中,可以提高speed/accuracy trade-off
HC(M=15, I=20) 模型在imagenet mobile setting取得了SOTA的77.5 top-1 accuracy, 并且只用了380 Madds.


# **NAS and Others**
## 1.Adversarial Examples Improve Image Recognition

https://arxiv.org/pdf/1911.09665v1.pdf

https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet


提出与先前对抗样本实验结论相反的观点:加入对抗样本有利于提升模型分类性能,
clean data与adversarial examples具有distribution mismatch,简单加入advExamples训练不能有效地对mixed,distribution 做representation learning,而本文提出的AdvProp能够取得更好的泛化性,
网络结构：加入辅助BN层,clean data通过master BN, advExamples 通过辅助BN,
训练策略：先用辅助BN生成advExamples,按上述途径优化两个loss,
使用EfficientNet在ImageNet上取得SOTA,比原EN-B7高0.7%,在ImageNet-A -C上高出6~7%.

## 2.Search to Distill: Pearls are Everywhere but not the Eyes

https://arxiv.org/pdf/1911.09074.pdf

提出了Architecture-aware Knowledge Distillation (AKD),寻找出最适合distill某个teacher model对应的一些stundet model,
搜索过程是一个基于RL的NAS,其中加入一个 Knowledge Distillation based reward
在imagenet classification task上,在不同的latency settings中，AKD都达到了SOTA.


# **Detection**

## 1.IoU-balanced Loss Functions for Single-stage Object Detection

https://arxiv.org/pdf/1908.05641.pdf

提出普通的cross-entropy loss和l1 loss并不适合做准确的object detection,
提出IoU-balanced classification loss来增强classification和localization之间的correlation,提出IoU-balanced localization loss来提升localization的训练效率,
在COCO detection任务上,用IoU-balanced Loss, 比baseline的AP可以提升1.1左右,
在越高的IoU treshold上,IoU-balanced Loss的AP提升越高

## 2.Teacher Supervises Students How to Learn From Partially Labeled Images for Facial Landmark Detection

https://arxiv.org/pdf/1908.02116.pdf

Facial Landmark Detection,提出一种Teacher Supervises Students的设计,一个teacher model监督两个结构不同的student models,在300-W上,用全部数据,得到了3.49的NME.只用20%数据,得到了5.03的NME,比其他semi-supervised方法要更准确

## 3.Character Region Awareness for Text Detection

https://arxiv.org/abs/1904.01941.pdf

https://github.com/clovaai/CRAFT-pytorch

自然场景文字检测SOTA,
算法开源,代码友好
算法基于字符检测,可扩展性强,
算法给出了一种若监督的训练方式,
多个trick提升速度.

## 4.Consistent Scale Normalization for Object Recognition

https://arxiv.org/pdf/1908.07323.pdf

提出了Consistent Scale Normalizatoin (CSN) 来减轻scale variation带来的问题,
CSN可以加在任何backbone上面,
在Object dection, instance segmentation和keypoint detection的任务上都能带来提升.

## 5.Residual Objectness for Imbalance Reduction

https://arxiv.org/pdf/1908.09075.pdf

提出了Residual Objectness模块,用来解决object detection中foreground background不平衡的问题,
可以加在one-stage和two-stage detector上面,
加在RetinaNet, YOLOv3 和 FasterRCNN 上面之后AP提升了至少3.2%.

## 6.Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network

https://arxiv.org/pdf/1908.05900.pdf

https://github.com/WenmuZhou/PAN.pytorch

使用了轻量级的特征提取网络（ResNet-18）,利用聚类完成文字检测
提出可串联的FPEM模块来提升检测性能,
实验结果兼顾了速度和精度,并不断更新源码.

# **Segmentation/Action Recognition/Pose Estimation**

## 1.STM: SpatioTemporal and Motion Encoding for Action Recognition

https://arxiv.org/pdf/1908.02486.pdf

提出SpatioTemporal and Motion (STM) block,包括Channel-wise SpatioTemporal Module (CSTM) 和 Channel-wise Motion Module (CMM) 来整合信息,
整个网络是2D CNN, 不需要3D Conv和optical flow 信息,
在几个temporal-related datasets上,例如something-somethingv1/v2, jester,STM得到了SOTA的准确率,
在scene-related datasets上,例如Kinetics, UCF-101, HMDB-51, STM取得了近似SOTA的准确率,
整个STM的计算量偏小，效率很高

## 2.FastPose: Towards Real-time Pose Estimation and Tracking via Scale-normalized Multi-task Networks

https://arxiv.org/pdf/1908.05593.pdf

提出一个multi-task模型,同时实现human detection, pose estimation, person re-ID,
提出scale-normalized image and feature pyramid (SIFP)来解决多尺度的问题,
利用pose信息来实现occlusion-aware, 从而提高Re-ID中用occlusion的情况,
实现了real-time pose tracking,取得了SOTA级别的accuracy/speed tradeoff

## 3.Expectation-Maximization Attention Networks for Semantic Segmentation

https://arxiv.org/pdf/1907.13426.pdf

把self-attention机制转化成一个expectation-maximization iteration的形式,减少了计算量,
提出了一个expectation-maximization attention unit (EMA Unit),
在增加较少计算量的前提下,在PASCAL VOC和COCO Stuff的semantic segmenation任务上取得了SOTA效果

## 4.Action recognition with spatial-temporal discriminative filter banks

https://arxiv.org/pdf/1908.07625.pdf

提出使用3个branch来做最后的classification:
Global feature branch,
Local feature branch,
Discriminative filter banch 使得网络更关注重要的location,
在Kinetics-400和Something-Something-V1数据集上都达到了SOTA的准确率.

## 5.TSM: Temporal Shift Module for Efficient Video Understanding

https://arxiv.org/pdf/1811.08383.pdf

https://github.com/mit-han-lab/temporal-shift-module

提出了Temporal Shift Module,在不增加计算量的前提下增强模型学习spatio-temporal 信息的能力,
提出bi-directional和uni-directional两种TSM, 分别应对offline和online两种计算量不同的需求,
在something-something和jester上面都达到了SOTA的水平.

# **Regularization**

## 1.Adaptive Regularization of Labels

https://arxiv.org/pdf/1908.05474.pdf

提出一种新的label regularization的方法
提出residual correlation matrix, residual label, 以及一个新的residual loss,可以减少overfitting,
在image classification (CIFAR-10/100, ImageNet),以及text classification (AGNews, Yahoo, Yelp), 在不用增加参数量的情况下,相对于baseline都有一些提升

## 2.DropBlock: A regularization method for convolutional networks

https://arxiv.org/abs/1810.12890

motivation: Dropout在全连接网络中效果较好，但是在卷积网络中由于空间相关性很高，信息流动依然很顺畅，因此不能很好地起到正则化的效果,
提出了一种Dropout的升级版,不是随机drop某些孤立的像素点,而是drop一个m*n的block,
inference时不增加任何计算量的前提下,imagenet ResNet-50识别正确率提高1.6%，检测分割等任务都有不同程度的提升.


# **NAS**
## 1.AutoML: A Survey of the State-of-the-Art

https://arxiv.org/pdf/1908.00709.pdf

关于Neural Architecture Search的一个survey


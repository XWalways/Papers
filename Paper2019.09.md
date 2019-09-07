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



# **Regularization**

## 1.Adaptive Regularization of Labels

https://arxiv.org/pdf/1908.05474.pdf

提出一种新的label regularization的方法
提出residual correlation matrix, residual label, 以及一个新的residual loss,可以减少overfitting,
在image classification (CIFAR-10/100, ImageNet),以及text classification (AGNews, Yahoo, Yelp), 在不用增加参数量的情况下,相对于baseline都有一些提升

# **NAS**
## 1.AutoML: A Survey of the State-of-the-Art

https://arxiv.org/pdf/1908.00709.pdf

关于Neural Architecture Search的一个survey


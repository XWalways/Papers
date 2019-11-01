# **Detection**

## 1.Geometry Normalization Networks for Accurate Scene Text Detection

https://arxiv.org/abs/1909.00794

通过分析说明了目前自然场景中文字检测主要的难点在于文字区域的大小和方向存在差异,
通过对数据集进行简单的放缩,旋转操作进行数据增强,提高了检测效果,
方法简单,效果出众.

## 2.FreeAnchor: Learning to Match Anchors for Visual Object Detection

https://arxiv.org/abs/1909.02466

https://github.com/zhangxiaosong18/FreeAnchor

提出了learning-to-match来做groundtruth bounding box和anchor box的matching,而不是像以前用IOU来做matching,
matching strategy是通过特殊设计的loss来学习的,以匹配到最合适的anchor box,
加在现有的object detector上面在COCO的detection任务上可以带来3个点的AP提升.

## 3.Cascade RPN: Delving into High-Quality Region Proposal Network with Adaptive Convolution

https://arxiv.org/pdf/1909.06720.pdf

https://github.com/thangvubk/Cascade-RPN

提出cascade PRN, 用较小的anchor数量,多次refine proposal box,来拿到比较准确的proposal,
提出adaptive covolution来提升convolution sampling的准确性（同期有好几篇文章提出了一样的想法）,
Cascade RPN在COCO上做proposal 的AR100达到了61.1,比普通RPN的AR1000=58.3 还要高,
速度上RPN 0.04s, Cascade RPN 0.06s,
在COCO detection任务上, 替换掉faster R-CNN中本来的PRN,mAP从37.1提升到了40.6.

## 4.Towards Unconstrained End-to-End Text Spotting

https://arxiv.org/pdf/1908.09231.pdf

这是一套end-to-end的文字检测+识别的方法,训练损失函数由检测和识别共同构成,可尽量少的避免检测和识别的错误叠加,
检测框架基于Mask RCNN,识别框架为attention,均为主流的方法,很简单的使用RoI Masking连接了两个branch,
值的注意的是,作者在实验中额外收集了30k网络数据辅助训练,并说明了数据变大会有效的提升网络效果（比较暴力）,
Spotting效果全面超越现有的spotting工作.

## 5.TextDragon: An End-to-End Framework for Arbitrary Shaped Text Spotting

http://openaccess.thecvf.com/content_ICCV_2019/papers/Feng_TextDragon_An_End-to-End_Framework_for_Arbitrary_Shaped_Text_Spotting_ICCV_2019_paper.pdf

提出了RoISlide来连接文字检测和识别,
端到端的识别,基于局部区域,但只需要词标注或行标注,适应现有公开数据集标注格式,
在弯曲文本/多方向文本自然场景下取得了SOTA.

# **Segmentation/Action Recognition/Pose Estimation**

## 1.SSAP: Single-Shot Instance Segmentation With Affinity Pyramid

https://arxiv.org/pdf/1909.01616.pdf


提出了 instance-aware pixel-pair affinity pyramid,并且在此基础上提出了一个one-stage的instance segmentation方法,
提出cascaded graph partition module 来提升segmentation的准确率,
在Cityscapes上达到了SOTA的水平.

## 2.Graph Convolutional Networks for Temporal Action Localization

https://arxiv.org/pdf/1909.03252.pdf

https://github.com/Alvin-Zeng/PGCN

提出了利用graph convolution来处理action proposal之间的relations,
针对这个问题提出了一种graph construction的方法,主要利用到tIOU来判断两个proposal是否需要有edge来连接,
在THUMOS14和ActivityNet v1.3上面比之前的SOTA提升了很多（3%~6%）.

## 3.Omni-Scale Feature Learning for Person Re-Identification

https://arxiv.org/pdf/1905.00953.pdf

https://github.com/KaiyangZhou/deep-person-reid

提出了Omni-Scale Feature Learning （类似于inception module）,
提出利用Unified Aggregation Gate 来整合不同scale的信息,
提出了OSNet,一个轻量级网络结构,
在Market1501, CUHK03, Duke, MSMT17数据集上用较小的OSNet达到了SOTA的mAP.

## 4.AdaptIS: Adaptive Instance Selection Network

https://arxiv.org/pdf/1909.07829.pdf

https://github.com/saic-vul/adaptis

提出了一种class-agnostic instance segmentation的方法,输入一张图片和一个点,输出这个点所属object的mask,
一个网络直接输出instance mask,不需要object detection stage,
在复杂形状和遮挡的情况下有更好的performance,
可以跟semantic segmentation的方法结合起来做Panoptic segmentation,
在Cityscapes和Mapillary数据集上取得了SOTA的成绩.

## 5.Anchor Loss: Modulating Loss Scale based on Prediction Difficulty

https://arxiv.org/pdf/1909.11155.pdf

提出了anchor loss, 可以根据prediction difficulty 来调整loss curve,
用anchor loss可以扩大true label和hard negative的gap,
anchor loss也可以用在pose esimation 任务上.

# **Regularization and Network Structure**

## 1.LIP: Local Importance-based Pooling

https://arxiv.org/pdf/1908.04156.pdf

https://github.com/sebgao/LIP

提出了Local Importance-based Pooling，可以代替ave/max pooling, 可以自动学习importance weights,
在imagenet classification上面能带来accuracy的提升,
在COCO detection上面只替换backbone中的pooling能提升1.5左右的AP,
在faster R-CNN FPN的框架中,加入LIP可以提升超过5个点的AP.

## 2.Second-order Non-local Attention Networks for Person Re-identification

https://arxiv.org/pdf/1909.00295.pdf

提出了一种新的 Second-order Non-local Attention Module,可以加在现有的backbone上面,几乎不增加computation,
在Market1501, CUHK03, DukeMTMC-reID数据及上取得了SOTA的成绩.

## 3.Linear Context Transform Block

https://arxiv.org/pdf/1909.03834.pdf

提出了 linear context transform (LCT) block, 类似于Squeeze-and-Excitation的模块,可以很容易的结合到backbone网络中,
计算量略小于SE, 参数量远小于SE,效果比SE要好,
在imagenet classification, COCO detection/segmentation任务上,可以在增加计算量很小的情况下,提升1.0~1.7个点.

## 4.Gated Channel Transformation for Visual Recognition

https://arxiv.org/pdf/1909.11519.pdf

提出了Gated Channel Transformation (GCT),来做类似于SE的attention,
GCT包括了Global Context Embedding, Channel Normalization和Gating Adaptation
GCT模块的参数量远小于SE, 计算量与SE相似,对于accuracy的提升比SE更明显,
在COCO的detection和segmentation任务上,加上GCT对AP有大概2个百分点的提升.

# **NAS and Others**

## 1.Artistic Glyph Image Synthesis via One-Stage Few-Shot Learning

http://arxiv.org/abs/1910.04987

https://hologerry.github.io/AGIS-Net/

可使用少量样本生成出艺术特效字（英文5,中文30）,
适应小样本学习,重新优化了损失函数,
该方法可对中文进行风格化迁移,代码开源,便于使用进行数据增强.

## 2.Seeing What GAN Cannot Generate

https://arxiv.org/pdf/1910.11626.pdf

https://github.com/davidbau/ganseeing

作者visually展示了GAN生成的图片跟实际图片相比缺少了哪些东西,对了解GAN的生成有一定的意义.


## 3.Guided Image-to-Image Translation with Bi-Directional Feature Transformation

https://arxiv.org/pdf/1910.11328v1.pdf

https://github.com/vt-vl-lab/Guided-pix2pix

特定的图片转换任务都会有特定的限定,本文研究如何将这些限定加入到转换生成,双向特征转换.

## 4. Distilling Object Detectors with Fine-grained Feature Imitation

http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Distilling_Object_Detectors_With_Fine-Grained_Feature_Imitation_CVPR_2019_paper.pdf

https://github.com/twangnh/Distilling-Object-Detectors

仅在物体出现的周围做feature mimic,比全图feature做mimic效果显著提升,
Faster R-CNN Res50在Res101的guild下在Pascal voc上AP可以提升3个点左右.
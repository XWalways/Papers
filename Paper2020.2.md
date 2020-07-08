# **Detection**
## 1.EHSOD: CAM-Guided End-to-end Hybrid-Supervised Object Detection with Cascade Refinement

https://arxiv.org/pdf/2002.07421.pdf

Hybrid-Supervised Object Detection 是目标检测领域较新的subtopic,关注如何混合利用少量的instance-level 标注数据和大量image-level 标注数据训练目标检测模型,输出分类和定位框结果,本文提出了一个基于二阶段检测架构(Cascade Faster-RCNN)的端到端的网络,可以将上述两类数据混合在一起训练.根据数据信息的不同结构,进行了不同的target assignment操作,加入了多个分支多种loss实现有效地监督训练,如在RPN结构上得到c-layer (num_class=c)的class activation map 经过GAP+softmax 与 image-lavel 的GT label计算loss,与先前的fully-supervised OD模型相比(SSD Retina Faster-RCNN FSAF AlignDet),在COCO数据集上，仅利用30/50%的instance-level标注数据,能取得相当或者更好的性能.




# **Segmentation/Action Recognition/Pose Estimation/Video**

## 1.Temporal Interlacing Network

https://arxiv.org/pdf/2001.06499.pdf

https://github.com/deepcs233/TIN

提出了一个高效的Temporal Interlacing Network用来融合temporal信息,可以加在2D网络中,增加很少的计算量,使其有处理video的能力,与类似方法不同的是,本文提出的temporal shifting是adaptive的,而不像同类方法shift模式
是固定的.在几个action recognition数据集上达到了SOTA,比同类的TSM高大概1%.

## 2.Audiovisual SlowFast Networks for Video Recognition

https://arxiv.org/pdf/2001.08740.pdf

https://github.com/facebookresearch/SlowFast

提出audiovisual slowfast network,在visual network的基础上，加入一个audio pathway,在多个尺度融合对视频进行预测,提出了DropPathway,用来在训练过程中随机关掉audio path way，来保证整体训练的稳定性.在Kinetics,Charades和AVA action detection任务上都取得了SOTA.

## 3.A Simple Framework for Contrastive Learning of Visual Representations

https://arxiv.org/pdf/2002.05709.pdf

提出了Mixed Temporal Convolution,利用几个不同kernel size的temporal convolution filter,来高效的完成spatiotemporal feature的提取Mixed Temporal Convolution可以很容易的加进2D模型中,便可用来训练video action
classification,在something-something和jester数据集上取得了SOTA.



# **Regularization/Distillation and Network Structure**

## 1.Compounding the Performance Improvements of Assembled Techniques in a Convolutional Neural Network

https://arxiv.org/pdf/2001.06268.pdf

https://github.com/clovaai/assembled-cnn

注意到近些年来在Image classification问题上,有很多提高准确率的方法被提出，但是几乎没有尝试过把各种方法都混在一起使用,这篇文章做了很多实验，加入了很多能够提升image classification的方法，包括ResNet1D,SK,Anti-alias,DropBlock, BigLittleNet, AutoAugment,Mixup,Label Smoothing,Knowledge Distillation.最终训练出的模型，相对于之前的SOTAefficientNet, 在相同精度的情况下,速度快了3倍.

## 2.Subclass Distillation

https://arxiv.org/pdf/2002.03936v1


提出Subclass Distillation (SC-G)的方法，在多分类数据集上将数据划分为二分类,训练Teacher Model时加入subclass head,经过softmax得到按多个子类的概率分布后再按照划分分组相加得到最后二分类的概率分布.在训练TM时引入Auxiliary Loss,避免出现某个子类分数过高的情况,同时也引入了更多的information entropy.通过subclass head 和 auxiliary loss 在二分类的数据上训练(CIFAR-2*5),TM会学习到子类信息.ablation 实验显示,在(CIFAR-10)上测试多分类性能有显著提升.注意,并没有提供子类hard targets,SC-D 相较于传统Distillation 和 penultimate
distillation有更快的SM训练速度和准确率,在实际subclass 数量与head 设计数量不match时,甚至数据no known subclass 时,SC-D也能学到更多semantic subclass信息.

## 3.Towards Stabilizing Batch Statistics in Backward Propagation of Batch Normalization

https://arxiv.org/pdf/2001.06838.pdf

https://github.com/megvii-model/MABN

分析出BatchNorm在小batch的情况下不好的原因是因为在back propogation过程中有两个gradient term非常不稳定,提出用 simple moving average statistics (SMAS)和 exponential moving average statistics (EMAS)来解决batch statistics不稳定的问题.提出简化BatchNorm的形式,只用E(X^2)来做normalization,而不是mean和std,减少不稳定
性,最终结果在小batch size上也能达到和大batch的效果,在inference速度上,由于没有group的操作，所以速度和BN一样,甚至可以融合进conv操作里.

## 4.Residual Knowledge Distillation

https://arxiv.org/pdf/2002.09168.pdf

提出了Residual Knowledge Distillation (RKD),一种新的模型压缩方法.不同于其他方法一般只做一轮distillation,RKD会通过训练一个assistant 模型来学习Student 和 Teacher之间的residual error,RKD可以应用在其他现有的Knowledge Distillation方法之上来提升效果,提出了一个高效的model separation方法,使得在不增加总体参数量的情况下,能同时兼顾Student和Assitant模型在CIFAR-100和ImageNet的任务上,取得了SOTA的效果.

## 5.Triple Wins: Boosting Accuracy, Robustness and Efficiency Together by Enabling Input-Adaptive Inference

https://arxiv.org/pdf/2002.10025.pdf

提出了一种multi-exit网络结构，在加强了efficiency的同时,提升了accuracy以及robustness.提出Robust Dynamic Inference Networks (RDI-Nets),不同的输入会经过模型不同的path输出结果.通过多个loss,作者针对这个架构提出了一些attack和defense的方法,最终相比于普通模型提升了robustness.


# **GAN and NAS**

## 1.CNN-generated images are surprisingly easy to spot... for now

https://arxiv.org/pdf/1912.11035.pdf

https://peterwang512.github.io/CNNDetection/

目的:探索有没有一个“通用”的fake images的检测器,方法:收集11中生成器的数据,careful 前处理和数据增强,训练一个标准的分类器,结论：CNN生成的图片有合成的共性,使得fake image从real image能够分离出来


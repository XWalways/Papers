# **Detection**

## 1.FBNetV3: Joint Architecture-Recipe Search using Neural Acquisition Function

https://arxiv.org/pdf/2006.02049.pdf

本文指出训练的方式（方法，参数）和模型结构同样重要。作者提出了constraint-aware 并且同时搜索网络结构以及训练方式的 NAS 方法，搜索出的训练方法，在其他模型上也同样使用，甚至可以超过weakly supervised 训练处的模型。同搜索出的stochastic weight averaging via EMA在object detection任务上也可以提升mAP，搜索出的模型结构FBNetV3, 在各种不同计算量的情况下，在Imagenet上面accuracy都高于efficientnet

## 2.DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution

https://github.com/joe-siyuan-qiao/DetectoRS

本文有两点创新：宏观上对检测网络中FPN层进行改进，提出RFP(Recursive Feature Pyramid)；微观层面，引入switchable不同间隔的空洞卷积(SAC, Switchable Atrous Convolution)替代backbone中普通卷积，RFP 将FPN层的feature map融合进backbone相应尺度中，相当于look twice and more，实验效果看，能提升遮挡物体的recall，SAC 将原有卷积变成多个branch不同rate的空洞卷积，同时增加了一个选择模块进行特征融合。权重锁机制可以保证SAC在改变原有网络卷积操作的情况下，仍然能够加载pretrained model，结合上述两点得到的检测器DetectoRS 在COCO数据集上取得多项SOTA结果，54.7 box AP 47.1mask AP 另外DetectoRS with ResNet-50 极大超过了相同backbone的HTC网络（当前SOTA之一）7.7% box AP and 5.9% mask AP

## 3.Generalized Focal Loss:Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

https://arxiv.org/pdf/2006.04388.pdf

https://github.com/implus/GFocal

指出现有检测方法的一些问题：在训练和测试时，quality estimation和classification的做法不统一，localization 的target 是固定的Diracdistribution，但现实中并不那么确定一个物体的位置，为了解决上述问题，提出了Genralized Focal Loss(GFL)，是普通focal loss的continuous的版本，可以在continuous的classification label上训练，GFL包含 Quality Focal Loss (QFL)和Distribution FocalLoss (DFL)分别提升classification和localization，在COCO上达到了SOTA的效果，同样backbone相似速度下AP高于其他方法

## 4.Learning a Unified Sample Weighting Network for Object Detection

https://arxiv.org/abs/2006.06568

https://github.com/caiqi/sample-weighting-network

本文提出了一种新的sample weighting的方法，同时兼顾学习classification和regression目标，给图片中每一个不同的object不同的weight,sample weighting的过程是data-driven的，避免了一些人工调参,这个sample weighting方法可以加在大多数object detector中来提升性能，并且不影响速度

# **Segmentation/Action Recognition/Pose Estimation/Video**


# **Regularization/Distillation and Network Structure**

## 1.Rethinking Pre-training and Self-training

https://arxiv.org/pdf/2006.06882.pdf

文章主要探讨了预训练和自训练的作用,主要任务：目标检测和语义分割,训练方式上分为：是否进行预训练（有监督&无监督），以及是否进行自训练,数据上设计了：不同数据规模，不同数据增强程度,得到了如下结论：当标注数据的规模越大或数据增强越激进时，监督训练的预训练模型的好处会失效甚至有害,作者猜想无监督的预训练方式更general，在大数据/激进的数据增强下或许可以发挥作用，但实验结果是否定的,在预训练模型或从头训的模型基础上，自训练在不同数据规模和数据增强方式下均能带来稳定的性能提升,同时自训练对于不同的网络结构，数据来源都是有效的.

## 2.Disentangled Non-Local Neural Networks

https://arxiv.org/abs/2006.06668

提出了Distangled Non-Local (DNL) block，把原本的non-local block分解成两个部分。DNL中的两个path, 一个model不同位置之间的关系，另外一个model每个位置的saliency,文章中用大量实验结果以及可视化信息来证实了这个设计的合理性,在Segmentation, object detection以及action recognition任务上，在模型中加入DNL，都可以提升模型性能，效果比普通NL更明显

## 3.Are we done with ImageNet

https://arxiv.org/abs/2006.10029

https://github.com/google-research/reassessed-imagenet

图片分类任务的模型一般都会在ImageNet上预训练，ImageNet上面的性能也被看作是一个分类模型性能衡量的标准,一些SOTA模型已经开始overfit ImageNet validation set,本文提出了一个更好更准确的标注方法来重新标注,ImageNet validation set,利用新标注的label，本文重新测试了不同模型在imagenet 上的accuracy, 发现模型之间的提升其实小于在原有validation set的提升,作者提出用本文提供的新的label会让ImageNet更适合来测验识别模型的性能

# **GAN and NAS**

## 1.AutoHAS: Differentiable Hyper-parameter and Architecture Search

https://arxiv.org/abs/2006.03656

指出不同的结构用不同的hyper-paramter会有更好的效果，所以提出同时搜索architecture和hyperparameter，提出了AutoHAS，把原本continuous的hyper-paramterspace用一些categorical basis的linear combination代替在AutoHAS中，所有的architecture 和hyperparameters都用到了weight sharing的方法来提高搜索效率，用MobileNet/ResNet/EfficientNet在ImageNet上面实验，提升了最多2%的accuracy，用BERT在SQuAD上提升了0.4的F1 score，相比其他的AutoML方法，在用非常少（1/10）的computation cost情况下，提升了accuracy.


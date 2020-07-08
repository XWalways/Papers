# **Detection**

## 1.NETNet: Neighbor Erasing and Transferring Network for Better Single Shot Object Detection

https://arxiv.org/pdf/2001.06690.pdf

指出Single-shot FPN detector普遍存在的问题:小的object容易被miss, 大的object的重要部分可能被detect成object,针对这些问题，本文提出了Neighbor Erasing and Transferring (NET) mechanism来处理生成不同的features,NET可以使得不同的feature map体现不同scale的feature,减少其中的scale-confusion,提出了 Neighbor Erasing Module (NEM) and Neighbor Transferring Module (NTM)来分别减少scale confusion和增强feature aggregation,提出了Nearest Neighbor Fusion Module来增强FPN中的feature,在COCO detection问题上,基于SSD baseline,得到了SOTA的speed vs accuracy trade-off.

## 2.Equalization Loss for Long-Tailed Object Recognition

https://arxiv.org/pdf/2003.05176.pdf

https://github.com/tztztztztz/eql.detectron2

为解决长尾多类数据在检测任务里类间样本不均衡的问题,提出了Equalization Loss.在rare class为positive sample时,作为negative sample的frequent class 数量众多,其梯度norm远大于positive的梯度norm,形成类间不均衡竞争,从而影响rare class的学习.Equalization Loss在原cross-entropy loss上乘以weight项,该weight项是样本频率的函数,可以抑制frequent class作为neg sample时的梯度,以Mask RCNN为baseline，添加Equalization Loss于其他loss相比在LVIS OpenImage 等数据集上的AP取得了2~3%显著提升,尤其是rare class类别.另外,本文可以是另一个解决样本不均衡的策略的补充 [DECOUPLING REPRESENTATION AND CLASSIFIER FOR LONG-TAILED RECOGNITION](https://arxiv.org/pdf/1910.09217.pdf).

## 3.Revisiting the Sibling Head in Object Detector

https://arxiv.org/pdf/2003.07540.pdf

本文回顾了目标检测中为解耦分类(cls)和定位框回归(loc)任务的双预测头(sibling head)设计,进一步提出了在原有proposal region上对应cls,loc两个任务分别产生不同的proposal,called taskaware spatial disentanglement (TSD),旨在解决两个任务所利用的特征在空间上分布不一致的问题,本文基于faster rcnn二阶段检测框架,从RoIAlign后加入产生cls proposal和loc proposal的两个FC layer,cls proposal继承deformable RoI pooling的设计,由不规则kxk grid构成,loc proposal在共享的原有proposal上shift空间位置得到,同时保留sibling head 分支联合训练,并加入progressive constraint loss的loss设计,减小TSD和sibling Head两个个分支的差距,得到更高的cls score和更准确的loc,在不同复杂度的backbone上和不同二阶段检测架构的实验显示,上述设计能够实现3-5%mAP(COCO)的提升增加about 10% time cost.

# **Segmentation/Action Recognition/Pose Estimation/Video**

## 1. Conditional Convolutions for Instance Segmentation

https://arxiv.org/pdf/2003.05664.pdf

https://github.com/aim-uofa/adet

提出了CondInst,一个新的instance segmentation方法,相比Mask R-CNN速度更快,精度更高,CondInst是一个Fully Convolutional模型,不需要任何ROI pooling的操作,Mask Head的filters是在inference过程中根据不同的instance而dynamicly生成的,这样大大的节省了计算量,在COCO instance segmentation任务上取得了SOTA.

## 2.PointINS: Point-based Instance Segmentation

https://arxiv.org/pdf/2003.06148.pdf

提出了PointINS,一个基于 点 的instance segmentation方法,提出 instance-aware convolution，其中利用instance-aware Weight Generation模块和Instance agnostic Feature Generation 模块分别处理feature misalignment和feature representation的问题.instance-aware convolution是一个独立于box head的模块,可以轻松的加到大多数one-stage detector上面（例如RetinaNet, FCOS）来做instance segmentation的问题.在COCO instance segementation任务上,在同类point-based方法中,得到了更高的AP.与其他SOTA方法相比,取得了更高的speed-accuracy trade-off.

# **Regularization/Distillation and Network Structure**

## 1.Channel Equilibrium Networks for Learning Deep Representation

https://arxiv.org/pdf/2003.00214.pdf

提出了Channel Equilibrium (CE) block，来使得在同一个convolution layer中的不同channel都可以对学习出来的representation有一些贡献,CE block 和 SE block类似,都可以很轻松的加入到其他CNN模型中,来提升模型的效果,并且几乎不增加计算量.作者从理论和实验的角度证明了CE可以防止inhibited channels的出现,在imagenet classification问题上,把SE加入到resnet, mobilenet中，可以把准确率提升1~2%,比SE更有效,在COCO detection/segmentation问题上,使用CE可以提升1~2%的AP,如果使用cross-gpu synced CE, 可以再提升约1%.

## 2.SlimConv: Reducing Channel Redundancy in Convolutional Neural Networks by Weights Flipping

https://arxiv.org/pdf/2003.07469.pdf

提出了SlimConv Module,可以用来代替普通的convolution操作，在用比较小的参数和计算量的同时,增强representation能力提出里weights flipping操作,可以增强feature的diversity,同时用reconstruct模块降低channel,redundancy,SlimConv可以代替普通Conv放进ResNet/MobileNet中,在ImageNet,COCO上,在FLOPS和Params都减少约30%的情况下,accuracy有大概1%的提升.


## 3.ReZero is All You Need: Fast Convergence at Large Depth

https://arxiv.org/abs/2003.04887

https://github.com/majumderb/rezero

提出了一种神经网络结构改进方法（类似正则化）.通过在神经网络中增加一个可学习的参数Alpha,并在网络初始化时alpha=0,可以使得网络在训练过程中避免梯度消失和梯度爆炸，同时能加快网络收敛速度.文章从网络的输入输出雅可比矩阵入手,研究了梯度反向传播在初始化时如何影响网络参数更新,并通过alpha的优化说明了该参数的有效性,并且说明了使用该参数可以避免使用其他Normalization而引入繁复的计算.在ResNet,Transformer上进行实验,验证其方法能够更快和更好的进行训练.

## 4.Dynamic ReLU

https://arxiv.org/pdf/2003.10027.pdf

提出了Dynamic ReLU, 一个新的Activation function.类似于Parametric ReLU,不同的是postitve和negative两个slope都是根据输入的feature来决定的.增加的计算量很小,同时很大程度上增强了模型的representation capability,指出了Squeeze-Excitation以及类似的方法也是Dynamic ReLU的一种特殊情况,提出了三种Dynamic ReLU, 分别有不同的sharing方式,在ImageNet和COCO数据集上,使用Dynamic ReLU替换MobileNet/ResNet中的普通ReLU,效果有显著提升(1-5%).

## 5. Circle Loss: A Unified Perspective of Pair Similarity Optimization

https://arxiv.org/pdf/2002.10857.pdf

提出了Circle loss,通过让每个相似性得分以不同的步调学习,Circle Loss 赋予深度特征学习的更灵活的优化途径,以及更明确的收敛目标,Circle loss可以用class level,或者pair-wise标注来训练,Triplet Loss和softmax cross-entropy loss都是Circle Loss的特殊形式,在人脸识别,行人再识别,细粒度的图像检索等多种深度特征学习任务上，Circle Loss 都取得了极具竞争力的性能.

## 6.Circumventing Outliers of AutoAugment with Knowledge Distillation

https://arxiv.org/pdf/2003.11342.pdf

指出过多的augmentation可能会去除一些discriminative的信息,这种情况下用原本的groundtruth label并不是最好的方法,本文提出用Knowledge Distillation的方法来解决这个问题,用teacher model的output作为student model的target,在CIFAR-10/100,和ImageNet上,在使用AutoAugment/RandAugment的情况下,加上本文提出的方法,都可以提升accuracy,在ImageNet上,用EfficientNet-B8作为backbone,在没有用额外数据的情况下得到了85.8%的top-1accuracy.


# **GAN and NAS**


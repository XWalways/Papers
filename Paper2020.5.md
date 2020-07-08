# **Detection**

## 1.Stitcher: Feedback-driven Data Provider for Object Detection

https://arxiv.org/pdf/2004.12432.pdf

本文提出了Stitcher，一种根据loss反馈而变化输入的训练方法，用来提升object detection的性能,Stitcher把不同的图片缩小后，拼接成一个大的图片进行训练，可以提升小物体的比例,Stitcher方法通过Loss 反馈来控制不同大小sample之间的平衡,用Stitcher方法代替multi-scale training，可以有效的提升AP，并且训练速度更快.

## 2.Scale-Equalizing Pyramid Convolution for Object Detection

https://arxiv.org/pdf/2005.03101.pdf

https://github.com/jshilong/SEPC

本文提出了Pyramid Convolution (PConv)，一个轻量级的3D Convolution，可以用来融合不同scale的信息,提出了scale-equalizing pyramid convolution(SEPC), 来使得在feature pyramid上的PConv操作与在Gaussian pyramid上提取feature更类似,把本文提出的SEPC结合在其他多个detection方法上(FSAF, FCOS, FreeAnchor, Reppoints,RetinaNet...)，在COCO上面AP都有至少2个点的提升.

## 3.Cheaper Pre-training Lunch: An Efficient Paradigm for Object Detection

https://arxiv.org/pdf/2004.12178.pdf

本文提出了一个高效通用的pre-training方法，只需要目标object detection数据用来做预训练，不需要额外数据(ImageNet),本文提出了Jigsaw方法来做生成预训练数据，并且设计了可以适应Effective Receptive Field的dense classification训练方法。这套训练方法可以提高数据的利用率，并且更适合后续的objectdetection 训练,在COCO detection上，本文提出的方法可以用很少的预训练iteration来达到甚至超过imagenet预训练的效果.

## 4.Scope Head for Accurate Localization in Object Detection

https://arxiv.org/pdf/2005.04854.pdf

本文提出不同位置上的anchor是互相影响依赖的，并且提出用softmax来比较不同anchor之间与ground-truth matching的程度,提出了一个coarse-to-fine的object localization流程，可以在保持anchor-free方法的自由度的同时，提升准确度并且减低多余的输出提出了把categroy-classification score和anchorselection score合并起来代表confidence of detection box,用上述方法提出了ScopeNet, 在COCO detection任务上上使用可以有大概2%的AP提升.

# **Segmentation/Action Recognition/Pose Estimation/Video**


# **Regularization/Distillation and Network Structure**

## 1.Exploring Self-attention for Image Recognition

https://arxiv.org/pdf/2004.13621.pdf

https://github.com/hszhao/SAN

本文探索了两种self-attention.一种是pairwise self-attention, 另外一种是partchwise selfattention,作者指出pairwise self-attention本质上是一个set operation。而partchwise self-attention比普通convolution有更强的modeling power,本文直接用self-attention来做一个模型的buiding block, 而不像其他的网络一般是在convolution后面加self-attention来增强,在ImageNet上，SAN19比ResNet50 准确率高了1.3%，参数和计算量都小了大概20%,Self-attention Networks在大角度旋转，翻转，以及adversarial attack的情况下，都展现出了比普通CNN更强的robustness.

## 2.Supervised Contrastive Learning

https://arxiv.org/pdf/2004.11362.pdf

提出了一种新的Supervised Contrastive Loss，使得每个anchor都可以有多个正样本。这样就可以利用label和constrastive loss做supervised learning,本文提出的方法可以比cross entropy让模型学到更好的representation，提升了模型的accuracy和robustness.Supervised Contrastive Loss对于不同的超参数设置相对于cross-entropy更加稳定,本文指出Supervised Contrastive Loss是triples loss的generalization，每个样本有过个positive/negative。Supervised Contrastive Loss会更注重于学习hard postivie/negative.

## 3.When Ensembling Smaller Models is More Efficient than Single Large Models

https://arxiv.org/pdf/2005.00570.pdf

本文尝试比较了ensemble一个较小模型，与直接使用一个比较大的模型，那种更加高效。几组在CIFAR-10和ImageNet上用ResNet和EfficientNet的实验表明，用ensemble的效果比直接用大模型要好,另外对于ensemble的模型结构，本文的实验结果表明直接选择一个最好的结构训练多个模型便可以达到最好的效果，尝试不同的模型结构并不会带来ensemble准确率的提升.

# **GAN and NAS**

## 1.MobileDets: Searching for Object Detection Architectures for Mobile Accelerators

https://arxiv.org/pdf/2004.14525.pdf

提出以Inverted BottleNecks (IBN)为基础模块的搜索空间，可能在mobile设定下并不能达到最好的效果,针对mobile accelerators，重新分析了普通convolution的效用，并提出了Tensor-Decomposition-Based search space (TDB),在更好的search space的基础上，利用NAS的方法可以搜索到在各种不同mobile accelerator下非常高效的detection模型。

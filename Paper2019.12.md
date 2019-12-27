# **Detection**

## 1.RDSNet: A New Deep Architecture for Reciprocal Object Detection and Instance Segmentation

https://arxiv.org/pdf/1912.05070.pdf

提出了detection和segmentation互补的two-stream模型,做到single-stage和end-to-end,
通过object-level和pixel-level的信息互补得到更高的bounding box和mask精度,
在COCO数据集上,以近3倍的速度取得了和TensorMask相近的精度,
在目标检测任务上的computational cost极低,
可以尝试更换backbone来进一步提升性能.

## 2.Learning from Noisy Anchors for One-stage Object Detection

https://arxiv.org/pdf/1912.05086.pdf

文章提出通常在做anchor和gt匹配的过程中出现的两类问题:
anchor包含了关键局部信息（如上图长颈鹿头部红色框）,但是IoU小于阈值被判为negtive anchor,
anchor虽然与gt有很高的IoU,但是包含太多的背景以及其他object信息（如上图长颈鹿蓝色框）,却被判为positive anchor.
即这样二分是不clear的,包含很多noise,
引入包含预测框和分类信息的soft cleanliness score替换原始0/1二分的anchor label,能够更好得代表这类noisy anchor的分类以及位置置信度,
又对这个score采用re-weighting的策略,平衡hard 和 easy example的影响,以RetinaNet-ResNet101为baseline在coco上提升~2%.

## 3.MnasFPN : Learning Latency-aware Pyramid Architecture for Object Detection on Mobile Devices

https://arxiv.org/pdf/1912.01106.pdf

提出MnasFPN, 主要包括一个mobile-friendly的detection head搜索空间,以及latency-aware NAS来搜索高效的detection模型结构,
搜索出来的detection head architecture要优于SSDLite以及NAS-FPNLite,
MnasFPN比NAS-FPNLite快10%,并且高了1%的AP,
通过ablation studies发现本文用到的搜索空间是用到的NAS Controller的local optimal.

# **Segmentation/Action Recognition/Pose Estimation/Video**

## 1.Learning Efficient Video Representation with Video Shuffle Networks

https://arxiv.org/pdf/1911.11319.pdf

提出了video shuffle操作,可以加在2D Conv中间,使得2D Convnet也能获取到temporal的信息,
video shuffle操作本身不增加任何参数和计算量,同时可以明显提升video recognition性能,
在几个action recognition数据集例如kinetics, Moments in Time, Something-Something上都取得了（近似）SOTA的效果.

## 2.A Multigrid Method for Efficiently Training Video Models

https://arxiv.org/pdf/1912.00998.pdf

提出Multigrid方法来大幅度提升video模型的训练效率,
提出一个根据training过程变化mini-batch 的 spatial-temporal resolutions 的schedule,
generally,先用比较小的resolution训练，然后逐渐把resolution变大,
在各种不同 模型结构(I3D, non-local, slow-fast),几个数据集(Kinetics, Something-Something, Charades),以及不用的training setting (with and without pretraining, 128 GPUs or 1 GPU)都把训练速度加快了3~6倍,并且准确率大多数都有提升.

## 3. SOLO: Segmenting Objects by Locations

https://arxiv.org/pdf/1912.04488.pdf

提出了一个one-stage instance segmentation的方法SOLO,
把instance segmentation分解成两个classification task,
一个branch负责在instance中心点预测category,
另外一个branch负责预测相对应的instance mask,这个branch的channel数和第一个branch的feature map里面cell数量相同,
在同样backbone的情况下,在coco instance segmentation效果优于Mask R-CNN和TensorMask.

## 4.PointRend: Image Segmentation as Rendering

https://arxiv.org/pdf/1912.08193.pdf

把image segmentation当做rendering的问题,利用CG的传统方法来解决undersampling带来的问题,
提出PointRend模块,通过 iterative subdivision算法adaptively选出一些locations,来做point-based segmentation predictions,
PointRend可以很容易的用在semenatic和instance segmentation任务的网络中,
在Cityscapes和COCO上加上PointRend都有至少1%的提升,而且preditected mask更加贴合boundary.


# **Regularization and Network Structure**

## 1.GhostNet: More Features from Cheap Operations

https://arxiv.org/pdf/1911.11907.pdf

https://github.com/iamhankai/ghostnet

提出了轻量级的Ghost Module来代替不同的convolution,
Ghost Module先用一个convolution得到一个channel数较小的feature map,
后面接一个depthwise convolution (with channel multiplier 来控制输出channel 数),
然后和中间feature map concatenate到一起,
用Ghost Bottleneck来搭建一个mobilenetv3, 在imagenet classification上取得了小模型的SOTA,
速度相同的情况下,比mobilenetv3的准确率高0.5%左右.




# **NAS and Others**

# Config.md

# Configuration

[English](Config.md) **|** [简体中文](Config_CN.md)

#### Contents

1. [Experiment Name Convention](#Experiment-Name-Convention)
1. [Configuration Explanation](#Configuration-Explanation)
    1. [Training Configuration](#Training-Configuration)
    1. [Testing Configuration](#Testing-Configuration)

## Experiment Name Convention

Taking `001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb` as an example:

- `001`: We usually use index for managing experiments
- `MSRResNet`: Model name, here is Modified SRResNet
- `x4_f64b16`: Import configuration parameters. It means the upsampling ratio is 4; the channel number of middle features is 64; and it uses 16 residual block
- `DIV2K`: Training data is DIV2K
- `1000k`: Total training iteration is 1000k
- `B16G1`: Batch size is 16; one GPU is used for training
- `wandb`: Use wandb logger; the training process has beed uploaded to wandb server

**Note**: If `debug` is in the experiment name, it will enter the debug mode. That is, the program will log and validate more intensively and will not use `tensorboard logger` and `wandb logger`.

## Configuration Explanation

We use yaml files for configuration.

### Training Configuration

Taking [train_MSRResNet_x4.yml](../options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml) as an example:

```yml
####################################
# The following are general settings
####################################
# Experiment name, more details are in [Experiment Name Convention]. If debug in the experiment name, it will enter debug mode
name: 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb
# Model type. Usually the class name defined in the `models` folder
model_type: SRModel
# The scale of the output over the input. In SR, it is the upsampling ratio. If not defined, use 1
scale: 4
# The number of GPUs for training
num_gpu: 1  # set num_gpu: 0 for cpu mode
# Random seed
manual_seed: 0

########################################################
# The following are the dataset and data loader settings
########################################################
datasets:
  # Training dataset settings
  train:
    # Dataset name
    name: DIV2K
    # Dataset type. Usually the class name defined in the `data` folder
    type: PairedImageDataset
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # GT (Ground-Truth) folder path
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    # LQ (Low-Quality) folder path
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    # template for file name. Usually, LQ files have suffix like `_x4`. It is used for file name mismatching
    filename_tmpl: '{}'
    # IO backend, more details are in [docs/DatasetPreparation.md]
    io_backend:
      # directly read from disk
      type: disk

    # Ground-Truth training patch size
    gt_size: 128
    # Whether to use horizontal flip. Here, flip is for horizontal flip
    use_hflip: true
    # Whether to rotate. Here for rotations with every 90 degree
    use_rot: true

    #### The following are data loader settings
    # Whether to shuffle
    use_shuffle: true
    # Number of workers of reading data for each GPU
    num_worker_per_gpu: 6
    # Total training batch size
    batch_size_per_gpu: 16
    # THe ratio of enlarging dataset. For example, it will repeat 100 times for a dataset with 15 images
    # So that after one epoch, it will read 1500 times. It is used for accelerating data loader
    # since it costs too much time at the start of a new epoch
    dataset_enlarge_ratio: 100

  # validation dataset settings
  val:
    # Dataset name
    name: Set5
    # Dataset type. Usually the class name defined in the `data` folder
    type: PairedImageDataset
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # GT (Ground-Truth) folder path
    dataroot_gt: datasets/Set5/GTmod12
    # LQ (Low-Quality) folder path
    dataroot_lq: datasets/Set5/LRbicx4
    # IO backend, more details are in [docs/DatasetPreparation.md]
    io_backend:
      # directly read from disk
      type: disk

##################################################
# The following are the network structure settings
##################################################
# network g settings
network_g:
  # Architecture type. Usually the class name defined in the `basicsr/archs` folder
  type: MSRResNet
  #### The following arguments are flexible and can be obtained in the corresponding doc
  # Channel number of inputs
  num_in_ch: 3
  # Channel number of outputs
  num_out_ch: 3
  # Channel number of middle features
  num_feat: 64
  # block number
  num_block: 16
  # upsampling ratio
  upscale: 4

#########################################################
# The following are path, pretraining and resume settings
#########################################################
path:
  # Path for pretrained models, usually end with pth
  pretrain_network_g: ~
  # Whether to load pretrained models strictly, that is the corresponding parameter names should be the same
  strict_load_g: true
  # Path for resume state. Usually in the `experiments/exp_name/training_states` folder
  # This argument will over-write the `pretrain_network_g`
  resume_state: ~


#####################################
# The following are training settings
#####################################
train:
  # Optimizer settings
  optim_g:
    # Optimizer type
    type: Adam
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # Learning rate
    lr: !!float 2e-4
    weight_decay: 0
    # beta1 and beta2 for the Adam
    betas: [0.9, 0.99]

  # Learning rate scheduler settings
  scheduler:
    # Scheduler type
    type: CosineAnnealingRestartLR
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # Cosine Annealing periods
    periods: [250000, 250000, 250000, 250000]
    # Cosine Annealing restart weights
    restart_weights: [1, 1, 1, 1]
    # Cosine Annealing minimum learning rate
    eta_min: !!float 1e-7

  # Total iterations for training
  total_iter: 1000000
  # Warm up iterations. -1 indicates no warm up
  warmup_iter: -1

  #### The following are loss settings
  # Pixel-wise loss options
  pixel_opt:
    # Loss type. Usually the class name defined in the `basicsr/models/losses` folder
    type: L1Loss
    # Loss weight
    loss_weight: 1.0
    # Loss reduction mode
    reduction: mean


#######################################
# The following are validation settings
#######################################
val:
  # validation frequency. Validate every 5000 iterations
  val_freq: !!float 5e3
  # Whether to save images during validation
  save_img: false

  # Metrics in validation
  metrics:
    # Metric name. It can be arbitrary
    psnr:
      # Metric type. Usually the function name defined in the`basicsr/metrics` folder
      type: calculate_psnr
      #### The following arguments are flexible and can be obtained in the corresponding doc
      # Whether to crop border during validation
      crop_border: 4
      # Whether to convert to Y(CbCr) for validation
      test_y_channel: false

########################################
# The following are the logging settings
########################################
logger:
  # Logger frequency
  print_freq: 100
  # The frequency for saving checkpoints
  save_checkpoint_freq: !!float 5e3
  # Whether to tensorboard logger
  use_tb_logger: true
  # Whether to use wandb logger. Currently, wandb only sync the tensorboard log. So we should also turn on tensorboard when using wandb
  wandb:
    # wandb project name. Default is None, that is not using wandb.
    # Here, we use the basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # If resuming, wandb id could automatically link previous logs
    resume_id: ~

################################################
# The following are distributed training setting
# Only require for slurm training
################################################
dist_params:
  backend: nccl
  port: 29500
```

### Testing Configuration

Taking [test_MSRResNet_x4.yml](../options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml) as an example:

```yml
# Experiment name
name: 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb
# Model type. Usually the class name defined in the `models` folder
model_type: SRModel
# The scale of the output over the input. In SR, it is the upsampling ratio. If not defined, use 1
scale: 4
# The number of GPUs for testing
num_gpu: 1  # set num_gpu: 0 for cpu mode

########################################################
# The following are the dataset and data loader settings
########################################################
datasets:
  # Testing dataset settings. The first testing dataset
  test_1:
    # Dataset name
    name: Set5
    # Dataset type. Usually the class name defined in the `data` folder
    type: PairedImageDataset
    #### The following arguments are flexible and can be obtained in the corresponding doc
    # GT (Ground-Truth) folder path
    dataroot_gt: datasets/Set5/GTmod12
    # LQ (Low-Quality) folder path
    dataroot_lq: datasets/Set5/LRbicx4
    # IO backend, more details are in [docs/DatasetPreparation.md]
    io_backend:
      # directly read from disk
      type: disk
  # Testing dataset settings. The second testing dataset
  test_2:
    name: Set14
    type: PairedImageDataset
    dataroot_gt: datasets/Set14/GTmod12
    dataroot_lq: datasets/Set14/LRbicx4
    io_backend:
      type: disk
  # Testing dataset settings. The third testing dataset
  test_3:
    name: DIV2K100
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_valid_HR
    dataroot_lq: datasets/DIV2K/DIV2K_valid_LR_bicubic/X4
    filename_tmpl: '{}x4'
    io_backend:
      type: disk

##################################################
# The following are the network structure settings
##################################################
# network g settings
network_g:
  # Architecture type. Usually the class name defined in the `basicsr/archs` folder
  type: MSRResNet
  #### The following arguments are flexible and can be obtained in the corresponding doc
  # Channel number of inputs
  num_in_ch: 3
  # Channel number of outputs
  num_out_ch: 3
  # Channel number of middle features
  num_feat: 64
  # block number
  num_block: 16
  # upsampling ratio
  upscale: 4
  upscale: 4

#################################################
# The following are path and pretraining settings
#################################################
path:
  ## Path for pretrained models, usually end with pth
  pretrain_network_g: experiments/001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb/models/net_g_1000000.pth
  # Whether to load pretrained models strictly, that is the corresponding parameter names should be the same
  strict_load_g: true

##########################################################
# The following are validation settings (Also for testing)
##########################################################
val:
  # Whether to save images during validation
  save_img: true
  # Suffix for saved images. If None, use exp name
  suffix: ~

  # Metrics in validation
  metrics:
    # Metric name. It can be arbitrary
    psnr:
      # Metric type. Usually the function name defined in the`basicsr/metrics` folder
      type: calculate_psnr
      #### The following arguments are flexible and can be obtained in the corresponding doc
      # Whether to crop border during validation
      crop_border: 4
      # Whether to convert to Y(CbCr) for validation
      test_y_channel: false
    # Another metric
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false
```


---

# Config_CN.md

# 配置文件

[English](Config.md) **|** [简体中文](Config_CN.md)

#### 目录

1. [实验命名](#实验命名)
1. [配置文件说明](#配置文件说明)
    1. [训练配置文件](#训练配置文件)
    1. [测试配置文件](#测试配置文件)

## 实验命名

以`001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb`为例:

- `001`: 我们一般给实验标号, 方便实验管理
- `MSRResNet`: 模型名称, 这里是Modified SRResNet
- `x4_f64b16`: 重要配置参数, 这里表示放大4倍; 中间feature通道数是64, 使用了16个Residual Block
- `DIV2K`: 训练数据集是DIV2K
- `1000k`: 训练了1000k iterations
- `B16G1`: Batch size为16, 使用一个GPU训练
- `wandb`: 使用了wandb, 训练过程上传到了wandb云服务器

**注意**: 如果在实验名字中有`debug`字样, 则会进入debug模式, 即程序会更密集地log和validate, 并且不会使用`tensorboard logger`和`wandb logger`.

## 配置文件说明

我们使用了 yaml 格式来做配置文件.

### 训练配置文件

我们以[train_MSRResNet_x4.yml](../options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml)为例, 说明训练配置文件的含义:

```yml
#################
# 以下为通用的设置
#################
# 实验名称, 具体可参见 [实验名称命名], 若实验名字中有debug字样, 则会进入debug模式
name: 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb
# 使用的model类型, 一般为在`models`目录下定义的模型的类名
model_type: SRModel
# 输出相比输入的放大比率, 在SR中是放大倍数; 若有些任务没有这个配置, 则写1
scale: 4
# 训练卡数
num_gpu: 1  # set num_gpu: 0 for cpu mode
# 随机种子设定
manual_seed: 0

#################################
# 以下为dataset和data loader的设置
#################################
datasets:
  # 训练数据集的设置
  train:
    # 数据集的名称
    name: DIV2K
    # 数据集的类型, 一般为在`data`目录下定义的dataset的类名
    type: PairedImageDataset
    #### 以下属性是灵活的, 可以在相应类的说明文档中获得; 若新加数据集, 则可以根据需要添加
    # GT (Ground-Truth) 图像的文件夹路径
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    # LQ (Low-Quality) 图像的文件夹路径
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    # 文件名字模板, 一般LQ文件会有类似`_x4`这样的文件后缀, 这个就是来处理GT和LQ文件后缀不匹配的问题的
    filename_tmpl: '{}'
    # IO 读取的backend, 详细可以参见 [docs/DatasetPreparation_CN.md]
    io_backend:
      # disk 表示直接从硬盘读取
      type: disk

    # 训练中Ground-Truth的Training patch的大小
    gt_size: 128
    # 是否使用horizontal flip, 这里的flip特指 horizontal flip
    use_hflip: true
    # 是否使用rotation, 这里指的是每隔90°旋转
    use_rot: true

    #### 下面是data loader的设置
    # data loader是否使用shuffle
    use_shuffle: true
    # 每一个GPU的data loader读取进程数目
    num_worker_per_gpu: 6
    # 总共的训练batch size
    batch_size_per_gpu: 16
    # 扩大dataset的倍率. 比如数据集有15张图, 则会重复这些图片100次, 这样一个epoch下来, 能够读取1500张图
    # (事实上是重复读的). 它经常用来加速data loader, 因为在有的机器上, 一个epoch结束, 会重启进程, 往往会很慢
    dataset_enlarge_ratio: 100

  # validation 数据集的设置
  val:
    # 数据集名称
    name: Set5
    # 数据集的类型, 一般为在`data`目录下定义的dataset的类名
    type: PairedImageDataset
    #### 以下属性是灵活的, 可以在相应类的说明文档中获得; 若新加数据集, 则可以根据需要添加
    # GT (Ground-Truth) 图像的文件夹路径
    dataroot_gt: datasets/Set5/GTmod12
    # LQ (Low-Quality) 图像的文件夹路径
    dataroot_lq: datasets/Set5/LRbicx4
    # IO 读取的backend, 详细可以参见 [docs/DatasetPreparation_CN.md]
    io_backend:
      # disk 表示直接从硬盘读取
      type: disk

#####################
# 以下为网络结构的设置
#####################
# 网络g的设置
network_g:
  # 网络结构 (Architecture)的类型, 一般为在`basicsr/archs`目录下定义的dataset的类名
  type: MSRResNet
  #### 以下属性是灵活的, 可以在相应类的说明文档中获得
  # 输入通道数目
  num_in_ch: 3
  # 输出通道数目
  num_out_ch: 3
  # 中间特征通道数目
  num_feat: 64
  # 使用block的数目
  num_block: 16
  # SR的放大倍数
  upscale: 4

######################################
# 以下为路径和与训练模型、重启训练的设置
######################################
path:
  # 预训练模型的路径, 需要以pth结尾的模型
  pretrain_network_g: ~
  # 加载预训练模型的时候, 是否需要网络参数的名称严格对应
  strict_load_g: true
  # 重启训练的状态路径, 一般在`experiments/exp_name/training_states`目录下
  # 这个设置了, 会覆盖  pretrain_network_g 的设定
  resume_state: ~


#################
# 以下为训练的设置
#################
train:
  # 优化器设置
  optim_g:
    # 优化器类型
    type: Adam
    ##### 以下属性是灵活的, 根据不同优化器有不同的设置
    # 学习率
    lr: !!float 2e-4
    weight_decay: 0
    # Adam优化器的 beta1 和 beta2
    betas: [0.9, 0.99]

  # 学习率的设定
  scheduler:
    # 学习率Scheduler的类型
    type: CosineAnnealingRestartLR
    #### 以下属性是灵活的, 根据学习率Scheduler有不同的设置
    # Cosine Annealing的周期
    periods: [250000, 250000, 250000, 250000]
    # Cosine Annealing每次Restart的权重
    restart_weights: [1, 1, 1, 1]
    # Cosine Annealing的学习率最小值
    eta_min: !!float 1e-7

  # 总共的训练迭代次数
  total_iter: 1000000
  # warm up的iteration数目, 如是-1, 表示没有warm up
  warmup_iter: -1  # no warm up

  #### 以下是loss的设置
  # pixel-wise loss的options
  pixel_opt:
    # loss类型, 一般为在`basicsr/models/losses`目录下定义的loss类名
    type: L1Loss
    # loss 权重
    loss_weight: 1.0
    # loss reduction方式
    reduction: mean


#######################
# 以下为Validation的设置
#######################
val:
  # validation的频率, 每隔 5000 iterations 做一次validation
  val_freq: !!float 5e3
  # 是否需要在validation的时候保存图片
  save_img: false

  # Validation时候使用的metric
  metrics:
    # metric的名字, 这个名字可以是任意的
    psnr:
      # metric的类型, 一般为在`basicsr/metrics`目录下定义的metric函数名
      type: calculate_psnr
      #### 以下属性是灵活的, 根据metric有不同的设置
      # 计算metric时, 是否需要crop border
      crop_border: 4
      # 是否转成在Y(CbCr)空间上计算metric
      test_y_channel: false

####################
# 以下为Logging的设置
####################
logger:
  # 屏幕上打印的logger频率
  print_freq: 100
  # 保存checkpoint的频率
  save_checkpoint_freq: !!float 5e3
  # 是否使用tensorboard logger
  use_tb_logger: true
  # 是否使用wandb logger, 目前wandb只是同步tensorboard的内容, 因此要使用wandb, 必须也同时使用tensorboard
  wandb:
    # wandb的project. 默认是 None, 即不使用wandb.
    # 这里使用了 basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # 如果是resume, 可以输入上次的wandb id, 则log可以接起来
    resume_id: ~

#############################################################
# 以下为distributed training的设置, 目前只有在Slurm训练下才需要
#############################################################
dist_params:
  backend: nccl
  port: 29500
```

### 测试配置文件

我们以[test_MSRResNet_x4.yml](../options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml)为例, 说明测试配置文件的含义:

```yml
# 实验名称
name: 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb
# 使用的model类型, 一般为在`models`目录下定义的模型的类名
model_type: SRModel
# 输出相比输入的放大比率, 在SR中是放大倍数; 若有些任务没有这个配置, 则写1
scale: 4
# 测试卡数
num_gpu: 1  # set num_gpu: 0 for cpu mode

#################################
# 以下为dataset和data loader的设置
#################################
datasets:
  # 测试数据集的设置, 后缀1表示第一个测试集
  test_1:
    # 数据集的名称
    name: Set5
    # 数据集的类型, 一般为在`data`目录下定义的dataset的类名
    type: PairedImageDataset
    #### 以下属性是灵活的, 可以在相应类的说明文档中获得; 若新加数据集, 则可以根据需要添加
    # GT (Ground-Truth) 图像的文件夹路径
    dataroot_gt: datasets/Set5/GTmod12
    # LQ (Low-Quality) 图像的文件夹路径
    dataroot_lq: datasets/Set5/LRbicx4
    # IO 读取的backend, 详细可以参见 [docs/DatasetPreparation_CN.md]
    io_backend:
      # disk 表示直接从硬盘读取
      type: disk
  # 测试数据集的设置, 后缀2表示第二个测试集
  test_2:
    name: Set14
    type: PairedImageDataset
    dataroot_gt: datasets/Set14/GTmod12
    dataroot_lq: datasets/Set14/LRbicx4
    io_backend:
      type: disk
  # 测试数据集的设置, 后缀3表示第三个测试集
  test_3:
    name: DIV2K100
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_valid_HR
    dataroot_lq: datasets/DIV2K/DIV2K_valid_LR_bicubic/X4
    filename_tmpl: '{}x4'
    io_backend:
      type: disk

#####################
# 以下为网络结构的设置
#####################
# 网络g的设置
network_g:
  # 网络结构 (Architecture)的类型, 一般为在`basicsr/archs`目录下定义的dataset的类名
  type: MSRResNet
  #### 以下属性是灵活的, 可以在相应类的说明文档中获得
  # 输入通道数目
  num_in_ch: 3
  # 输出通道数目
  num_out_ch: 3
  # 中间特征通道数目
  num_feat: 64
  # 使用block的数目
  num_block: 16
  # SR的放大倍数
  upscale: 4

#############################
# 以下为路径和与训练模型的设置
#############################
path:
  # 预训练模型的路径, 需要以pth结尾的模型
  pretrain_network_g: experiments/001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb/models/net_g_1000000.pth
  # 加载预训练模型的时候, 是否需要网络参数的名称严格对应
  strict_load_g: true

##################################
# 以下为Validation (也是测试)的设置
##################################
val:
  # 是否需要在测试的时候保存图片
  save_img: true
  # 对保存的图片添加后缀，如果是None, 则使用exp name
  suffix: ~

  # 测试时候使用的metric
  metrics:
    # metric的名字, 这个名字可以是任意的
    psnr:
      # metric的类型, 一般为在`basicsr/metrics`目录下定义的metric函数名
      type: calculate_psnr
      #### 以下属性是灵活的, 根据metric有不同的设置
      # 计算metric时, 是否需要crop border
      crop_border: 4
      # 是否转成在Y(CbCr)空间上计算metric
      test_y_channel: false
    # 另外一个metric
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: false
```


---

# DatasetPreparation.md

# Dataset Preparation

[English](DatasetPreparation.md) **|** [简体中文](DatasetPreparation_CN.md)

#### Contents

1. [Data Storage Format](#Data-Storage-Format)
    1. [How to Use](#How-to-Use)
    1. [How to Implement](#How-to-Implement)
    1. [LMDB Description](#LMDB-Description)
    1. [Data Pre-fetcher](#Data-Pre-fetcher)
1. [Image Super-Resolution](#Image-Super-Resolution)
    1. [DIV2K](#DIV2K)
    1. [Common Image SR Datasets](#Common-Image-SR-Datasets)
1. [Video Super-Resolution](#Video-Super-Resolution)
    1. [REDS](#REDS)
    1. [Vimeo90K](#Vimeo90K)
1. [StylgeGAN2](#StyleGAN2)
    1. [FFHQ](#FFHQ)

## Data Storage Format

At present, there are three types of data storage formats supported:

1. Store in `hard disk` directly in the format of images / video frames.
1. Make [LMDB](https://lmdb.readthedocs.io/en/release/), which could accelerate the IO and decompression speed during training.
1. [memcached](https://memcached.org/) is also supported, if they are installed (usually on clusters).

#### How to Use

At present, we can modify the configuration yaml file to support different data storage formats. Taking [PairedImageDataset](../basicsr/data/paired_image_dataset.py) as an example, we can modify the yaml file according to different requirements.

1. Directly read disk data.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    io_backend:
      type: disk
    ```

1. Use LMDB.
We need to make LMDB before using it. Please refer to [LMDB description](#LMDB-Description). Note that we add meta information to the original LMDB, and the specific binary contents are also different. Therefore, LMDB from other sources can not be used directly.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub.lmdb
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb
    io_backend:
      type: lmdb
    ```

1. Use Memcached
Your machine/clusters mush support memcached before using it. The configuration file should be modified accordingly.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K_train_HR_sub
    dataroot_lq: datasets/DIV2K_train_LR_bicubicX4_sub
    io_backend:
      type: memcached
      server_list_cfg: /mnt/lustre/share/memcached_client/server_list.conf
      client_cfg: /mnt/lustre/share/memcached_client/client.conf
      sys_path: /mnt/lustre/share/pymc/py3
    ```

#### How to Implement

The implementation is to call the elegant fileclient design in [mmcv](https://github.com/open-mmlab/mmcv). In order to be compatible with BasicSR, we have made some changes to the interface (mainly to adapt to LMDB). See [file_client.py](../basicsr/utils/file_client.py) for details.

When we implement our own dataloader, we can easily call the interfaces to support different data storage forms. Please refer to [PairedImageDataset](../basicsr/data/paired_image_dataset.py) for more details.

#### LMDB Description

During training, we use LMDB to speed up the IO and CPU decompression. (During testing, usually the data is limited and it is generally not necessary to use LMDB). The acceleration depends on the configurations of the machine, and the following factors will affect the speed:

1. Some machines will clean cache regularly, and LMDB depends on the cache mechanism. Therefore, if the data fails to be cached, you need to check it. After the command `free -h`, the cache occupied by LMDB will be recorded under the `buff/cache` entry.
1. Whether the memory of the machine is large enough to put the whole LMDB data in. If not, it will affect the speed due to the need to constantly update the cache.
1. If you cache the LMDB dataset for the first time, it may affect the training speed. So before training, you can enter the LMDB dataset directory and cache the data by: ` cat data.mdb > /dev/nul`.

In addition to the standard LMDB file (data.mdb and lock.mdb), we also add `meta_info.txt` to record additional information.
Here is an example:

**Folder Structure**

```txt
DIV2K_train_HR_sub.lmdb
├── data.mdb
├── lock.mdb
├── meta_info.txt
```

**meta information**

`meta_info.txt`, We use txt file to record for readability. The contents are:

```txt
0001_s001.png (480,480,3) 1
0001_s002.png (480,480,3) 1
0001_s003.png (480,480,3) 1
0001_s004.png (480,480,3) 1
...
```

Each line records an image with three fields, which indicate:

- Image name (with suffix): 0001_s001.png
- Image size: (480, 480,3) represents a 480x480x3 image
- Other parameters (BasicSR uses cv2 compression level for PNG): In restoration tasks, we usually use PNG format, so `1` represents the PNG compression level `CV_IMWRITE_PNG_COMPRESSION` is 1. It can be an integer in [0, 9]. A larger value indicates stronger compression, that is, smaller storage space and longer compression time.

**Binary Content**

For convenience, the binary content stored in LMDB dataset is encoded image by cv2: `cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]`. You can control the compression level by `compress_level`, balancing storage space and the speed of reading (including decompression).

**How to Make LMDB**
We provide a script to make LMDB. Before running the script, we need to modify the corresponding parameters accordingly. At present, we support DIV2K, REDS and Vimeo90K datasets; other datasets can also be made in a similar way.<br>
 `python scripts/data_preparation/create_lmdb.py`

#### Data Pre-fetcher

Apar from using LMDB for speed up, we could use data per-fetcher. Please refer to [prefetch_dataloader](../basicsr/data/prefetch_dataloader.py) for implementation.<br>
It can be achieved by setting `prefetch_mode` in the configuration file. Currently, it provided three modes:

1. None. It does not use data pre-fetcher by default. If you have already use LMDB or the IO is OK, you can set it to None.

    ```yml
    prefetch_mode: ~
    ```

1. `prefetch_mode: cuda`. Use CUDA prefetcher. Please see [NVIDIA/apex](https://github.com/NVIDIA/apex/issues/304#) for more details. It will occupy more GPU memory. Note that in the mode. you must also set `pin_memory=True`.

    ```yml
    prefetch_mode: cuda
    pin_memory: true
    ```

1. `prefetch_mode: cpu`. Use CPU prefetcher, please see [IgorSusmelj/pytorch-styleguide](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#) for more details. (In my tests, this mode does not accelerate)

    ```yml
    prefetch_mode: cpu
    num_prefetch_queue: 1  # 1 by default
    ```

## Image Super-Resolution

It is recommended to symlink the dataset root to `datasets` with the command `ln -s xxx yyy`. If your folder structure is different, you may need to change the corresponding paths in config files.

### DIV2K

[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) is a widely-used dataset in image super-resolution. In many research works, a MATLAB bicubic downsampling kernel is assumed. It may not be practical because the MATLAB bicubic downsampling kernel is not a good approximation for the implicit degradation kernels in real-world scenarios. And there is another topic named *blind restoration* that deals with this gap.

**Preparation Steps**

1. Download the datasets from the [official DIV2K website](https://data.vision.ee.ethz.ch/cvl/DIV2K/).<br>
1. Crop to sub-images: DIV2K has 2K resolution (e.g., 2048 × 1080) images but the training patches are usually small (e.g., 128x128 or 192x192). So there is a waste if reading the whole image but only using a very small part of it. In order to accelerate the IO speed during training, we crop the 2K resolution images to sub-images (here, we crop to 480x480 sub-images). <br>
Note that the size of sub-images is different from the training patch size (`gt_size`) defined in the config file. Specifically, the cropped sub-images with 480x480 are stored. The dataloader will further randomly crop the sub-images to `GT_size x GT_size` patches for training. <br/>
    Run the script [extract_subimages.py](../scripts/data_preparation/extract_subimages.py):

    ```python
    python scripts/data_preparation/extract_subimages.py
    ```

    Remember to modify the paths and configurations if you have different settings.
1. [Optional] Create LMDB files. Please refer to [LMDB Description](#LMDB-Description). `python scripts/data_preparation/create_lmdb.py`. Use the `create_lmdb_for_div2k` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_paired_image_dataset.py`.
Remember to modify the paths and configurations accordingly.
1. [Optional] If you want to use meta_info_file, you may need to run `python scripts/data_preparation/generate_meta_info.py` to generate the meta_info_file.

### Common Image SR Datasets

We provide a list of common image super-resolution datasets.

<table>
  <tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Download</th>
  </tr>
  <tr>
    <td rowspan="3">Classical SR Training</td>
    <td>T91</td>
    <td><sub>91 images for training</sub></td>
    <td rowspan="9"><a href="https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg">Baidu Drive</a></td>
  </tr>
 <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS200</a></td>
    <td><sub>A subset (train) of BSD500 for training</sub></td>
  </tr>
  <tr>
    <td><a href="http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html">General100</a></td>
    <td><sub>100 images for training</sub></td>
  </tr>
  <tr>
    <td rowspan="6">Classical SR Testing</td>
    <td>Set5</td>
    <td><sub>Set5 test dataset</sub></td>
  </tr>
  <tr>
    <td>Set14</td>
    <td><sub>Set14 test dataset</sub></td>
  </tr>
  <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS100</a></td>
    <td><sub>A subset (test) of BSD500 for testing</sub></td>
  </tr>
  <tr>
    <td><a href="https://sites.google.com/site/jbhuang0604/publications/struct_sr">urban100</a></td>
    <td><sub>100 building images for testing (regular structures)</sub></td>
  </tr>
  <tr>
    <td><a href="http://www.manga109.org/en/">manga109</a></td>
    <td><sub>109 images of Japanese manga for testing</sub></td>
  </tr>
  <tr>
    <td>historical</td>
    <td><sub>10 gray low-resolution images without the ground-truth</sub></td>
  </tr>

  <tr>
    <td rowspan="3">2K Resolution</td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">DIV2K</a></td>
    <td><sub>proposed in <a href="http://www.vision.ee.ethz.ch/ntire17/">NTIRE17</a> (800 train and 100 validation)</sub></td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">official website</a></td>
  </tr>
 <tr>
    <td><a href="https://github.com/LimBee/NTIRE2017">Flickr2K</a></td>
    <td><sub>2650 2K images from Flickr for training</sub></td>
    <td><a href="https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar">official website</a></td>
  </tr>
 <tr>
    <td>DF2K</td>
    <td><sub>A merged training dataset of DIV2K and Flickr2K</sub></td>
    <td>-</a></td>
  </tr>

  <tr>
    <td rowspan="2">OST (Outdoor Scenes)</td>
    <td>OST Training</td>
    <td><sub>7 categories images with rich textures</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/u/1/folders/1iZfzAxAwOpeutz27HC56_y5RNqnsPPKr">Google Drive</a> / <a href="https://pan.baidu.com/s/1neUq5tZ4yTnOEAntZpK_rQ#list/path=%2Fpublic%2FSFTGAN&parentPath=%2Fpublic">Baidu Drive</a></td>
  </tr>
 <tr>
    <td>OST300</td>
    <td><sub>300 test images of outdoor scenes</sub></td>
  </tr>

  <tr>
    <td >PIRM</td>
    <td>PIRM</td>
    <td><sub>PIRM self-val, val, test datasets</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/folders/17FmdXu5t8wlKwt8extb_nQAdjxUOrb1O?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1gYv4tSJk_RVCbCq4B6UxNQ">Baidu Drive</a></td>
  </tr>
</table>

## Video Super-Resolution

It is recommended to symlink the dataset root to `datasets` with the command `ln -s xxx yyy`. If your folder structure is different, you may need to change the corresponding paths in config files.

### REDS

[Official website](https://seungjunnah.github.io/Datasets/reds.html).<br>
We regroup the training and validation dataset into one folder. The original training dataset has 240 clips from 000 to 239. And we  rename the validation clips from 240 to 269.

**Validation Partition**

The official validation partition and that used in EDVR for competition are different:

| name | clips | total number |
|:----------:|:----------:|:----------:|
| REDSOfficial | [240, 269] | 30 clips |
| REDS4 | 000, 011, 015, 020 clips from the *original training set* | 4 clips |

All the left clips are used for training. Note that it it not required to explicitly separate the training and validation datasets; and the dataloader does that.

**Preparation Steps**

1. Download the datasets from the [official website](https://seungjunnah.github.io/Datasets/reds.html).
1. Regroup the training and validation datasets: `python scripts/data_preparation/regroup_reds_dataset.py`
1. [Optional] Make LMDB files when necessary. Please refer to [LMDB Description](#LMDB-Description). `python scripts/data_preparation/create_lmdb.py`. Use the `create_lmdb_for_reds` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_reds_dataset.py`.
Remember to modify the paths and configurations accordingly.

### Vimeo90K

[Official webpage](http://toflow.csail.mit.edu/)

1. Download the dataset: [`Septuplets dataset --> The original training + test set (82GB)`](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip).This is the Ground-Truth (GT). There is a `sep_trainlist.txt` file listing the training samples in the download zip file.
1. Generate the low-resolution images (TODO)
The low-resolution images in the Vimeo90K test dataset are generated with the MATLAB bicubic downsampling kernel. Use the script `data_scripts/generate_LR_Vimeo90K.m` (run in MATLAB) to generate the low-resolution images.
1. [Optional] Make LMDB files when necessary. Please refer to [LMDB Description](#LMDB-Description). `python scripts/data_preparation/create_lmdb.py`. Use the `create_lmdb_for_vimeo90k` function and remember to modify the paths and configurations accordingly.
1. Test the dataloader with the script `tests/test_vimeo90k_dataset.py`.
Remember to modify the paths and configurations accordingly.

## StyleGAN2

### FFHQ

Training dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset).

1. Download FFHQ dataset. Recommend to download the tfrecords files from [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).
1. Extract tfrecords to images or LMDBs. (TensorFlow is required to read tfrecords). For each resolution, we will create images folder or LMDB files separately.

    ```bash
    python scripts/data_preparation/extract_images_from_tfrecords.py
    ```


---

# DatasetPreparation_CN.md

# 数据准备

[English](DatasetPreparation.md) **|** [简体中文](DatasetPreparation_CN.md)

#### 目录

1. [数据存储形式](#数据存储形式)
    1. [如何使用](#如何使用)
    1. [如何实现](#如何实现)
    1. [LMDB具体说明](#LMDB具体说明)
    1. [预读取数据](#预读取数据)
1. [图像数据](#图像数据)
    1. [DIV2K](#DIV2K)
    1. [其他常见图像超分数据集](#其他常见图像超分数据集)
1. [视频帧数据](#视频帧数据)
    1. [REDS](#REDS)
    1. [Vimeo90K](#Vimeo90K)
1. [StylgeGAN2](#StyleGAN2)
    1. [FFHQ](#FFHQ)

## 数据存储形式

目前支持的数据存储形式有以下三种:

1. 直接以图像/视频帧的格式存放在硬盘
2. 制作成 [LMDB](https://lmdb.readthedocs.io/en/release/). 训练数据使用这种形式, 一般会加快读取速度.
3. 若是支持 [Memcached](https://memcached.org/), 则可以使用. 它们一般应用在集群上.

#### 如何使用

目前, 我们可以通过 configuration yaml 文件方便的修改. 以支持DIV2K的 [PairedImageDataset](../basicsr/data/paired_image_dataset.py) 为例, 根据不同的要求修改yaml文件:

1. 直接读取硬盘数据

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub
    io_backend:
      type: disk
    ```

1. 使用LMDB.
在使用前需要先制作LMDB, 参见 [LMDB具体说明](#LMDB具体说明), 注意我们在原有的 LMDB 上, 新增加了 meta 信息, 而且具体保存二进制内容也不同, 因此其他来源的LMDB并不能直接拿过来使用.

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K/DIV2K_train_HR_sub.lmdb
    dataroot_lq: datasets/DIV2K/DIV2K_train_LR_bicubic_X4_sub.lmdb
    io_backend:
      type: lmdb
    ```

1. 使用Memcached
机器/集群需要支持 Memcached. 具体的配置文件根据实际的 Memcached 需要进行修改:

    ```yaml
    type: PairedImageDataset
    dataroot_gt: datasets/DIV2K_train_HR_sub
    dataroot_lq: datasets/DIV2K_train_LR_bicubicX4_sub
    io_backend:
      type: memcached
      server_list_cfg: /mnt/lustre/share/memcached_client/server_list.conf
      client_cfg: /mnt/lustre/share/memcached_client/client.conf
      sys_path: /mnt/lustre/share/pymc/py3
    ```

#### 如何实现

实现是调用了[MMCV](https://github.com/open-mmlab/mmcv) 优雅的 FileClient 设计. 为了兼容 BasicSR, 我们对接口做了一些改动 (主要是为了适应LMDB), 参见 [file_client.py](../basicsr/utils/file_client.py).

在实现我们自己的 dataloader 的时候, 可以方便地调用接口, 以实现对不同数据存储形式的支持, 具体可以参考 [PairedImageDataset](../basicsr/data/paired_image_dataset.py) 的写法.

#### LMDB具体说明

我们在训练的时候使用 LMDB 存储形式可以加快IO和CPU解压缩的速度 (测试的时候数据较少, 一般就没有太必要使用 LMDB). 其具体的加速要根据机器的配置来, 以下几个因素会影响:

1. 有的机器设置了定时清理缓存, 而 LMDB 依赖于缓存. 因此若一直缓存不进去, 则需要检查一下. 一般 `free -h` 命令下, LMDB 占用的缓存会记录在 `buff/cache` 条目下面
1. 机器的内存是否足够大, 能够把整个 LMDB 数据都放进去. 如果不是, 则它由于需要不断更换缓存, 会影响速度
1. 若是第一次缓存 LMDB 数据集, 可能会影响训练速度. 可以在训练前, 进入 LMDB 数据集目录, 把数据先缓存进去: `cat data.mdb > /dev/nul`

除了标准的 LMDB 文件 (data.mdb 和 lock.mdb) 外, 我们还增加了 `meta_info.txt` 来记录额外的信息.
下面用一个例子来说明:

**文件结构**

```txt
DIV2K_train_HR_sub.lmdb
├── data.mdb
├── lock.mdb
├── meta_info.txt
```

**meta信息**

`meta_info.txt`, 我们采用txt来记录, 是为了可读性. 其里面的内容为:

```txt
0001_s001.png (480,480,3) 1
0001_s002.png (480,480,3) 1
0001_s003.png (480,480,3) 1
0001_s004.png (480,480,3) 1
...
```

每一行记录了一张图片, 有三个字段, 分别表示:

- 图像名称 (带后缀): 0001_s001.png
- 图像大小: (480,480,3) 表示是480x480x3的图像
- 其他参数 (BasicSR里面使用了 cv2 压缩 png 程度): 因为在复原任务中, 我们通常使用 png 来存储, 所以这个 1 表示 png 的压缩程度 `CV_IMWRITE_PNG_COMPRESSION` 是 1. 它可以取值为[0, 9]的整数, 更大的值表示更强的压缩, 即更小的储存空间和更长的压缩时间.

**二进制内容**

为了方便, 我们在 LMDB 数据集中存储的二进制内容是 cv2 encode过的 image: `cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, compress_level]`. 可以通过 `compress_level` 控制压缩程度, 平衡存储空间和读取(包括解压缩)的速度.

**如何制作**

我们提供了脚本来制作. 在运行脚本前, 需要根据需求修改相应的参数. 目前支持 DIV2K, REDS 和 Vimeo90K 数据集; 其他数据集可仿照进行制作. <br>
 `python scripts/data_preparation/create_lmdb.py`

#### 预读取数据

除了使用LMDB来加速外, 还可以采用预读取数据来加速, 实现参见 [prefetch_dataloader](../basicsr/data/prefetch_dataloader.py).<br>
这个可以通过配置文件中的 `prefetch_mode` 来指定. 目前提供了三种模式:

1. None. 默认不使用. 如果使用了 LMDB 或者 IO 不成问题, 则可不使用

    ```yml
    prefetch_mode: ~
    ```

1. `prefetch_mode: cuda`. 使用 CUDA prefetcher, 具体介绍参见 [NVIDIA/apex](https://github.com/NVIDIA/apex/issues/304#). 它会多占用一些GPU显存. 注意: 这个模式下, 一定要设置 `pin_memory=True`

    ```yml
    prefetch_mode: cuda
    pin_memory: true
    ```

1. `prefetch_mode: cpu`. 使用 CPU prefetcher, 具体介绍参见 [IgorSusmelj/pytorch-styleguide](https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#). (目前测试，这个加速不明显)

    ```yml
    prefetch_mode: cpu
    num_prefetch_queue: 1  # 1 by default
    ```

## 图像数据

推荐把数据通过 `ln -s xxx yyy` 软链到`BasicSR/datasets`下. 如果你的文件结构不同, 需要相应地修改configuration yaml文件的路径.

### DIV2K

DIV2K 数据集被广泛使用在图像复原的任务中.

**数据准备步骤**

1. 从[官网](https://data.vision.ee.ethz.ch/cvl/DIV2K)下载数据.
1. Crop to sub-images: 因为 DIV2K 数据集是 2K 分辨率的 (比如: 2048x1080), 而我们在训练的时候往往并不要那么大 (常见的是 128x128 或者 192x192 的训练patch). 因此我们可以先把2K的图片裁剪成有overlap的 480x480 的子图像块. 然后再由 dataloader 从这个 480x480 的子图像块中随机crop出 128x128 或者 192x192 的训练patch.<br>
    运行脚本 [extract_subimages.py](../scripts/data_preparation/extract_subimages.py):

    ```python
    python scripts/data_preparation/extract_subimages.py
    ```

    使用之前可能需要修改文件里面的路径和配置参数.
    **注意**: sub-image 的尺寸和训练patch的尺寸 (`gt_size`) 是不同的. 我们先把2K分辨率的图像 crop 成 sub-images (往往是 480x480), 然后存储起来. 在训练的时候, dataloader会读取这些sub-images, 然后进一步随机裁剪成 `gt_size` x `gt_size`的大小.
1. [可选] 若需要使用 LMDB, 则需要制作 LMDB, 参考 [LMDB具体说明](#LMDB具体说明).  `python scripts/data_preparation/create_lmdb.py`, 注意选择`create_lmdb_for_div2k`函数, 并需要修改函数相应的配置和路径.
1. 测试: `tests/test_paired_image_dataset.py`, 注意修改函数相应的配置和路径.
1. [可选] 若需要使用 meta_info_file, 运行 `python scripts/data_preparation/generate_meta_info.py` 来生成 meta_info_file.

### 其他常见图像超分数据集

我们提供了常见图像超分数据集的列表.

<table>
  <tr>
    <th>Name</th>
    <th>Datasets</th>
    <th>Short Description</th>
    <th>Download</th>
  </tr>
  <tr>
    <td rowspan="3">Classical SR Training</td>
    <td>T91</td>
    <td><sub>91 images for training</sub></td>
    <td rowspan="9"><a href="https://drive.google.com/drive/folders/1gt5eT293esqY0yr1Anbm36EdnxWW_5oH?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1q_1ERCMqALH0xFwjLM0pTg">Baidu Drive</a></td>
  </tr>
 <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS200</a></td>
    <td><sub>A subset (train) of BSD500 for training</sub></td>
  </tr>
  <tr>
    <td><a href="http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html">General100</a></td>
    <td><sub>100 images for training</sub></td>
  </tr>
  <tr>
    <td rowspan="6">Classical SR Testing</td>
    <td>Set5</td>
    <td><sub>Set5 test dataset</sub></td>
  </tr>
  <tr>
    <td>Set14</td>
    <td><sub>Set14 test dataset</sub></td>
  </tr>
  <tr>
    <td><a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/">BSDS100</a></td>
    <td><sub>A subset (test) of BSD500 for testing</sub></td>
  </tr>
  <tr>
    <td><a href="https://sites.google.com/site/jbhuang0604/publications/struct_sr">urban100</a></td>
    <td><sub>100 building images for testing (regular structures)</sub></td>
  </tr>
  <tr>
    <td><a href="http://www.manga109.org/en/">manga109</a></td>
    <td><sub>109 images of Japanese manga for testing</sub></td>
  </tr>
  <tr>
    <td>historical</td>
    <td><sub>10 gray low-resolution images without the ground-truth</sub></td>
  </tr>

  <tr>
    <td rowspan="3">2K Resolution</td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">DIV2K</a></td>
    <td><sub>proposed in <a href="http://www.vision.ee.ethz.ch/ntire17/">NTIRE17</a> (800 train and 100 validation)</sub></td>
    <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">official website</a></td>
  </tr>
 <tr>
    <td><a href="https://github.com/LimBee/NTIRE2017">Flickr2K</a></td>
    <td><sub>2650 2K images from Flickr for training</sub></td>
    <td><a href="https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar">official website</a></td>
  </tr>
 <tr>
    <td>DF2K</td>
    <td><sub>A merged training dataset of DIV2K and Flickr2K</sub></td>
    <td>-</a></td>
  </tr>

  <tr>
    <td rowspan="2">OST (Outdoor Scenes)</td>
    <td>OST Training</td>
    <td><sub>7 categories images with rich textures</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/u/1/folders/1iZfzAxAwOpeutz27HC56_y5RNqnsPPKr">Google Drive</a> / <a href="https://pan.baidu.com/s/1neUq5tZ4yTnOEAntZpK_rQ#list/path=%2Fpublic%2FSFTGAN&parentPath=%2Fpublic">Baidu Drive</a></td>
  </tr>
 <tr>
    <td>OST300</td>
    <td><sub>300 test images of outdoor scenes</sub></td>
  </tr>

  <tr>
    <td >PIRM</td>
    <td>PIRM</td>
    <td><sub>PIRM self-val, val, test datasets</sub></td>
    <td rowspan="2"><a href="https://drive.google.com/drive/folders/17FmdXu5t8wlKwt8extb_nQAdjxUOrb1O?usp=sharing">Google Drive</a> / <a href="https://pan.baidu.com/s/1gYv4tSJk_RVCbCq4B6UxNQ">Baidu Drive</a></td>
  </tr>
</table>

## 视频帧数据

推荐把数据通过 `ln -s xxx yyy` 软链到`BasicSR/datasets`下. 如果你的文件结构不同, 需要相应地修改configuration yaml文件的路径.

### REDS

[官网](https://seungjunnah.github.io/Datasets/reds.html) <br>
我们重新整合了 training 和 validation 数据到一个文件夹中: 训练集合原来有240个clip (序号从000到239), 我们把validation clips重命名, 从240到269.

**Validation的划分**

官方的validation划分和EDVR的划分不同 (当时为了比赛的设置):

| name | clips | total number |
|:----------:|:----------:|:----------:|
| REDSOfficial | [240, 269] | 30 clips |
| REDS4 | 000, 011, 015, 020 clips from the *original training set* | 4 clips |

余下的clips拿来做训练集合. 注意: 我们不需要显式地分开训练和验证集合, dataloader会做这件事.

**数据准备步骤**

1. 从[官网](https://seungjunnah.github.io/Datasets/reds.html)下载数据
1. 整合 training 和 validation 数据: `python scripts/data_preparation/regroup_reds_dataset.py`
1. [可选] 若需要使用 LMDB, 则需要制作 LMDB, 参考 [LMDB具体说明](#LMDB具体说明).  `python scripts/data_preparation/create_lmdb.py`, 注意选择`create_lmdb_for_reds`函数, 并需要修改函数相应的配置和路径.
1. 测试: `python tests/test_reds_dataset.py`, 注意修改函数相应的配置和路径.

### Vimeo90K

[官网](http://toflow.csail.mit.edu/)

**数据准备步骤**

1. 下载数据: [`Septuplets dataset --> The original training + test set (82GB)`](http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip). 这些是Ground-Truth. 里面有`sep_trainlist.txt`文件来区分训练数据.
1. 生成低分辨率图片. (TODO)
The low-resolution images in the Vimeo90K test dataset are generated with the MATLAB bicubic downsampling kernel. Use the script `data_scripts/generate_LR_Vimeo90K.m` (run in MATLAB) to generate the low-resolution images.
1. [可选] 若需要使用 LMDB, 则需要制作 LMDB, 参考 [LMDB具体说明](#LMDB具体说明).  `python scripts/data_preparation/create_lmdb.py`, 注意选择`create_lmdb_for_vimeo90k`函数, 并需要修改函数相应的配置和路径.
1. 测试: `python tests/test_vimeo90k_dataset.py`, 注意修改函数相应的配置和路径.

## StyleGAN2

### FFHQ

训练数据集: [FFHQ](https://github.com/NVlabs/ffhq-dataset).

1. 下载 FFHQ 数据集. 推荐从 [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) 下载 tfrecords 文件.
1. 从 tfrecords 提取到*图片*或者*LMDB*. (需要安装 TensorFlow 来读取 tfrecords). 我们对每一个分辨率的人脸都单独创建文件夹或者LMDB文件.

    ```bash
    python scripts/data_preparation/extract_images_from_tfrecords.py
    ```


---

# Datasets.md

# Datasets

[English](Datasets.md) **|** [简体中文](Datasets_CN.md)

## Supported Datasets

| Class         | Task   |Train/Test | Description       |
| :------------- | :----------:| :----------:    | :----------:   |
| [PairedImageDataset](../basicsr/data/paired_image_dataset.py) | Image Super-Resolution | Train|Support paired data |
| [SingleImageDataset](../basicsr/data/single_image_dataset.py) | Image Super-Resolution | Test|Only read low quality images, used in tests without Ground-Truth|
| [REDSDataset](../basicsr/data/reds_dataset.py) | Video Super-Resolution | Train|REDS training dataset |
| [Vimeo90KDataset](../basicsr/data/vimeo90k_dataset.py) | Video Super-Resolution |Train| Vimeo90K training dataset|
| [VideoTestDataset](../basicsr/data/video_test_dataset.py) | Video Super-Resolution | Test|Base video test dataset, supporting Vid4, REDS testing datasets|
| [VideoTestVimeo90KDataset](../basicsr/data/video_test_dataset.py) | Video Super-Resolution |Test| Inherit `VideoTestDataset`, Vimeo90K testing dataset|
| [VideoTestDUFDataset](../basicsr/data/video_test_dataset.py) | Video Super-Resolution |Test| Inherit `VideoTestDataset`, testing dataset for method DUF, supporting Vid4 dataset|
| [FFHQDataset](../basicsr/data/ffhq_dataset.py) | Face Generation |Train| FFHQ training dataset|

1. Common transformations and functions are in [transforms.py](../basicsr/data/transforms.py) and [util.py](../basicsr/data/util.py), respectively


---

# Datasets_CN.md

# 数据处理

[English](Datasets.md) **|** [简体中文](Datasets_CN.md)

## 支持的数据处理

| 类         | 任务    |训练/测试 | 描述       |
| :------------- | :----------:| :----------:    | :----------:   |
| [PairedImageDataset](../basicsr/data/paired_image_dataset.py) | 图像超分 | 训练|支持读取成对的训练数据 |
| [SingleImageDataset](../basicsr/data/single_image_dataset.py) | 图像超分 | 测试|只读取low quality的图像, 用在没有Ground-Truth的测试中 |
| [REDSDataset](../basicsr/data/reds_dataset.py) | 视频超分 | 训练|REDS的训练数据集 |
| [Vimeo90KDataset](../basicsr/data/vimeo90k_dataset.py) | 视频超分 |训练| Vimeo90K的训练数据集|
| [VideoTestDataset](../basicsr/data/video_test_dataset.py) | 视频超分 | 测试|基础的视频超分测试集, 支持Vid4, REDS测试集|
| [VideoTestVimeo90KDataset](../basicsr/data/video_test_dataset.py) | 视频超分 |测试| 继承`VideoTestDataset`, Vimeo90K的测试数据集|
| [VideoTestDUFDataset](../basicsr/data/video_test_dataset.py) | 视频超分 |测试| 继承`VideoTestDataset`, 方法DUF的测试数据集, 支持Vid4|
| [FFHQDataset](../basicsr/data/ffhq_dataset.py) | 人脸生成 |训练| FFHQ的训练数据集|

1. 共用的变换和函数分别在 [transforms.py](../basicsr/data/transforms.py) 和 [util.py](../basicsr/data/util.py) 中


---

# DesignConvention.md

# Codebase Designs and Conventions

[English](DesignConvention.md) **|** [简体中文](DesignConvention_CN.md)

#### Contents

1. [Overall Framework](#Overall-Framework)
1. [Features](#Features)
    1. [Dynamic Instantiation](#Dynamic-Instantiation)
1. [Conventions](#Conventions)

## Overall Framework

The `BasicSR` framework can be divided into the following parts: data, model, options/configs and training process. <br>
When we modify or add a new method, we often modify/add it from the above aspects. <br>
The figure below shows the overall framework.

![overall_structure](../assets/overall_structure.png)

## Features

### Dynamic Instantiation

When we add a new class or function, it can be used directly in the configuration file. The program will automatically scan, find and instantiate according to the class name or function name in the configuration file. This process is called dynamic instantiation.

Specifically, we implement it through `importlib` and `getattr`. Taking the data module as example, we follow the below steps in [`data/__init__.py`](../basicsr/data/__init__.py):

1. Scan all the files under the data folder with '_dataset' in file names
1. Import the classes or functions in these files through `importlib`
1. Instantiate through `getattr` according to the name in the configuration file

```python
# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder)
    if v.endswith('_dataset.py')
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'basicsr.data.{file_name}')
    for file_name in dataset_filenames
]

...

# dynamic instantiation
for module in _dataset_modules:
    dataset_cls = getattr(module, dataset_type, None)
    if dataset_cls is not None:
        break
```
We use the similar techniques for the following modules. Pay attention to the conventions of file suffix when using them:

| Module         | File Suffix     | Example        |
| :------------- | :----------:    | :----------:   |
| Data           | `_dataset.py`   | `data/paired_image_dataset.py` |
| Model          | `_model.py`     | `basicsr/models/sr_model.py` |
| Archs          | `_arch.py`      | `basicsr/archs/srresnet_arch.py`|

Note:

1. The above file suffixes are only used when necessary. Other file names should avoid using the above suffixes.
1. Note that the class name or function name cannot be repeated.

In addition, we also use `importlib` and `getattr` for `losses` and `metrics`. However, for losses and metrics, the number of files is smaller and the changes are less. So, we do not use the strategy of scanning files.
For these two modules, after adding new classes or functions, we need to add the corresponding class or function names to `__init__.py`.

| Module         | Path     | Modify `__init__.py`        |
| :------------- | :----------:    | :----------:   |
| Losses           | `basicsr/models/losses`   | [`basicsr/models/losses/__init__.py`](../basicsr/models/losses/__init__.py) |
| Metrics          | `basicsr/metrics`     | [`basicsr/metrics/__init__.py`](../basicsr/metrics/__init__.py)|

## Conventions

1. In dynamic instantiation, there are requirements to the file suffix in the following module. Otherwise, automatic instantiation cannot be achieved.

    | Module         | File Suffix     | Example        |
    | :------------- | :----------:    | :----------:   |
    | Data           | `_dataset.py`   | `data/paired_image_dataset.py` |
    | Model          | `_model.py`     | `basicsr/models/sr_model.py` |
    | Archs          | `_arch.py`      | `basicsr/archs/srresnet_arch.py`|

1. When logging, the loss items are recommended to start with `l_`, so that all these loss items will be grouped together in tensorboard. For example, in [basicsr/models/srgan_model.py](../basicsr/models/srgan_model.py), we use `l_g_pix`, `l_g_percep`, `l_g_gan`, etc for loss items. In [basicsr/utils/logger.py](../basicsr/utils/logger.py), these items will be grouped together:

    ```python
    if k.startswith('l_'):
        self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
    else:
        self.tb_logger.add_scalar(k, v, current_iter)
    ```


---

# DesignConvention_CN.md

# 代码库的设计和约定

[English](DesignConvention.md) **|** [简体中文](DesignConvention_CN.md)

#### 目录

1. [整体框架](#整体框架)
1. [特性](#特性)
    1. [动态实例化](#动态实例化)
1. [约定](#约定)

## 整体框架

整个 `BasicSR` 框架可以分为以下几个部分 —— 数据 (Data), 模型 (Model), 配置文件 (Options/Configs) 和训练过程.<br>
当我们修改或定义新的方法时, 也往往是从以上几个方面进行修改/添加的.<br>
下图概括了整体的框架.

![overall_structure](../assets/overall_structure.png)

## 特性

### 动态实例化

(Dynamic Instantiation)<br>

当我们新写了类 (Class) 或 函数 时, 可直接在配置文件中使用. 程序会根据配置文件的类名 或 函数名, 自动查找并实例化. 这个过程称为 动态实例化 (Dynamic Instantiation).

具体而言, 我们是通过 `importlib` 和 `getattr` 来实现的. 以data为例, 我们在[`data/__init__.py`](../basicsr/data/__init__.py) 中是如下做的:

1. 扫描所有以`_dataset.py`为结尾的文件 (这是约定)
1. 把这些文件中的 类 或 函数 通过 importlib 都 import 进来
1. 根据配置文件中的名称, 通过`getattr`实例化

```python
# automatically scan and import dataset modules
# scan all the files under the data folder with '_dataset' in file names
data_folder = osp.dirname(osp.abspath(__file__))
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(data_folder)
    if v.endswith('_dataset.py')
]
# import all the dataset modules
_dataset_modules = [
    importlib.import_module(f'basicsr.data.{file_name}')
    for file_name in dataset_filenames
]

...

# dynamic instantiation
for module in _dataset_modules:
    dataset_cls = getattr(module, dataset_type, None)
    if dataset_cls is not None:
        break
```

我们对以下模块使用了类似的技巧, 在使用的时候需要注意文件后缀名称的约定:

| Module         | File Suffix     | Example        |
| :------------- | :----------:    | :----------:   |
| Data           | `_dataset.py`   | `data/paired_image_dataset.py` |
| Model          | `_model.py`     | `basicsr/models/sr_model.py` |
| Archs          | `_arch.py`      | `basicsr/archs/srresnet_arch.py`|

注意:

1. 上面的文件后缀只用在需要的文件中, 其他文件命名尽量避免使用以上的后缀
1. 注意 类名 或 函数名 不能重复

另外对 `losses` 和 `metrics`, 我们也使用了 `importlib` 和 `getattr`, 但是和上面不一样的是, 对于losses和metrics, 由于文件数量比较少, 改动也少, 因此我们不采用扫描文件的方式, 而是在新增加类/函数后, 需要在相应的 `__init__.py` 中增加类/函数名称.

| Module         | Path     | Modify `__init__.py`        |
| :------------- | :----------:    | :----------:   |
| Losses           | `basicsr/models/losses`   | [`basicsr/models/losses/__init__.py`](../basicsr/models/losses/__init__.py) |
| Metrics          | `basicsr/metrics`     | [`basicsr/metrics/__init__.py`](../basicsr/metrics/__init__.py)|

## 约定

1. 动态实例化, 以下模块文件后缀名有要求, 否则不能做到自动实例化.

    | Module         | File Suffix     | Example        |
    | :------------- | :----------:    | :----------:   |
    | Data           | `_dataset.py`   | `data/paired_image_dataset.py` |
    | Model          | `_model.py`     | `basicsr/models/sr_model.py` |
    | Archs          | `_arch.py`      | `basicsr/archs/srresnet_arch.py`|

1. 在Log的时候, loss项使用`l_`开头, 这样在 tensorboard 显示的时候, 所有loss会被组织到一起. 比如在 [basicsr/models/srgan_model.py](../basicsr/models/srgan_model.py)中, 使用了`l_g_pix`, `l_g_percep`, `l_g_gan`等. 在[basicsr/utils/logger.py](../basicsr/utils/logger.py), 他们会被组织到一起:

    ```python
    if k.startswith('l_'):
        self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
    else:
        self.tb_logger.add_scalar(k, v, current_iter)
    ```


---

# HOWTOs.md

# HOWTOs

[English](HOWTOs.md) **|** [简体中文](HOWTOs_CN.md)

## How to train StyleGAN2

1. Prepare training dataset: [FFHQ](https://github.com/NVlabs/ffhq-dataset). More details are in [DatasetPreparation.md](DatasetPreparation.md#StyleGAN2)
    1. Download FFHQ dataset. Recommend to download the tfrecords files from [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset).
    1. Extract tfrecords to images or LMDBs (TensorFlow is required to read tfrecords):

        > python scripts/data_preparation/extract_images_from_tfrecords.py

1. Modify the config file in `options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml`
1. Train with distributed training. More training commands are in [TrainTest.md](TrainTest.md).

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ_800k.yml --launcher pytorch

## How to inference StyleGAN2

1. Download pre-trained models from **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) to the `experiments/pretrained_models` folder.
1. Test.

    > python inference/inference_stylegan2.py

1. The results are in the `samples` folder.

## How to inference DFDNet

1. Install [dlib](http://dlib.net/), because DFDNet uses dlib to do face recognition and landmark detection. [Installation reference](https://github.com/davisking/dlib).
    1. Clone dlib repo: `git clone git@github.com:davisking/dlib.git`
    1. `cd dlib`
    1. Install: `python setup.py install`
2. Download the dlib pretrained models from **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) to the `experiments/pretrained_models/dlib` folder.<br>
    You can download by run the following command OR manually download the pretrained models.

    > python scripts/download_pretrained_models.py dlib

3. Download pretrained DFDNet models, dictionary and face template from **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) to the `experiments/pretrained_models/DFDNet` folder.<br>
    You can download by run the the following command OR manually download the pretrained models.

    > python scripts/download_pretrained_models.py DFDNet

4. Prepare the testing dataset in the `datasets`, for example, we put images in the `datasets/TestWhole` folder.
5. Test.

    >  python inference/inference_dfdnet.py --upscale_factor=2 --test_path datasets/TestWhole

6. The results are in the `results/DFDNet` folder.

## How to train SwinIR (SR)

We take the classical SR X4 with DIV2K for example.

1. Prepare the training dataset: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/). More details are in [DatasetPreparation.md](DatasetPreparation.md#image-super-resolution)
1. Prepare the validation dataset: Set5. You can download with [this guidance](DatasetPreparation.md#common-image-sr-datasets)
1. Modify the config file in [`options/train/SwinIR/train_SwinIR_SRx4_scratch.yml`](../options/train/SwinIR/train_SwinIR_SRx4_scratch.yml) accordingly.
1. Train with distributed training. More training commands are in [TrainTest.md](TrainTest.md).

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4331 basicsr/train.py -opt options/train/SwinIR/train_SwinIR_SRx4_scratch.yml --launcher pytorch  --auto_resume

Note that:

1. Different from the original setting in the paper where the X4 model is finetuned from the X2 model, we directly train it from scratch.
1. We also use `EMA (Exponential Moving Average)`. Note that all model trainings in BasicSR supports EMA.
1. In the **250K iteration** of training X4 model, it can achieve comparable performance to the official model.

|  ClassicalSR DIV2KX4 | PSNR (RGB) | PSNR (Y) | SSIM (RGB)  | SSIM (Y) |
| :--- | :---:        |     :---:      | :---: | :---:        |
|  Official  | 30.803 | 32.728 | 0.8738|0.9028 |
|  Reproduce |30.832  | 32.756 | 0.8739| 0.9025 |

## How to inference SwinIR (SR)

1. Download pre-trained models from the [**official SwinIR repo**](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) to the `experiments/pretrained_models/SwinIR` folder.
1. Inference.

    > python inference/inference_swinir.py --input datasets/Set5/LRbicx4 --patch_size 48 --model_path experiments/pretrained_models/SwinIR/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth --output results/SwinIR_SRX4_DIV2K/Set5

1. The results are in the `results/SwinIR_SRX4_DIV2K/Set5` folder.
1. You may want to calculate the PSNR/SSIM values.

    > python scripts/metrics/calculate_psnr_ssim.py --gt datasets/Set5/GTmod12/ --restored results/SwinIR_SRX4_DIV2K/Set5 --crop_border 4

    or test with the Y channel with the `--test_y_channel` argument.

    > python scripts/metrics/calculate_psnr_ssim.py --gt datasets/Set5/GTmod12/ --restored results/SwinIR_SRX4_DIV2K/Set5 --crop_border 4  --test_y_channel


---

# HOWTOs_CN.md

# HOWTOs

[English](HOWTOs.md) **|** [简体中文](HOWTOs_CN.md)

## 如何训练 StyleGAN2

1. 准备训练数据集: [FFHQ](https://github.com/NVlabs/ffhq-dataset). 更多细节: [DatasetPreparation_CN.md](DatasetPreparation_CN.md#StyleGAN2)
    1. 下载 FFHQ 数据集. 推荐从 [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) 下载 tfrecords 文件.
    1. 从tfrecords 提取到*图片*或者*LMDB*. (需要安装 TensorFlow 来读取 tfrecords).

        > python scripts/data_preparation/extract_images_from_tfrecords.py

1. 修改配置文件 `options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml`
1. 使用分布式训练. 更多训练命令: [TrainTest_CN.md](TrainTest_CN.md)

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/StyleGAN/train_StyleGAN2_256_Cmul2_FFHQ.yml --launcher pytorch

## 如何测试 StyleGAN2

1. 从 **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) 下载预训练模型到 `experiments/pretrained_models` 文件夹.
1. 测试.

    > python inference/inference_stylegan2.py

1. 结果在 `samples` 文件夹

## 如何测试 DFDNet

1. 安装 [dlib](http://dlib.net/). 因为 DFDNet 使用 dlib 做人脸检测和关键点检测. [安装参考](https://github.com/davisking/dlib).
    1. 克隆 dlib repo: `git clone git@github.com:davisking/dlib.git`
    1. `cd dlib`
    1. 安装: `python setup.py install`
2. 从 **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) 下载预训练的 dlib 模型到 `experiments/pretrained_models/dlib` 文件夹.<br>
    你可以通过运行下面的命令下载 或 手动下载.

    > python scripts/download_pretrained_models.py dlib

3. 从 **ModelZoo** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g)) 下载 DFDNet 模型, 字典和人脸关键点模板到 `experiments/pretrained_models/DFDNet` 文件夹.<br>
     你可以通过运行下面的命令下载 或 手动下载.

    > python scripts/download_pretrained_models.py DFDNet

4. 准备测试图片到 `datasets`, 比如说我们把测试图片放在 `datasets/TestWhole` 文件夹.
5. 测试.

    >  python inference/inference_dfdnet.py --upscale_factor=2 --test_path datasets/TestWhole

6. 结果在 `results/DFDNet` 文件夹.

## How to train SwinIR (SR)

We take the classical SR X4 with DIV2K for example.

1. Prepare the training dataset: [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/). More details are in [DatasetPreparation.md](DatasetPreparation.md#image-super-resolution)
1. Prepare the validation dataset: Set5. You can download with [this guidance](DatasetPreparation.md#common-image-sr-datasets)
1. Modify the config file in [`options/train/SwinIR/train_SwinIR_SRx4_scratch.yml`](../options/train/SwinIR/train_SwinIR_SRx4_scratch.yml) accordingly.
1. Train with distributed training. More training commands are in [TrainTest.md](TrainTest.md).

    > python -m torch.distributed.launch --nproc_per_node=8 --master_port=4331 basicsr/train.py -opt options/train/SwinIR/train_SwinIR_SRx4_scratch.yml --launcher pytorch  --auto_resume

Note that:

1. Different from the original setting in the paper where the X4 model is finetuned from the X2 model, we directly train it from scratch.
1. We also use `EMA (Exponential Moving Average)`. Note that all model trainings in BasicSR supports EMA.
1. In the **250K iteration** of training X4 model, it can achieve comparable performance to the official model.

|  ClassicalSR DIV2KX4 | PSNR (RGB) | PSNR (Y) | SSIM (RGB)  | SSIM (Y) |
| :--- | :---:        |     :---:      | :---: | :---:        |
|  Official  | 30.803 | 32.728 | 0.8738|0.9028 |
|  Reproduce |30.832  | 32.756 | 0.8739| 0.9025 |

## How to inference SwinIR (SR)

1. Download pre-trained models from the [**official SwinIR repo**](https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0) to the `experiments/pretrained_models/SwinIR` folder.
1. Inference.

    > python inference/inference_swinir.py --input datasets/Set5/LRbicx4 --patch_size 48 --model_path experiments/pretrained_models/SwinIR/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth --output results/SwinIR_SRX4_DIV2K/Set5

1. The results are in the `results/SwinIR_SRX4_DIV2K/Set5` folder.
1. You may want to calculate the PSNR/SSIM values.

    > python scripts/metrics/calculate_psnr_ssim.py --gt datasets/Set5/GTmod12/ --restored results/SwinIR_SRX4_DIV2K/Set5 --crop_border 4

    or test with the Y channel with the `--test_y_channel` argument.

    > python scripts/metrics/calculate_psnr_ssim.py --gt datasets/Set5/GTmod12/ --restored results/SwinIR_SRX4_DIV2K/Set5 --crop_border 4  --test_y_channel


---

# Logging.md

# Logging

[English](Logging.md) **|** [简体中文](Logging_CN.md)

#### Contents

1. [Text Logger](#Text-Logger)
1. [Tensorboard Logger](#Tensorboard-Logger)
1. [Wandb Logger](#Wandb-Logger)

## Text Logger

Print the log to both the text file and screen. The text file usually locates in `experiments/exp_name/train_exp_name_timestamp.txt`.

## Tensorboard Logger

- Use Tensorboard logger. Set `use_tb_logger: true` in the yml configuration file:

    ```yml
    logger:
      use_tb_logger: true
    ```

- File location: `tb_logger/exp_name`
- View in the browser:

    ```bash
    tensorboard --logdir tb_logger --port 5500 --bind_all
    ```

## Wandb Logger

[wandb](https://www.wandb.com/) can be viewed as a cloud version of tensorboard. One can easily view training processes and curves in wandb. Currently, we only sync the tensorboard log to wandb. So we should also turn on tensorboard when using wandb.

Configuration file:

```yml
ogger:
  # Whether to tensorboard logger
  use_tb_logger: true
  # Whether to use wandb logger. Currently, wandb only sync the tensorboard log. So we should also turn on tensorboard when using wandb
  wandb:
    # wandb project name. Default is None, that is not using wandb.
    # Here, we use the basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # If resuming, wandb id could automatically link previous logs
    resume_id: ~
```

**[Examples of training curves in wandb](https://app.wandb.ai/xintao/basicsr)**

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="../assets/wandb.jpg" height="280">
</a></p>


---

# Logging_CN.md

# Logging日志

[English](Logging.md) **|** [简体中文](Logging_CN.md)

#### 目录

1. [文本屏幕日志](#文本屏幕日志)
1. [Tensorboard日志](#Tensorboard日志)
1. [Wandb日志](#Wandb日志)

## 文本屏幕日志

将日志信息同时输出到文件和屏幕. 文件位置一般为`experiments/exp_name/train_exp_name_timestamp.txt`.

## Tensorboard日志

- 开启. 在 yml 配置文件中设置 `use_tb_logger: true`:

    ```yml
    logger:
      use_tb_logger: true
    ```

- 文件位置: `tb_logger/exp_name`
- 在浏览器中查看:

    ```bash
    tensorboard --logdir tb_logger --port 5500 --bind_all
    ```

## Wandb日志

[wandb](https://www.wandb.com/) 类似tensorboard的云端版本, 可以在浏览器方便地查看模型训练的过程和曲线. 我们目前只是把tensorboard的内容同步到wandb上, 因此要使用wandb, 必须打开tensorboard logger.

配置文件如下:

```yml
logger:
  # 是否使用tensorboard logger
  use_tb_logger: true
  # 是否使用wandb logger, 目前wandb只是同步tensorboard的内容, 因此要使用wandb, 必须也同时使用tensorboard
  wandb:
    # wandb的project. 默认是 None, 即不使用wandb.
    # 这里使用了 basicsr wandb project: https://app.wandb.ai/xintao/basicsr
    project: basicsr
    # 如果是resume, 可以输入上次的wandb id, 则log可以接起来
    resume_id: ~
```

**[wandb训练曲线样例](https://app.wandb.ai/xintao/basicsr)**

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="../assets/wandb.jpg" height="280">
</a></p>


---

# Metrics.md

# Metrics

[English](Metrics.md) **|** [简体中文](Metrics_CN.md)

## PSNR and SSIM

## NIQE

## FID

> FID measures the similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks.
> FID is calculated by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two Gaussians fitted to feature representations of the Inception network.

References

- https://github.com/mseitzer/pytorch-fid
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
- [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337)

### Pre-calculated FFHQ inception feature statistics

Usually, we put the downloaded inception feature statistics in `basicsr/metrics`.

:arrow_double_down: Google Drive: [metrics data](https://drive.google.com/drive/folders/13cWIQyHX3iNmZRJ5v8v3kdyrT9RBTAi9?usp=sharing)
:arrow_double_down: 百度网盘: [评价指标数据](https://pan.baidu.com/s/10mMKXSEgrC5y7m63W5vbMQ) <br>

| File Name         | Dataset | Image Shape    | Sample Numbers|
| :------------- | :----------:|:----------:|:----------:|
| inception_FFHQ_256-0948f50d.pth | FFHQ | 256 x 256 | 50,000 |
| inception_FFHQ_512-f7b384ab.pth | FFHQ | 512 x 512 | 50,000 |
| inception_FFHQ_1024-75f195dc.pth | FFHQ | 1024 x 1024 | 50,000 |
| inception_FFHQ_256_stylegan2_pytorch-abba9d31.pth | FFHQ | 256 x 256 | 50,000 |

- All the FFHQ inception feature statistics calculated on the resized 299 x 299 size.
- `inception_FFHQ_256_stylegan2_pytorch-abba9d31.pth` is converted from the statistics in [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).


---

# Metrics_CN.md

# 评价指标

[English](Metrics.md) **|** [简体中文](Metrics_CN.md)

## PSNR and SSIM

## NIQE

## FID

> FID measures the similarity between two datasets of images. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks.
> FID is calculated by computing the [Fréchet distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distance) between two Gaussians fitted to feature representations of the Inception network.

参考

- https://github.com/mseitzer/pytorch-fid
- [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
- [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/abs/1711.10337)

### Pre-calculated FFHQ inception feature statistics

通常, 我们把下载的 inception 网络的特征统计数据 (用于计算FID) 放在 `basicsr/metrics`.


:arrow_double_down: 百度网盘: [评价指标数据](https://pan.baidu.com/s/10mMKXSEgrC5y7m63W5vbMQ)
:arrow_double_down: Google Drive: [metrics data](https://drive.google.com/drive/folders/13cWIQyHX3iNmZRJ5v8v3kdyrT9RBTAi9?usp=sharing) <br>

| File Name         | Dataset | Image Shape    | Sample Numbers|
| :------------- | :----------:|:----------:|:----------:|
| inception_FFHQ_256-0948f50d.pth | FFHQ | 256 x 256 | 50,000 |
| inception_FFHQ_512-f7b384ab.pth | FFHQ | 512 x 512 | 50,000 |
| inception_FFHQ_1024-75f195dc.pth | FFHQ | 1024 x 1024 | 50,000 |
| inception_FFHQ_256_stylegan2_pytorch-abba9d31.pth | FFHQ | 256 x 256 | 50,000 |

- All the FFHQ inception feature statistics calculated on the resized 299 x 299 size.
- `inception_FFHQ_256_stylegan2_pytorch-abba9d31.pth` is converted from the statistics in [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch).


---

# ModelZoo.md

# Model Zoo and Baselines

[English](ModelZoo.md) **|** [简体中文](ModelZoo_CN.md)

:arrow_double_down: Google Drive: [Pretrained Models](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) **|** [Reproduced Experiments](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing)
:arrow_double_down: 百度网盘: [预训练模型](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g) **|** [复现实验](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ)

---

We provide:

1. Official models converted directly from official released models
1. Reproduced models with `BasicSR`. Pre-trained models and log examples are provided

You can put the downloaded models in the `experiments/pretrained_models` folder.

**[Download official pre-trained models]** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g))

You can use the scrip to download pre-trained models from Google Drive.

```python
python scripts/download_pretrained_models.py ESRGAN
# method can be ESRGAN, EDVR, StyleGAN, EDSR, DUF, DFDNet, dlib
```

**[Download reproduced models and logs]** ([Google Drive](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing), [百度网盘](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ))

In addition, we upload the training process and curves in [wandb](https://www.wandb.com/).

**[Training curves in wandb](https://app.wandb.ai/xintao/basicsr)**

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="../assets/wandb.jpg" height="350">
</a></p>

#### Contents

1. [Image Super-Resolution](#Image-Super-Resolution)
    1. [Image SR Official Models](#Image-SR-Official-Models)
    1. [Image SR Reproduced Models](#Image-SR-Reproduced-Models)
1. [Video Super-Resolution](#Video-Super-Resolution)

## Image Super-Resolution

When evaluation:

- We crop `scale` border pixels in each border
- Evaluated on RGB channels

### Image SR Official Models

|Exp Name         | Set5 (PSNR/SSIM)     | Set14 (PSNR/SSIM)   |DIV2K100 (PSNR/SSIM)   |
| :------------- | :----------:    | :----------:   |:----------:   |
| EDSR_Mx2_f64b16_DIV2K_official-3ba7b086 | 35.7768 / 0.9442 | 31.4966 / 0.8939 | 34.6291 / 0.9373 |
| EDSR_Mx3_f64b16_DIV2K_official-6908f88a | 32.3597 / 0.903 | 28.3932 / 0.8096 | 30.9438 / 0.8737 |
| EDSR_Mx4_f64b16_DIV2K_official-0c287733 | 30.1821 / 0.8641 | 26.7528 / 0.7432 | 28.9679 / 0.8183 |
| EDSR_Lx2_f256b32_DIV2K_official-be38e77d | 35.9979 / 0.9454 | 31.8583 / 0.8971 | 35.0495 / 0.9407 |
| EDSR_Lx3_f256b32_DIV2K_official-3660f70d | 32.643 / 0.906 | 28.644 / 0.8152 | 31.28 / 0.8798 |
| EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f | 30.5499 / 0.8701 | 27.0011 / 0.7509 | 29.277 / 0.8266 |

### Image SR Reproduced Models

Experiment name conventions are in [Config.md](Config.md).

|Exp Name         | Set5 (PSNR/SSIM)     | Set14 (PSNR/SSIM)   |DIV2K100 (PSNR/SSIM)   |
| :------------- | :----------:    | :----------:   |:----------:   |
| 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb | 30.2468 / 0.8651 | 26.7817 / 0.7451 | 28.9967 / 0.8195 |
| 002_MSRResNet_x2_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 35.7483 / 0.9442 | 31.5403 / 0.8937 |34.6699 / 0.9377|
| 003_MSRResNet_x3_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 32.4038 / 0.9032| 28.4418 / 0.8106|30.9726 / 0.8743 |
| 004_MSRGAN_x4_f64b16_DIV2K_400k_B16G1_wandb | 28.0158 / 0.8087|24.7474 / 0.6623 | 26.6504 / 0.7462|
| | | | |
| 201_EDSR_Mx2_f64b16_DIV2K_300k_B16G1_wandb | 35.7395 / 0.944|31.4348 / 0.8934 |34.5798 / 0.937 |
| 202_EDSR_Mx3_f64b16_DIV2K_300k_B16G1_201pretrain_wandb|32.315 / 0.9026 |28.3866 / 0.8088 |30.9095 / 0.8731|
| 203_EDSR_Mx4_f64b16_DIV2K_300k_B16G1_201pretrain_wandb|30.1726 / 0.8641 |26.721 / 0.743 |28.9506 / 0.818|
| 204_EDSR_Lx2_f256b32_DIV2K_300k_B16G1_wandb | 35.9792 / 0.9453 | 31.7284 / 0.8959 | 34.9544 / 0.9399 |
| 205_EDSR_Lx3_f256b32_DIV2K_300k_B16G1_204pretrain_wandb | 32.6467 / 0.9057 | 28.6859 / 0.8152 | 31.2664 / 0.8793 |
| 206_EDSR_Lx4_f256b32_DIV2K_300k_B16G1_204pretrain_wandb | 30.4718 / 0.8695 | 26.9616 / 0.7502 | 29.2621 / 0.8265 |

## Video Super-Resolution

#### Evaluation

In the evaluation, we include all the input frames and do not crop any border pixels unless otherwise stated.<br/>
We do not use the self-ensemble (flip testing) strategy and any other post-processing methods.

## EDVR

**Name convention**<br/>
EDVR\_(training dataset)\_(track name)\_(model complexity)

- track name. There are four tracks in the NTIRE 2019 Challenges on Video Restoration and Enhancement:
    - **SR**: super-resolution with a fixed downsampling kernel (MATLAB bicubic downsampling kernel is frequently used). Most of the previous video SR methods focus on this setting.
    - **SRblur**: the inputs are also degraded with motion blur.
    - **deblur**: standard deblurring (motion blur).
    - **deblurcomp**: motion blur + video compression artifacts.
- model complexity
    - **L** (Large): # of channels = 128, # of back residual blocks = 40. This setting is used in our competition submission.
    - **M** (Moderate): # of channels = 64, # of back residual blocks = 10.


| Model name |[Test Set] PSNR/SSIM |
|:----------:|:----------:|
| EDVR_Vimeo90K_SR_L | [Vid4] (Y<sup>1</sup>) 27.35/0.8264 [[↓Results]](https://drive.google.com/open?id=14nozpSfe9kC12dVuJ9mspQH5ZqE4mT9K)<br/> (RGB) 25.83/0.8077|
| EDVR_REDS_SR_M | [REDS] (RGB) 30.53/0.8699 [[↓Results]](https://drive.google.com/open?id=1Mek3JIxkjJWjhZhH4qVwTXnRZutKUtC-)|
| EDVR_REDS_SR_L | [REDS] (RGB) 31.09/0.8800 [[↓Results]](https://drive.google.com/open?id=1h6E0QVZyJ5SBkcnYaT1puxYYPVbPsTLt)|
| EDVR_REDS_SRblur_L | [REDS] (RGB) 28.88/0.8361 [[↓Results]](https://drive.google.com/open?id=1-8MNkQuMVMz30UilB9m_d0SXicwFEPZH)|
| EDVR_REDS_deblur_L | [REDS] (RGB) 34.80/0.9487 [[↓Results]](https://drive.google.com/open?id=133wCHTwiiRzenOEoStNbFuZlCX8Jn2at)|
| EDVR_REDS_deblurcomp_L | [REDS] (RGB) 30.24/0.8567 [[↓Results]](https://drive.google.com/open?id=1VjC4fXBXy0uxI8Kwxh-ijj4PZkfsLuTX)  |

<sup>1</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.

#### Stage 2 models for the NTIRE19 Competition

| Model name |[Test Set] PSNR/SSIM |
|:----------:|:----------:|
| EDVR_REDS_SR_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_SRblur_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_deblur_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_deblurcomp_Stage2 | [REDS] (RGB) / [[↓Results]]()  |


## DUF
The models are converted from the [officially released models](https://github.com/yhjo09/VSR-DUF). <br/>

| Model name | [Test Set] PSNR/SSIM<sup>1</sup> | Official Results<sup>2</sup> |
|:----------:|:----------:|:----------:|
| DUF_x4_52L_official<sup>3</sup> | [Vid4] (Y<sup>4</sup>) 27.33/0.8319 [[↓Results]](https://drive.google.com/open?id=1U9xGhlDSpPPQvKN0BAzXfjUCvaFxwsQf)<br/> (RGB) 25.80/0.8138   | (Y) 27.33/0.8318 [[↓Results]](https://drive.google.com/open?id=1HUmf__cSL7td7J4cXo2wvbVb14Y8YG2j)<br/> (RGB) 25.79/0.8136 |
| DUF_x4_28L_official | [Vid4]  | |
| DUF_x4_16L_official | [Vid4]  | |
| DUF_x3_16L_official | [Vid4]  | |
| DUF_x2_16L_official | [Vid4]  | |

<sup>1</sup> We crop eight pixels near image boundary for DUF due to its severe boundary effects. <br/>
<sup>2</sup> The official results are obtained by running the official codes and models. <br/>
<sup>3</sup> Different from the official codes, where `zero padding` is used for border frames, we use `new_info` strategy. <br/>
<sup>4</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.

## TOF
The models are converted from the [officially released models](https://github.com/anchen1011/toflow).<br/>

| Model name | [Test Set] PSNR/SSIM | Official Results<sup>1</sup> |
|:----------:|:----------:|:----------:|
| TOF_official<sup>2</sup> | [Vid4] (Y<sup>3</sup>) 25.86/0.7626 [[↓Results]](https://drive.google.com/open?id=1Xp5U6uZeM44ShzawfuW_E-NmQ30hk-Be)<br/> (RGB)  24.38/0.7403 | (Y) 25.89/0.7651 [[↓Results]](https://drive.google.com/open?id=1WY3CcdzbRhpvDi3aGc1jAhIbeC6GUrM8)<br/> (RGB)  24.41/0.7428 |

<sup>1</sup> The official results are obtained by running the official codes and models. Note that TOFlow does not provide a strategy for border frame recovery and we simply use a `replicate` strategy for border frames. <br/>
<sup>2</sup> The converted model has slightly different results, due to different implementation. And we use `new_info` strategy for border frames. <br/>
<sup>3</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.


---

# ModelZoo_CN.md

# 模型库和基准

[English](ModelZoo.md) **|** [简体中文](ModelZoo_CN.md)

:arrow_double_down: 百度网盘: [预训练模型](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g) **|** [复现实验](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ)
:arrow_double_down: Google Drive: [Pretrained Models](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing) **|** [Reproduced Experiments](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing)

---

我们提供了:

1. 官方的模型, 它们是从官方release的models直接转化过来的
1. 复现的模型, 使用`BasicSR`的框架复现的, 提供模型和log的例子

下载的模型可以放在 `experiments/pretrained_models` 文件夹.

**[下载官方提供的预训练模型]** ([Google Drive](https://drive.google.com/drive/folders/15DgDtfaLASQ3iAPJEVHQF49g9msexECG?usp=sharing), [百度网盘](https://pan.baidu.com/s/1R6Nc4v3cl79XPAiK0Toe7g))
你可以使用以下脚本从Google Drive下载预训练模型.

```python
python scripts/download_pretrained_models.py ESRGAN
# method can be ESRGAN, EDVR, StyleGAN, EDSR, DUF, DFDNet, dlib
```

**[下载复现的模型和log]** ([Google Drive](https://drive.google.com/drive/folders/1XN4WXKJ53KQ0Cu0Yv-uCt8DZWq6uufaP?usp=sharing), [百度网盘](https://pan.baidu.com/s/1UElD6q8sVAgn_cxeBDOlvQ))

此外, 我们在 [wandb](https://www.wandb.com/) 上更新了模型训练的过程和曲线. 大家可以方便的比较:

**[wandb训练曲线](https://app.wandb.ai/xintao/basicsr)**

<p align="center">
<a href="https://app.wandb.ai/xintao/basicsr" target="_blank">
   <img src="../assets/wandb.jpg" height="350">
</a></p>

#### 目录

1. [图像超分辨率](#图像超分辨率)
    1. [图像超分官方模型](#图像超分官方模型)
    1. [图像超分复现模型](#图像超分复现模型)
1. [视频超分辨率](#视频超分辨率)

## 图像超分辨率

在计算指标时:

- 所有的图像各条边crop了scale的像素
- 都在RGB通道上测试

### 图像超分官方模型

|Exp Name         | Set5 (PSNR/SSIM)     | Set14 (PSNR/SSIM)   |DIV2K100 (PSNR/SSIM)   |
| :------------- | :----------:    | :----------:   |:----------:   |
| EDSR_Mx2_f64b16_DIV2K_official-3ba7b086 | 35.7768 / 0.9442 | 31.4966 / 0.8939 | 34.6291 / 0.9373 |
| EDSR_Mx3_f64b16_DIV2K_official-6908f88a | 32.3597 / 0.903 | 28.3932 / 0.8096 | 30.9438 / 0.8737 |
| EDSR_Mx4_f64b16_DIV2K_official-0c287733 | 30.1821 / 0.8641 | 26.7528 / 0.7432 | 28.9679 / 0.8183 |
| EDSR_Lx2_f256b32_DIV2K_official-be38e77d | 35.9979 / 0.9454 | 31.8583 / 0.8971 | 35.0495 / 0.9407 |
| EDSR_Lx3_f256b32_DIV2K_official-3660f70d | 32.643 / 0.906 | 28.644 / 0.8152 | 31.28 / 0.8798 |
| EDSR_Lx4_f256b32_DIV2K_official-76ee1c8f | 30.5499 / 0.8701 | 27.0011 / 0.7509 | 29.277 / 0.8266 |

### 图像超分复现模型

实验名称的命名规则参见 [Config_CN.md](Config_CN.md).

|Exp Name         | Set5 (PSNR/SSIM)     | Set14 (PSNR/SSIM)   |DIV2K100 (PSNR/SSIM)   |
| :------------- | :----------:    | :----------:   |:----------:   |
| 001_MSRResNet_x4_f64b16_DIV2K_1000k_B16G1_wandb | 30.2468 / 0.8651 | 26.7817 / 0.7451 | 28.9967 / 0.8195 |
| 002_MSRResNet_x2_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 35.7483 / 0.9442 | 31.5403 / 0.8937 |34.6699 / 0.9377|
| 003_MSRResNet_x3_f64b16_DIV2K_1000k_B16G1_001pretrain_wandb | 32.4038 / 0.9032| 28.4418 / 0.8106|30.9726 / 0.8743 |
| 004_MSRGAN_x4_f64b16_DIV2K_400k_B16G1_wandb | 28.0158 / 0.8087|24.7474 / 0.6623 | 26.6504 / 0.7462|
| | | | |
| 201_EDSR_Mx2_f64b16_DIV2K_300k_B16G1_wandb | 35.7395 / 0.944|31.4348 / 0.8934 |34.5798 / 0.937 |
| 202_EDSR_Mx3_f64b16_DIV2K_300k_B16G1_201pretrain_wandb|32.315 / 0.9026 |28.3866 / 0.8088 |30.9095 / 0.8731|
| 203_EDSR_Mx4_f64b16_DIV2K_300k_B16G1_201pretrain_wandb|30.1726 / 0.8641 |26.721 / 0.743 |28.9506 / 0.818|
| 204_EDSR_Lx2_f256b32_DIV2K_300k_B16G1_wandb | 35.9792 / 0.9453 | 31.7284 / 0.8959 | 34.9544 / 0.9399 |
| 205_EDSR_Lx3_f256b32_DIV2K_300k_B16G1_204pretrain_wandb | 32.6467 / 0.9057 | 28.6859 / 0.8152 | 31.2664 / 0.8793 |
| 206_EDSR_Lx4_f256b32_DIV2K_300k_B16G1_204pretrain_wandb | 30.4718 / 0.8695 | 26.9616 / 0.7502 | 29.2621 / 0.8265 |

## 视频超分辨率

#### Evaluation

In the evaluation, we include all the input frames and do not crop any border pixels unless otherwise stated.<br/>
We do not use the self-ensemble (flip testing) strategy and any other post-processing methods.

## EDVR

**Name convention**<br/>
EDVR\_(training dataset)\_(track name)\_(model complexity)

- track name. There are four tracks in the NTIRE 2019 Challenges on Video Restoration and Enhancement:
    - **SR**: super-resolution with a fixed downsampling kernel (MATLAB bicubic downsampling kernel is frequently used). Most of the previous video SR methods focus on this setting.
    - **SRblur**: the inputs are also degraded with motion blur.
    - **deblur**: standard deblurring (motion blur).
    - **deblurcomp**: motion blur + video compression artifacts.
- model complexity
    - **L** (Large): # of channels = 128, # of back residual blocks = 40. This setting is used in our competition submission.
    - **M** (Moderate): # of channels = 64, # of back residual blocks = 10.

| Model name |[Test Set] PSNR/SSIM |
|:----------:|:----------:|
| EDVR_Vimeo90K_SR_L | [Vid4] (Y<sup>1</sup>) 27.35/0.8264 [[↓Results]](https://drive.google.com/open?id=14nozpSfe9kC12dVuJ9mspQH5ZqE4mT9K)<br/> (RGB) 25.83/0.8077|
| EDVR_REDS_SR_M | [REDS] (RGB) 30.53/0.8699 [[↓Results]](https://drive.google.com/open?id=1Mek3JIxkjJWjhZhH4qVwTXnRZutKUtC-)|
| EDVR_REDS_SR_L | [REDS] (RGB) 31.09/0.8800 [[↓Results]](https://drive.google.com/open?id=1h6E0QVZyJ5SBkcnYaT1puxYYPVbPsTLt)|
| EDVR_REDS_SRblur_L | [REDS] (RGB) 28.88/0.8361 [[↓Results]](https://drive.google.com/open?id=1-8MNkQuMVMz30UilB9m_d0SXicwFEPZH)|
| EDVR_REDS_deblur_L | [REDS] (RGB) 34.80/0.9487 [[↓Results]](https://drive.google.com/open?id=133wCHTwiiRzenOEoStNbFuZlCX8Jn2at)|
| EDVR_REDS_deblurcomp_L | [REDS] (RGB) 30.24/0.8567 [[↓Results]](https://drive.google.com/open?id=1VjC4fXBXy0uxI8Kwxh-ijj4PZkfsLuTX)  |

<sup>1</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.

#### Stage 2 models for the NTIRE19 Competition

| Model name |[Test Set] PSNR/SSIM |
|:----------:|:----------:|
| EDVR_REDS_SR_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_SRblur_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_deblur_Stage2 | [REDS] (RGB) / [[↓Results]]()|
| EDVR_REDS_deblurcomp_Stage2 | [REDS] (RGB) / [[↓Results]]()  |


## DUF
The models are converted from the [officially released models](https://github.com/yhjo09/VSR-DUF). <br/>

| Model name | [Test Set] PSNR/SSIM<sup>1</sup> | Official Results<sup>2</sup> |
|:----------:|:----------:|:----------:|
| DUF_x4_52L_official<sup>3</sup> | [Vid4] (Y<sup>4</sup>) 27.33/0.8319 [[↓Results]](https://drive.google.com/open?id=1U9xGhlDSpPPQvKN0BAzXfjUCvaFxwsQf)<br/> (RGB) 25.80/0.8138   | (Y) 27.33/0.8318 [[↓Results]](https://drive.google.com/open?id=1HUmf__cSL7td7J4cXo2wvbVb14Y8YG2j)<br/> (RGB) 25.79/0.8136 |
| DUF_x4_28L_official | [Vid4]  | |
| DUF_x4_16L_official | [Vid4]  | |
| DUF_x3_16L_official | [Vid4]  | |
| DUF_x2_16L_official | [Vid4]  | |

<sup>1</sup> We crop eight pixels near image boundary for DUF due to its severe boundary effects. <br/>
<sup>2</sup> The official results are obtained by running the official codes and models. <br/>
<sup>3</sup> Different from the official codes, where `zero padding` is used for border frames, we use `new_info` strategy. <br/>
<sup>4</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.

## TOF
The models are converted from the [officially released models](https://github.com/anchen1011/toflow).<br/>

| Model name | [Test Set] PSNR/SSIM | Official Results<sup>1</sup> |
|:----------:|:----------:|:----------:|
| TOF_official<sup>2</sup> | [Vid4] (Y<sup>3</sup>) 25.86/0.7626 [[↓Results]](https://drive.google.com/open?id=1Xp5U6uZeM44ShzawfuW_E-NmQ30hk-Be)<br/> (RGB)  24.38/0.7403 | (Y) 25.89/0.7651 [[↓Results]](https://drive.google.com/open?id=1WY3CcdzbRhpvDi3aGc1jAhIbeC6GUrM8)<br/> (RGB)  24.41/0.7428 |

<sup>1</sup> The official results are obtained by running the official codes and models. Note that TOFlow does not provide a strategy for border frame recovery and we simply use a `replicate` strategy for border frames. <br/>
<sup>2</sup> The converted model has slightly different results, due to different implementation. And we use `new_info` strategy for border frames. <br/>
<sup>3</sup> Y or RGB denotes the evaluation on Y (luminance) or RGB channels.


---

# Models.md

# Models

[English](Models.md) **|** [简体中文](Models_CN.md)

#### Contents

1. [Supported Models](#Supported-Models)
1. [Inheritance Relationship](#Inheritance-Relationship)

## Supported Models

| Class         | Description    |Supported Algorithms |
| :------------- | :----------:| :----------:    |
| [BaseModel](../basicsr/models/base_model.py) | Abstract base class, define common functions||
| [SRModel](../basicsr/models/sr_model.py) | Base image SR class | SRCNN, EDSR, SRResNet, RCAN, RRDBNet, etc |
| [SRGANModel](../basicsr/models/srgan_model.py) | SRGAN image SR class | SRGAN |
| [ESRGANModel](../basicsr/models/esrgan_model.py) | ESRGAN image SR class|ESRGAN|
| [VideoBaseModel](../basicsr/models/video_base_model.py) | Base video SR class | |
| [EDVRModel](../basicsr/models/edvr_model.py) | EDVR video SR class |EDVR|
| [StyleGAN2Model](../basicsr/models/stylegan2_model.py) | StyleGAN2 generation class |StyleGAN2|

## Inheritance Relationship

In order to reuse components among models, we use a lot of inheritance. The following is the inheritance relationship:

```txt
BaseModel
├── SRModel
│   ├── SRGANModel
│   │   └── ESRGANModel
│   └── VideoBaseModel
│       ├── VideoGANModel
│       └── EDVRModel
└── StyleGAN2Model
```


---

# Models_CN.md

# 模型

[English](Models.md) **|** [简体中文](Models_CN.md)

#### 目录

1. [支持的模型](#支持的模型)
1. [继承关系](#继承关系)

## 支持的模型

| 类         | 描述    |支持的算法 |
| :------------- | :----------:| :----------:    |
| [BaseModel](../basicsr/models/base_model.py) | 抽象基类, 定义了共有的函数||
| [SRModel](../basicsr/models/sr_model.py) | 基础图像超分类 | SRCNN, EDSR, SRResNet, RCAN, RRDBNet, etc |
| [SRGANModel](../basicsr/models/srgan_model.py) | SRGAN图像超分类 | SRGAN |
| [ESRGANModel](../basicsr/models/esrgan_model.py) | ESRGAN图像超分类 |ESRGAN|
| [VideoBaseModel](../basicsr/models/video_base_model.py) | 基础视频超分类 | |
| [EDVRModel](../basicsr/models/edvr_model.py) | EDVR视频超分类 |EDVR|
| [StyleGAN2Model](../basicsr/models/stylegan2_model.py) | StyleGAN2图像生成类 |StyleGAN2|

## 继承关系

为增加模型间的复用, 我们大量使用了继承, 以下为各个模型之间的继承关系:

```txt
BaseModel
├── SRModel
│   ├── SRGANModel
│   │   ├── ESRGANModel
│   ├── VideoBaseModel
│       ├── EDVRModel
├── StyleGAN2Model
```


---

# TrainTest.md

# Training and Testing

[English](TrainTest.md) **|** [简体中文](TrainTest_CN.md)

Please run the commands in the root path of `BasicSR`. <br>
In general, both the training and testing include the following steps:

1. Prepare datasets. Please refer to [DatasetPreparation.md](DatasetPreparation.md)
1. Modify config files. The config files are under the `options` folder. For more specific configuration information, please refer to [Config](Config.md)
1. [Optional] You may need to download pre-trained models if you are testing or using pre-trained models. Please see [ModelZoo](ModelZoo.md)
1. Run commands. Use [Training Commands](#Training-Commands) or [Testing Commands](#Testing-Commands) accordingly.

#### 目录

1. [Training Commands](#Training-Commands)
    1. [Single GPU Training](#Single-GPU-Training)
    1. [Distributed (Multi-GPUs) Training](#Distributed-Training)
    1. [Slurm Training](#Slurm-Training)
1. [Testing Commands](#Testing-Commands)
    1. [Single GPU Testing](#Single-GPU-Testing)
    1. [Distributed (Multi-GPUs) Testing](#Distributed-Testing)
    1. [Slurm Testing](#Slurm-Testing)

## Training Commands

### Single GPU Training

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml

### Distributed Training

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> ./scripts/dist_train.sh 8 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> ./scripts/dist_train.sh 4 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml

### Slurm Training

[Introduction to Slurm](https://slurm.schedmd.com/quickstart.html)

**1 GPU**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=MSRResNetx4 --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml --launcher="slurm"

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=EDVRMwoTSA --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher="slurm"

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=EDVRMwoTSA --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher="slurm"

## Testing Commands

### Single GPU Testing

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml

### Distributed Testing

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> ./scripts/dist_test.sh 8 options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml  --launcher pytorch

or

> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> ./scripts/dist_test.sh 4 options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml

### Slurm Testing

[Introduction to Slurm](https://slurm.schedmd.com/quickstart.html)

**1 GPU**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml --launcher="slurm"

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher="slurm"

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher="slurm"


---

# TrainTest_CN.md

# 训练和测试

[English](TrainTest.md) **|** [简体中文](TrainTest_CN.md)

所有的命令都在 `BasicSR` 的根目录下运行. <br>
一般来说, 训练和测试都有以下的步骤:

1. 准备数据. 参见 [DatasetPreparation_CN.md](DatasetPreparation_CN.md)
1. 修改Config文件. Config文件在 `options` 目录下面. 具体的Config配置含义, 可参考 [Config说明](Config_CN.md)
1. [Optional] 如果是测试或需要预训练, 则需下载预训练模型, 参见 [模型库](ModelZoo_CN.md)
1. 运行命令. 根据需要，使用 [训练命令](#训练命令) 或 [测试命令](#测试命令)

#### 目录

1. [训练命令](#训练命令)
    1. [单GPU训练](#单GPU训练)
    1. [分布式(多卡)训练](#分布式训练)
    1. [Slurm训练](#Slurm训练)
1. [测试命令](#测试命令)
    1. [单GPU测试](#单GPU测试)
    1. [分布式(多卡)测试](#分布式测试)
    1. [Slurm测试](#Slurm测试)

## 训练命令

### 单GPU训练

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml

### 分布式训练

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch

或者

> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> ./scripts/dist_train.sh 8 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher pytorch

或者

> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> ./scripts/dist_train.sh 4 options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml

### Slurm训练

[Slurm介绍](https://slurm.schedmd.com/quickstart.html)

**1 GPU**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=MSRResNetx4 --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/SRResNet_SRGAN/train_MSRResNet_x4.yml --launcher="slurm"

**4 GPUs**


> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=EDVRMwoTSA --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher="slurm"

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=EDVRMwoTSA --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/train.py -opt options/train/EDVR/train_EDVR_M_x4_SR_REDS_woTSA.yml --launcher="slurm"

## 测试命令

### 单GPU测试

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0 \\\
> python basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml

### 分布式测试

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher pytorch

或者

> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\\
> ./scripts/dist_test.sh 8 options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> python -m torch.distributed.launch --nproc_per_node=4 --master_port=4321 basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml  --launcher pytorch

或者

> CUDA_VISIBLE_DEVICES=0,1,2,3 \\\
> ./scripts/dist_test.sh 4 options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml

### Slurm测试

[Slurm介绍](https://slurm.schedmd.com/quickstart.html)

**1 GPU**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/SRResNet_SRGAN/test_MSRResNet_x4.yml --launcher="slurm"

**4 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:4 --ntasks=4 --ntasks-per-node=4 --cpus-per-task=4 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher="slurm"

**8 GPUs**

> PYTHONPATH="./:${PYTHONPATH}" \\\
> GLOG_vmodule=MemcachedClient=-1 \\\
> srun -p [partition] --mpi=pmi2 --job-name=test --gres=gpu:8 --ntasks=8 --ntasks-per-node=8 --cpus-per-task=6 --kill-on-bad-exit=1 \\\
> python -u basicsr/test.py -opt options/test/EDVR/test_EDVR_M_x4_SR_REDS.yml --launcher="slurm"


---

# history_updates.md

# History of New Features/Updates

:triangular_flag_on_post: **New Features/Updates**

- :white_check_mark: Oct 5, 2021. Add **ECBSR training and testing** codes: [ECBSR](https://github.com/xindongzhang/ECBSR).
  > ACMMM21: Edge-oriented Convolution Block for Real-time Super Resolution on Mobile Devices <br>
  > Xindong Zhang, Hui Zeng, Lei Zhang
- :white_check_mark: Sep 2, 2021. Add **SwinIR training and testing** codes: [SwinIR](https://github.com/JingyunLiang/SwinIR) by [Jingyun Liang](https://github.com/JingyunLiang). More details are in [HOWTOs.md](docs/HOWTOs.md#how-to-train-swinir-sr)
  > ICCVW21: SwinIR: Image Restoration Using Swin Transformer <br>
  > Jingyun Liang, Jiezhang Cao, Sun, Guolei Sun, Kai Zhang, Luc Van Gool and Radu Timofte
- :white_check_mark: Aug 5, 2021. Add NIQE, which produces the same results as MATLAB (both are 5.7296 for tests/data/baboon.png).
- :white_check_mark: July 31, 2021. Add **bi-directional video super-resolution** codes: [**BasicVSR** and IconVSR](https://arxiv.org/abs/2012.02181).
  > CVPR21: BasicVSR: The Search for Essential Components in Video Super-Resolution and Beyond <br>
  > Kelvin C.K., Xintao Wang, Ke Yu, Chao Dong, Chen Change Loy
- :white_check_mark: July 20, 2021. Add **dual-blind face restoration** codes: [HiFaceGAN](https://github.com/Lotayou/Face-Renovation) codes by [Lotayou](https://lotayou.github.io/).
- :white_check_mark: Nov 29, 2020. Add **ESRGAN** and **DFDNet** [colab demo](../colab)
- :white_check_mark: Sep 8, 2020. Add **blind face restoration** inference codes: [DFDNet](https://github.com/csxmli2016/DFDNet).
  > ECCV20: Blind Face Restoration via Deep Multi-scale Component Dictionaries <br>
  > Xiaoming Li, Chaofeng Chen, Shangchen Zhou, Xianhui Lin, Wangmeng Zuo and Lei Zhang
- :white_check_mark: Aug 27, 2020. Add **StyleGAN2 training and testing** codes: [StyleGAN2](https://github.com/rosinality/stylegan2-pytorch).
  > CVPR20: Analyzing and Improving the Image Quality of StyleGAN <br>
  > Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen and Timo Aila
- :white_check_mark: Aug 19, 2020. A **brand-new** BasicSR v1.0.0 online.


---


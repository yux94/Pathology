# NCRF-Baidu 

@(Code)[Camelyon16|Baidu|ResNet18+CRF|Classification]

**NCRF-Baidu**是由[Baidu](https://github.com/baidu-research/NCRF#ncrf)提出的在Camelyon16数据集上的分类算法，本人仅是根据其公开的源代码及[Camelyon16](https://camelyon16.grand-challenge.org/data/)数据集，复现了整个过程。该算法的主要特点是通过CRF学习到3x3个patch组作为整体网络的输入时之间的相互作用关系。

原作者的代码及其说明非常详细，非常感谢。

-------------------

[TOC]

## Data-Camelyon16

### 原数据
比赛官网：https://camelyon16.grand-challenge.org/data/
下载地址：https://camelyon17.grand-challenge.org/data/
####数据构成
```
Camelyon16
│   README.md
│   checksums.md5    
│	license.txt
└───training
│   │   lesion_annotations.zip (111 xml)
│   │───normal (159 tif)
│       │   normal_001.tif
│       │   normal_002.tif
│       │   ...
│   │───tumor (111 tif)
│       │   tumor_001.tif
│       │   tumor_002.tif
│       │   ...
└───testing
│   │   lesion_annotations.zip (48 xml)
│   │   reference.csv (129 csv)
│   │───images (129 tif)
│       │   test_001.tif
│       │   test_002.tif
│       │   ...
│   │───evaluation
│       │   evaluation_matlab.zip
│       │   evaluation_python.zip
│       │───evaluation_mask (111 png)
│       │   tumor_001_evaluation_mask.png
│       │   tumor_002_evaluation_mask.png
│       │   ...
```

###Whole slide images
| Item      |    Norm训练集 | Tumor训练集  | 测试集|
| :-------- | --------:| :--: |
| 数量  | 159 |  111   | 129|

###Patch

生成Patch的程序（仅CPU）
```
srun --partition=xxx python ./wsi/bin/patch_gen.py ./Data/WSI_TRAIN/ ./coords/normal_train.txt ./Data/PATCHES_NORMAL_TRAIN/

srun --partition=xxx python ./wsi/bin/patch_gen.py ./Data/WSI_VAL/ ./coords/normal_valid.txt ./Data/PATCHES_NORMAL_VALID/

srun --partition=xxx python ./wsi/bin/patch_gen.py ./Data/WSI_TRAIN/ ./coords/tumor_train.txt ./Data/PATCHES_TUMOR_TRAIN/

srun --partition=xxx python ./wsi/bin/patch_gen.py ./Data/WSI_VAL/ ./coords/tumor_valid.txt ./Data/PATCHES_TUMOR_VALID/
```
| Item      |    Norm训练集 |  Norm验证集 |Tumor训练集  |Tumor验证集 |
| :-------- | --------:| :--: |
| 数量  | 200000|  20000   | 200000|20000|


###标注

xml标注点坐标
```
python NCRF/wsi/bin/camelyon16xml2json.py Tumor_001.xml Tumor_001.json
```
####Category
	根据标注文件xml中的part of group划分

	1.Tumor:tumor/_0/_1
	2.Normal:_2

####Size

	1.Macro: metastases greater than 2.0 mm. 
	2.Micro: metastases greater than 0.2 mm or more than 200 cells, but smaller than 2.0 mm. 
	3.None


####Type

	1.idc
	2.Non-idc

##Training
```
srun -p xxx python -u ./wsi/bin/train.py ./configs/resnet18_base.json ./model/base/
srun -p xxx python -u ./wsi/bin/train.py ./configs/resnet18_crf.json ./model/crf/
srun -p xxx python -u ./wsi/bin/train.py ./configs/resnet18_crf_sn.json ./model/crf_sn/
```

##Testing
###Visualization of CRF w

```
python NCRF/wsi/bin/plot_W.py /PATH_TO_MODEL/best.ckpt
```
###Tissue mask
```
srun -p MIA python tissue_mask.py ../../WSI_PATH/ ../../MASK_PATH/crf/
```

###Probability map
批量，很耗时
```
srun --partition=xxx -w xxx python -u probs_map.py ../../WSI_PATH/new/ ../../CKPT_PATH/Camelyon16/crf/best.ckpt ../../configs/resnet18_crf.json ../../MASK_PATH/crf/ ../../PROBS_MAP_PATH/crf/
```

###Tumor localization
```
python NCRF/wsi/bin/nms.py /PROBS_MAP_PATH/cfr/ /COORD_PATH/crf/
```

###FROC evaluation
```
python NCRF/wsi/bin/Evaluation_FROC.py /TEST_MASK/ /COORD_PATH/
```

- [x] 训练
- [ ] mask tif
- [ ] ...

### 快捷键

帮助    `Ctrl + /`
同步文档    `Ctrl + S`
创建文档    `Ctrl + Alt + N`
最大化编辑器    `Ctrl + Enter`
预览文档 `Ctrl + Alt + Enter`
文档管理    `Ctrl + O`
系统菜单    `Ctrl + M` 

加粗    `Ctrl + B`
插入图片    `Ctrl + G`
插入链接    `Ctrl + L`
提升标题    `Ctrl + H`

## 引用

【1】Yi Li and Wei Ping. Cancer Metastasis Detection With Neural Conditional Random Field. Medical Imaging with Deep Learning (MIDL), 2018.
https://github.com/baidu-research/NCRF#ncrf
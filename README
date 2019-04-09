# 修改

## 将TensorFlow 官方的代码修改，目录结构的修改，net从slim移植过来

# faster rcnn demo
- backbone pnasnet
  - todo
    - FPN类似的【多级特征提取】添加到rpn侧，
    - roipooling -> RoIAlign，
    - batch normal -> group normal，或者自适应
    - nms -> softer nms
    - 图像增强，data augmentation, 如mixup：BEYOND EMPIRICAL RISK MINIMIZATION


```
train_main.sh
sample/config/pipeline.config

dataset_tools/create_pascal_tf_record.py
dataset_tools/create_coco_dataset.sh
dataset_tools/create_coco_tf_record.py

data/*.json
data/*.pbtxt

```

# 需要添加backbone 时，在 `nets` 下添加
# 需要切换到ssd或者maskrcnn 时，在 `nets` 、 `meta_architectures`、 `models` 下添加



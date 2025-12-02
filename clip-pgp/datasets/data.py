import numpy as np
from torchvision import transforms


class iData():
    train_trans = []
    test_trans = []
    common_trans = []
    class_order = None


class iCifar100(iData):
    scale = (0.05, 1.)
    ratio = (3./4., 4./3.)
    train_trans = [
        transforms.RandomResizedCrop(224, scale=scale, ratio=ratio),
        transforms.RandomHorizontalFlip(p=0.5)
    ]
    test_trans = [
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ]
    common_trans = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]

    def __init__(self):
        class_order = np.arange(100).tolist()
        self.class_order = class_order


class iImageNetR(iData):
    # ImageNet-R 有200个类别
    scale = (0.08, 1.0)  # ImageNet常用的scale范围
    ratio = (3./4., 4./3.)
    
    # 训练时的数据增强
    train_trans = [
        transforms.RandomResizedCrop(224, scale=scale, ratio=ratio),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    
    # 测试时的预处理
    test_trans = [
        transforms.Resize(256),  # 先缩放到256x256
        transforms.CenterCrop(224)  # 中心裁剪到224x224
    ]
    
    # 通用的预处理（使用ImageNet的标准归一化参数）
    common_trans = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    def __init__(self, class_order=None):
        """
        初始化ImageNet-R数据处理类
        
        Args:
            class_order: 类别顺序，如果为None则使用原始顺序
        """
        if class_order is None:
            # ImageNet-R有200个类别，使用原始顺序
            class_order = np.arange(200).tolist()
        self.class_order = class_order
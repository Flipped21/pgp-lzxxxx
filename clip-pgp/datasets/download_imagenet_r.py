import os
import requests
import tarfile
import shutil
import random
from pathlib import Path
from tqdm import tqdm
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

class ImageNetRDownloader:
    def __init__(self, root_dir="/mnt/data0/lzx/PGP/clip-pgp/datasets"):
        self.root_dir = Path(root_dir)
        self.dataset_dir = self.root_dir / "imagenet-r"
        
        # 多个镜像源（按顺序尝试）
        self.mirror_urls = [
            "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
            "https://huggingface.co/datasets/imagenet-r/resolve/main/imagenet-r.tar",
        ]
    
    def download_with_progress(self, url, filepath):
        """带进度条的下载函数"""
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as file, tqdm(
                desc=f"下载 {url.split('/')[-1]}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            return True
        except Exception as e:
            print(f"下载失败: {e}")
            return False
    
    def download_dataset(self):
        """下载数据集"""
        self.root_dir.mkdir(parents=True, exist_ok=True)
        tar_path = self.root_dir / "imagenet-r.tar"
        
        # 如果tar文件已存在，跳过下载
        if tar_path.exists():
            print("发现已存在的tar文件，跳过下载")
            return True
            
        # 尝试不同的镜像源
        for url in self.mirror_urls:
            print(f"尝试从镜像源下载: {url}")
            if self.download_with_progress(url, tar_path):
                print("下载成功!")
                return True
            else:
                print(f"镜像源 {url} 失败，尝试下一个...")
        
        print("所有镜像源都失败了，请检查网络连接或手动下载")
        return False
    
    def extract_dataset(self):
        """解压数据集"""
        tar_path = self.root_dir / "imagenet-r.tar"
        if not tar_path.exists():
            print("未找到tar文件，请先下载数据集")
            return False
            
        print("正在解压数据集...")
        try:
            # 先检查解压目录是否存在
            if self.dataset_dir.exists():
                print(f"发现已存在的数据集目录 {self.dataset_dir}，跳过解压")
                return True
                
            with tarfile.open(tar_path) as tar:
                # 先查看tar文件内容结构
                print("tar文件内容结构:")
                members = tar.getmembers()
                print(f"包含 {len(members)} 个文件/目录")
                if members:
                    print(f"第一个文件: {members[0].name}")
                
                # 解压
                tar.extractall(self.root_dir)
            
            print("解压完成!")
            
            # 检查解压后的目录结构
            print("解压后的文件结构:")
            for item in self.root_dir.iterdir():
                if item.is_dir():
                    print(f"目录: {item.name}")
                else:
                    print(f"文件: {item.name}")
            
            # 检查实际的ImageNet-R数据目录
            actual_data_dir = self.find_imagenet_r_dir()
            if actual_data_dir and actual_data_dir != self.dataset_dir:
                print(f"检测到数据实际在: {actual_data_dir}")
                print(f"移动到: {self.dataset_dir}")
                if actual_data_dir.exists():
                    shutil.move(str(actual_data_dir), str(self.dataset_dir))
            
            # 删除tar文件以节省空间（可选）
            # tar_path.unlink()
            return True
        except Exception as e:
            print(f"解压失败: {e}")
            return False
    
    def find_imagenet_r_dir(self):
        """查找实际的ImageNet-R数据目录"""
        possible_names = ["imagenet-r", "imagenet_r", "imagenet-r.tar", "imagenet_r.tar"]
        for name in possible_names:
            path = self.root_dir / name.replace(".tar", "")
            if path.exists() and path.is_dir():
                # 检查是否包含类别目录（以n开头的目录）
                class_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.startswith('n')]
                if class_dirs:
                    print(f"找到包含 {len(class_dirs)} 个类别的数据目录: {path}")
                    return path
        return None
    
    def get_class_mapping(self):
        """获取ImageNet类别名称映射"""
        # 首先尝试找到正确的数据目录
        data_dir = self.find_imagenet_r_dir()
        if not data_dir:
            print("未找到包含类别数据的目录")
            return {}
            
        class_samples = {}
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir() and class_dir.name.startswith('n'):
                images = list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                if images:
                    class_samples[class_dir.name] = images
        
        print(f"总共找到 {len(class_samples)} 个类别")
        return class_samples
    
    def split_dataset(self, test_size=0.2, random_state=42):
        """划分训练集和测试集"""
        # 使用实际的数据目录
        data_dir = self.find_imagenet_r_dir()
        if not data_dir:
            print("数据集目录不存在，请先下载和解压")
            return False
        
        print(f"使用数据目录: {data_dir}")
        
        # 创建训练集和测试集目录（放在imagenet-r目录内部）
        train_dir = data_dir / "train"
        test_dir = data_dir / "test"
        
        # 清理现有目录
        for dir_path in [train_dir, test_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True)
        
        # 获取所有类别的样本
        class_samples = self.get_class_mapping()
        
        if not class_samples:
            print("未找到任何类别数据")
            return False
            
        print("开始划分数据集...")
        total_train = 0
        total_test = 0
        
        for class_name, image_paths in tqdm(class_samples.items(), desc="处理类别"):
            if len(image_paths) < 2:  # 如果类别样本太少，全部放入训练集
                train_paths = image_paths
                test_paths = []
            else:
                # 划分训练集和测试集
                train_paths, test_paths = train_test_split(
                    image_paths, 
                    test_size=test_size, 
                    random_state=random_state,
                    shuffle=True
                )
            
            # 为每个类别创建子目录
            class_train_dir = train_dir / class_name
            class_test_dir = test_dir / class_name
            class_train_dir.mkdir(parents=True, exist_ok=True)
            class_test_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制文件到相应目录
            for src_path in train_paths:
                dst_path = class_train_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                total_train += 1
                
            for src_path in test_paths:
                dst_path = class_test_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                total_test += 1
        
        print(f"划分完成!")
        print(f"训练集: {total_train} 张图像")
        print(f"测试集: {total_test} 张图像")
        print(f"总类别数: {len(class_samples)}")
        
        return True
    
    def verify_split(self):
        """验证划分结果"""
        data_dir = self.find_imagenet_r_dir()
        if not data_dir:
            print("数据集目录不存在")
            return False
            
        train_dir = data_dir / "train"
        test_dir = data_dir / "test"
        
        if not train_dir.exists() or not test_dir.exists():
            print("训练集或测试集目录不存在")
            return False
        
        train_classes = [d for d in train_dir.iterdir() if d.is_dir()]
        test_classes = [d for d in test_dir.iterdir() if d.is_dir()]
        
        print(f"训练集类别数: {len(train_classes)}")
        print(f"测试集类别数: {len(test_classes)}")
        
        # 统计图像数量
        train_count = sum([len(list((train_dir/class_dir).iterdir())) 
                          for class_dir in train_classes])
        test_count = sum([len(list((test_dir/class_dir).iterdir())) 
                         for class_dir in test_classes])
        
        print(f"训练集图像数: {train_count}")
        print(f"测试集图像数: {test_count}")
        print(f"总计图像数: {train_count + test_count}")
        
        return True

def main():
    # 设置目标目录
    target_dir = "/mnt/data0/lzx/PGP/clip-pgp/datasets"
    
    # 创建下载器实例
    downloader = ImageNetRDownloader(target_dir)
    
    # 先检查当前目录状态
    print("当前目录状态:")
    if downloader.root_dir.exists():
        print(f"根目录存在: {downloader.root_dir}")
        for item in downloader.root_dir.iterdir():
            if item.is_dir():
                print(f"  目录: {item.name}")
            else:
                print(f"  文件: {item.name}")
    else:
        print(f"根目录不存在: {downloader.root_dir}")
    
    # 1. 下载数据集
    print("=" * 50)
    print("步骤1: 下载ImageNet-R数据集")
    print("=" * 50)
    if not downloader.download_dataset():
        return
    
    # 2. 解压数据集
    print("\n" + "=" * 50)
    print("步骤2: 解压数据集")
    print("=" * 50)
    if not downloader.extract_dataset():
        return
    
    # 3. 划分训练集和测试集
    print("\n" + "=" * 50)
    print("步骤3: 划分训练集(80%)和测试集(20%)")
    print("=" * 50)
    if not downloader.split_dataset(test_size=0.2):
        return
    
    # 4. 验证划分结果
    print("\n" + "=" * 50)
    print("步骤4: 验证划分结果")
    print("=" * 50)
    downloader.verify_split()
    
    print("\n" + "=" * 50)
    print("数据集准备完成!")
    data_dir = downloader.find_imagenet_r_dir()
    if data_dir:
        print(f"位置: {data_dir}")
        print("目录结构:")
        print(f"{data_dir}/")
        print("├── train/")
        print("│   ├── n01440764/")
        print("│   ├── n01443537/")
        print("│   └── ... (200个类别)")
        print("├── test/")
        print("│   ├── n01440764/")
        print("│   ├── n01443537/")
        print("│   └── ... (200个类别)")
        print("└── [原始类别目录]/")
        print("    ├── n01440764/")
        print("    ├── n01443537/")
        print("    └── ... (200个类别)")
    print("=" * 50)

if __name__ == "__main__":
    main()
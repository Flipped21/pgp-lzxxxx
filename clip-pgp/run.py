import torch
import random
import numpy as np
from datasets.data_manager import DataManager
from utils import factory
from utils.toolkit import count_parameters
import json
from datetime import datetime

def run(args):
    seed = args["seed"]
    _set_random(seed)
    _set_device(args)
    train_and_evaluate(args)


def train_and_evaluate(args):
    # 创建结果文件名（包含时间戳避免覆盖）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = f'/mnt/data0/lzx/PGP/clip-pgp/results/{args["dataset"]}/results_{timestamp}.txt'

    data_manager = DataManager(args["dataset"], args["shuffle"], args["seed"], args["init_class"], args["increment"], args)
    args["class_order"] = data_manager._class_order
    model = factory.get_model(args["model_name"], args)

    cnn_curve = {"top1": []}
    grouped_accuracies = []  # 存储每次任务的grouped准确率

    for task in range(data_manager.nb_tasks):
        print("All params: {}".format(count_parameters(model._network)))
        print("Trainable params: {}".format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)
        cnn_acc = model.eval_task()
        model.after_task()

        # 保存当前任务的grouped准确率
        grouped_accuracies.append(cnn_acc["grouped"])

        print("CNN: {}".format(cnn_acc["grouped"]))
        cnn_curve["top1"].append(cnn_acc["top1"])
        print("CNN top1 curve: {}".format(cnn_curve['top1']))
    

    # 实验结束后，将结果写入文件
    with open(result_file, 'w', encoding='utf-8') as f:
        # 写入参数信息
        f.write("=== 实验参数 ===\n")
        f.write(f"数据集：{args['dataset']}\n")
        f.write(f"初始 epoch 数：{args['init_epochs']}\n")
        f.write(f"后续 epoch 数：{args['epochs']}\n")
        f.write(f"bs:{args['batch_size']}\n")
        f.write(f"prompt_length:{args['prompt_length']}\n")
        f.write("\n\n")
        
        # 写入每次任务的grouped准确率
        f.write("=== 每次任务的Grouped准确率 ===\n")
        for i, acc in enumerate(grouped_accuracies):
            f.write(f"任务 {i+1}: {acc}\n")
        f.write("\n")
        
        # 写入最终的top1曲线
        f.write("=== 最终Top1准确率曲线 ===\n")
        f.write(f"{cnn_curve['top1']}\n")
        f.write("\n")
        
    print(f"实验结果已保存到: {result_file}")

def _set_device(args):
    device_type = args["device"]
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))
        gpus.append(device)
    args["device"] = gpus


def _set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    print("Seed Initialized!")

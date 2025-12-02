import torch
import numpy as np


def get_representation_matrix(data_loader, device):
    count = 1
    representation, rep_tasks = [], []
    for tasks, inputs, targets in data_loader:
        inputs = inputs.to(device, non_blocking=True)
        representation.append(inputs)
        rep_tasks.append(tasks)
        count += 1
        if count > 24:  # 只采样24个批次，所以每个任务采样总样本数为 24×32=768个
            representation = torch.cat(representation)   #  [768,3,224,224]
            rep_tasks = torch.cat(rep_tasks)    # [768]
            break
    return representation, rep_tasks


def update_memory(representation, threshold, feature=None):
    representation = np.matmul(representation, representation.T)  # [768,768]
    if feature is None:
        U, S, Vh = np.linalg.svd(representation, full_matrices=False)   # [768,768] [768,]降序的奇异值 [768,768]
        sval_total = (S ** 2).sum()
        sval_ratio = (S ** 2) / sval_total
        r = np.sum(np.cumsum(sval_ratio) < threshold)   # r=2
        feature = U[:, 0:r]   # [768,2]  V_10
    else:
        U1, S1, Vh1 = np.linalg.svd(representation, full_matrices=False)
        sval_total = (S1 ** 2).sum()
        # Projected Representation
        act_hat = representation - np.dot(np.dot(feature, feature.transpose()), representation)
        U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
        # criteria
        sval_hat = (S ** 2).sum()
        sval_ratio = (S ** 2) / sval_total
        accumulated_sval = (sval_total - sval_hat) / sval_total
        r = 0
        for ii in range(sval_ratio.shape[0]):
            if accumulated_sval < threshold:
                accumulated_sval += sval_ratio[ii]
                r += 1
            else:
                break
        if r == 0:
            return feature
        # update GPM
        U = np.hstack((feature, U[:, 0:r]))
        if U.shape[1] > U.shape[0]:
            feature = U[:, 0:U.shape[0]]
        else:
            feature = U
    print('-'*40)
    print('Gradient Constraints Summary', feature.shape)
    print('-'*40)

    return feature

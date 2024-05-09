"""
使用DDP训练得到的模型权重参数与单机训练得到的模型权重参数并无本质区别，因此在训练过程中可以随时进行DDP版本和单机版本的切换（通过保存和重新
加载.pth文件）
但唯一的小问题是两种方法得到的.pth文件中模型状态字典(model_state_dict)的键(key)名称有微小的差异：
DDP版本的键名称会有一个前缀: module.
"""
import torch


# 键名称差异展示
single_model_state_dict = torch.load("./checkpoint/single/epoch0.pth")
ddp_model_state_dict = torch.load("./checkpoint/DDP/epoch0.pth")

print("Single Model Keys:\n")
count = 0
for key,_ in single_model_state_dict.items():
    print(f"{key}")
    count += 1
    if count >= 5: # 只打印前5个key名称
        break

print("DDP Model Keys:\n")
count = 0
for key,_ in ddp_model_state_dict.items():
    print(f"{key}")
    count += 1
    if count >= 5:
        break


"""
因此，如果需要更改单机或DDP训练，只需要在加载.pth文件时修改一下key名称即可，例如：
"""

def Single2DDP(model_state_dict):
    new_model_state_dict = {}
    for key, value in model_state_dict.items():
        if key.startswith("module."):
            return model_state_dict
        else:
            # 添加前缀 "module." 到每个 key
            new_key = "module." + key
            new_model_state_dict[new_key] = value

    return new_model_state_dict
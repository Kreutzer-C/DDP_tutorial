"""
DDP 使用示例：多卡版本
"""
import os
import torch
import torchvision.transforms as T
from myDataSet import ODIR_DataSet
from torch.utils.data import DataLoader
from myModel import ConvNet
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# 多GPU计算结果聚合函数 (先略过，后面会解释)
def get_global(input, device):
    input_tensor = torch.tensor(input, device=device)
    dist.all_reduce(input_tensor, op=dist.ReduceOp.SUM) # 将多个GPU上的结果数值相加
    input_tensor /= dist.get_world_size() # 除以总GPU数量求平均
    return input_tensor.item()


# 传入参数
"""
首先DDP必须使用argparser传入一个重要参数: local_rank 表示参与多卡计算的GPU数量
"""
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int, default=-1, help='rank of distributed processes')
parser.add_argument('--batch_size', type=int, default=4) # 其他参数正常添加即可
args = parser.parse_args()


# 初始化DDP
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl', init_method='env://') # 初始化端口及连接方式，nccl是效率最高的选择，直接照做即可
device = torch.device("cuda:{}".format(int(local_rank)))


# 定义训练结果保存路径
"""
初始化DDP之后，所有后续代码操作都将由多个GPU并行执行，然而对于某些操作，例如：创建路径，输出结果，保存结果等，我们仅需要执行一次，
因此加入一行条件控制语句，用于判断当前GPU是否是主GPU(rank=0)，仅在主GPU中执行这些操作。
"""
if dist.get_rank() == 0: # 判断当前GPU是否是主GPU
    model_save_path = './checkpoint/DDP/'
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)


# 加载数据
## 定义数据增强方法
augmentation = T.Compose([
    T.RandomResizedCrop(224, scale=(0.2, 1.0)), # 裁剪到统一大小
    T.ToTensor()
])

## 定义DataSet, DataLoader (DDP版本)
"""
在进行数据加载时，我们必须为每个GPU指定加载不同的数据，以达成并行计算的目的，因此需要对DataLoader进行一些修改：
在DataLoader中需要指定sampler参数为多GPU分布式采样器DistributedSampler
"""
train_dataset = ODIR_DataSet('./data', transform=augmentation) # 与单机版本相同
train_sampler = DistributedSampler(train_dataset) # 多GPU分布式采样器DistributedSampler
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                          drop_last=True, num_workers=0, pin_memory=True) # 这里一定不要加shuffle=True，因为shuffle和sampler是互斥的


# 定义模型 (正常定义即可，与单机版本相同)
model = ConvNet(num_classes=2, dropout_prob=0.5).to(device)


# 定义损失函数及优化器 (正常定义即可，与单机版本相同)
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters())


# 初始化DDP模型
"""
对定义好的模型进行DDP初始化，这一步必须在定义完模型、损失函数和优化器之后执行
"""
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


"""
当涉及到从断点加载模型参数继续训练，或对预训练模型参数进行微调等需要load模型参数的操作时，一定要在初始化DDP模型之后进行
"""
# torch.load(...)


# 训练
for epoch in tqdm(range(0,10)):
    train_sampler.set_epoch(epoch) # DDP独特的地方，每个epoch开始时需要调用分布式采样器的set_epoch()方法，并将当前epoch传入
    # 剩余的训练代码与单机版本无异
    model.train()
    epoch_total_loss = 0
    for iteration, image in enumerate(train_loader):
        image = image.to(torch.float).to(device)
        label = torch.randint(0, 2, (args.batch_size,)).to(torch.float).to(device)

        optimizer.zero_grad()
        output = model(image)

        loss = criterion(output, label.unsqueeze(1))
        epoch_total_loss += loss.item()

        # 每个GPU会单独输入一部分数据(具体数量是batch_size/local_rank)，单独计算各自的loss，再进行后续的合并与反向传播等操作，
        # 为了直观展示DDP效果，这里分别print多个GPU各自计算出的loss
        print(f"[GPU {dist.get_rank()}] Loss:{loss.item()}")

        loss.backward()
        optimizer.step()

    # epoch_average_loss = epoch_total_loss/len(train_loader)
    # print(f"Epoch: {epoch} Loss: {epoch_average_loss}")
    # 这里是DDP另一个需要注意的点：上述两行代码会分别在每个GPU上计算其各自的epoch loss，但此时我们希望的是一个聚合后的Loss来反映
    # 多个GPU上的训练进程，因此需要对多个GPU上的Loss值进行求平均聚合，在测试过程中的某些评价指标计算也同理：
    epoch_average_loss = get_global(epoch_total_loss, device) / len(train_loader) # 调用get_global()函数聚合多个GPU上的数值
    # 此时再输出聚合后的epoch loss，别忘了只在主GPU上执行输出操作
    if dist.get_rank() == 0:
        print(f"===Epoch:{epoch} Loss:{epoch_average_loss}")

    # 保存模型权重参数，同理，只在主GPU上执行保存操作
    if dist.get_rank() == 0:
        torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch{epoch}.pth'))
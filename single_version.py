"""
DDP 使用示例：单卡版本（用于对比）
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

# 定义GPU设备（单机）
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
GPU_device = torch.cuda.get_device_properties(device)
print("\n>>>=====device:{}({},{}MB=====<<<)\n".format(device, GPU_device.name, GPU_device.total_memory / 1024 ** 2))


# 定义训练结果保存路径
model_save_path = './checkpoint/single/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

batch_size = 4


# 加载数据
## 定义数据增强方法
augmentation = T.Compose([
    T.RandomResizedCrop(224, scale=(0.2, 1.0)), # 裁剪到统一大小
    T.ToTensor()
])

## 定义DataSet, DataLoader
train_dataset = ODIR_DataSet('./data', transform=augmentation)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=True)

# 定义模型
model = ConvNet(num_classes=2, dropout_prob=0.5).to(device)

# 定义损失函数及优化器
criterion = nn.BCEWithLogitsLoss().to(device)
optimizer = optim.Adam(model.parameters())



# 训练
for epoch in tqdm(range(0,10)):
    model.train()
    epoch_total_loss = 0
    for iteration, image in enumerate(train_loader):
        image = image.to(torch.float).to(device)
        label = torch.randint(0, 2, (batch_size,)).to(torch.float).to(device)

        optimizer.zero_grad()
        output = model(image)

        loss = criterion(output, label.unsqueeze(1))
        epoch_total_loss += loss.item()
        loss.backward()
        optimizer.step()

    epoch_average_loss = epoch_total_loss/len(train_loader)
    print(f"Epoch: {epoch} Loss: {epoch_average_loss}")
    torch.save(model.state_dict(), os.path.join(model_save_path, f'epoch{epoch}.pth'))
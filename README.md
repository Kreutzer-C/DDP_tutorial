# DDP_tutorial

### **DistributedDataParallel (DDP) 多卡训练参考教程**

我们使用一个简单的数据集和一个基础的卷积神经网络分别实现了单卡和DDP多卡训练的程序，以便于对比学习

- 首先请查看并运行单机版本：

```sh
python single_version.py
```

这很简单，你将会在输出中看到每个epoch的loss值，并在./checkpoint/single/中看到保存的模型权重文件。

------

- 接下来请详细阅读DDP版本代码，然后通过以下命令执行：

```sh
CUDA_VISIBLE_DEVICE=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=2424 DDP_version.py --batch_size 4
```

其中各命令参数解析如下：

**CUDA_VISIBLE_DEVICE:** 指定多卡训练过程中要使用的GPU编号，一般双卡则指定为 CUDA_VISIBLE_DEVICE=0,1四卡则指定为 CUDA_VISIBLE_DEVICE=0,1,2,3 以此类推。（当然您也可以指定任意GPU执行，前提是您了解自己设备内各GPU的编号）

**nproc_per_node:** 指定参与多卡训练的GPU数量，与CUDA_VISIBLE_DEVICE对应

**master_port:** 通信端口号，可以任意指定一个没有被占用的端口（可选：2424, 3333，别的没试过:D)

指定上述特殊的命令参数后正常指定python程序入口文件及argparser参数即可，与单机版本无区别。

你将会在输出中看到代码注释中阐明的各项输出内容，并在./checkpoint/DDP/中看到保存的模型权重文件。

------

- 关于单机版本和DDP版本训练过程中的切换：

  通过保存和重新加载模型状态字典文件.pth实现

  请阅读并执行：

  ```shell
  python pth_key_operation.py
  ```

------

Good Luck ^_^

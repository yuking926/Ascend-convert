# Ascend-convert
将YOLO11模型转化为华为昇腾框架
为保障新手不知道如何将YOLO11 pt模型转化为昇腾框架om模型所写的教程
首先本教程版本为cann社区版8.2.RC1，请自行去下载适配与自身框架的安装包
Ascend-cann-toolkit_8.2.RC1_linux-x86_64.run

# YOLO11 模型 ATC 转换教程

本教程记录了如何在 **华为昇腾 (Ascend)** 平台上，将 **YOLO11 模型** 从 `.pt` 转换为 `.onnx`，再转换为 `.om` 格式，方便在 RKNN 或 Ascend 硬件环境上部署。

---

## 📦 1. 环境准备

### 1.1 初始化 Conda
确保终端前有 `(base)`，如果没有，执行：
```bash
conda init
```
然后关闭终端，重新打开。

### 1.2 创建 ATC 专用环境
```bash
conda create -n atc python=3.11.4
conda activate atc
```

### 1.3 安装依赖
在 `atc` 环境下安装如下依赖（部分包需固定版本）：
```bash
pip install attrs cython numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20.0 scipy==1.15.3 requests absl-py cloudpickle ml-dtypes tornado
```

### 1.4 检查版本
必须确保：
```text
numpy==1.24.0
scipy==1.15.3
```
否则 ATC 转换会报错。

---

## ⚙️ 2. 安装 Ascend CANN Toolkit

上传安装包并赋权：
```bash
chmod +x Ascend-cann-toolkit_8.2.RC1_linux-x86_64.run
```

执行安装：
```bash
./Ascend-cann-toolkit_8.2.RC1_linux-x86_64.run --install
```

输入 `y` 确认安装。

---

## 🔧 3. 配置环境变量

执行以下命令写入 `~/.bashrc` 并立即生效：
```bash
echo -e "\n# Set Ascend environment variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/devlib/:$LD_LIBRARY_PATH" >> ~/.bashrc && source ~/.bashrc
```

验证：
```bash
echo $LD_LIBRARY_PATH
```

---

## 📤 4. 模型转换流程

### 4.1 PyTorch → ONNX
在本地电脑执行，先将 `.pt` 转为 `.onnx`：
```python
from ultralytics import YOLO

model = YOLO("/home/yuking/Desktop/yolotoatc/weights/yolo11n.pt")  # 修改为自己的路径
model.export(format="onnx", opset=17)
```

执行完成后，会在同目录下生成 `yolo11n.onnx`。

### 4.2 上传 ONNX 模型到服务器
将 `yolo11n.onnx` 上传到服务器。

### 4.3 ONNX → OM (ATC)
在 `atc` 环境下运行：
```bash
atc --model=/路径/yolo11n.onnx --framework=5 --input_format=NCHW --input_shape="images:1,3,640,640" --output=yolo11n_fp16 --soc_version=Ascend310B4 --precision_mode=allow_mix_precision
```

- `--model=`：替换为你的 onnx 模型路径  
- `--output=`：输出文件名，可自定义  
- `--soc_version=`：根据实际设备修改（如 `Ascend310B4`）

### 4.4 转换成功标志
如果成功，会输出：
```
ATC run success, welcome to the next use.
```

此时会生成 `.om` 模型文件，可直接在昇腾平台上部署。

---

## ✅ 总结
流程简要：
1. Conda 配环境 → 安装依赖  
2. 安装 Ascend Toolkit → 配置环境变量  
3. `pt → onnx → om`  
4. 成功生成 `.om` 文件即可部署  

---

✍️ 作者：虞文燚  
📧 邮箱：yuking926@outlook.com  
🔗 项目地址：[GitHub Repo](https://github.com/yuking926/model_convert.git)

# 使用说明

## 安装

安装 Python 3.10 或更新版本，以及适用于本机 CUDA 的 PyTorch 2.4.1，然后在仓库根目录运行：

```bash
python -m pip install -e .
```

从 [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) 下载预训练权重 `open_clip_pytorch_model.bin`。

## 数据与运行

每名患者需准备包含配准后 `pet` 和 `ct` 图像的 HDF5 文件，以及包含 `case_id`、`path`、`label` 的五折训练和验证 CSV。ISUP 任务的标签 1 表示 GG 3–5；EPE 任务的标签 1 表示 EPE 阳性。

预处理、ISUP/EPE 训练及推理的完整命令见[运行指南](../README.md)。训练和推理参数可通过以下命令查看：

```bash
python train_isup.py --help
python train_epe.py --help
python -m fm_vhgf.predict --help
```


# 模型权重（不入 git）

检测器权重是 5 个 ONNX 文件，合计约 50 MB，放在仓库根目录的 `models/` 下。
它们不进版本库（见 `.gitignore`），所以克隆仓库后需要单独获取。

## 需要哪些

| 文件 | 用途 |
|---|---|
| `models/live_position_accumulated_v1.onnx` | 主检测器（`run_fast_structure.py`、`run_five_circle_v3.py` 默认） |
| `models/live_position_accumulated_v2_finetune.onnx` | 对照/实验 |
| `models/live_position_accumulated_v3_consistency.onnx` | 对照/实验 |
| `models/live_position_baseline.onnx` | 基线 |
| `models/small_target_native33_v1.onnx` | 小目标 |

同时存在的 `models/*.json` 是各权重的配置/描述文件，**这些要入库**。

## 怎么拿到

从原维护者处获取（网盘/直传），或从已有部署目录复制。放到 `models/` 下同名即可。

## 校验

拿到后核对哈希，确认与实验所用一致：

```powershell
Get-ChildItem models\*.onnx | Get-FileHash -Algorithm SHA256 | Format-Table Hash, Path
```

约定写入：任何人替换权重必须同时更新本文件里的哈希表，否则无法复现历史结果。

## 哈希

| 文件 | 大小 | SHA256 |
|---|---:|---|
| live_position_accumulated_v1.onnx | 9.97 MB | `ED8016494A798DF884573FFBBCEC0E69D1331886DA3222909E90135048C0F7A4` |
| live_position_accumulated_v2_finetune.onnx | 9.97 MB | `326F23E3D93E1649325E57C04A9D0D1914265B39DD7630FC335145E59FCA58C0` |
| live_position_accumulated_v3_consistency.onnx | 9.97 MB | `7E356B8B2E74EAF5F37EECDC9554071DAAA78816D71B66B94BB2F2BDF92C8395` |
| live_position_baseline.onnx | 9.97 MB | `E2531556AD0431D904FB0D1D432727A2E2DD9EAED9E098A6ED3D9E55F474DB0B` |
| small_target_native33_v1.onnx | 9.97 MB | `FC228A1B355A3B8396C05DF63F7EB8BEAF192BCA758BF5089898E16E893A4255` |

（2026-10-24 在本机实测，`Get-FileHash -Algorithm SHA256`。）

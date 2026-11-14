# 快速开始

完整RAW流程仍从源码目录运行，统一入口是根目录`forecast.py`。`pip install .`现在还提供0.3.1的独立[位置运动核心](POSITION_CORE.md)，但原控制台CLI仍是历史接口，不会安装当前完整research/CUDA链。

## 环境

目前只实际测试 Windows、1280×720 的 Metavision RAW 和 NVIDIA CUDA。SDK 与 GPU 可以使用同一 Python，也可以使用两个独立 Python。本机分别为 Python 3.11 与 3.13。

1. 按 Prophesee 的许可和安装说明独立安装 Metavision SDK，确认 SDK Python 能导入 `metavision_core`、`metavision_sdk_core`、`metavision_sdk_cv`。
2. SDK Python 安装 `numpy`、`scipy`、`opencv-python-headless`。`requirements-lock.txt` 记录本机已用版本，不保证其他 SDK ABI 兼容。
3. GPU Python 需要 CUDA 可用的 PyTorch、CUDAExecutionProvider 可用的 ONNX Runtime，以及 Ultralytics。本机实际版本见环境自检结果；硬件/CUDA相关包不提供跨机器通用锁定。
4. 准备兼容检测权重，见 [模型说明](MODEL_CARD.md)。仓库不附带 SDK、RAW、模型权重或这些软件的安装器。

## 配置与自检

复制 `configs/runtime.example.json` 为 `configs/local.json`，填写 SDK/GPU Python 的完整路径、权重路径和 SHA256。路径相对于配置文件所在目录；`python` 这样的命令名从 PATH 解析。

```powershell
python forecast.py doctor --config configs/local.json
```

自检会检查依赖导入、CUDA/ORT provider、权重哈希。它不能替代一次真实推理，也不证明达到某个实时频率。只看帮助不需要 SDK 或权重：

```powershell
python forecast.py --help
python forecast.py run --help
```

## 最小运行

```powershell
python forecast.py run --config configs/local.json `
  --raw 'E:\dataset\train\6\media\event.raw' `
  --output 'D:\drone forecast\output\my_first_demo' `
  --start 22 --duration 2 --profile research --preview
```

请改成自己的输入和新输出目录。`start` 与 `duration` 以秒为单位，需为20ms整数倍；输出目录须不存在。加 `--dry-run` 可只打印将执行的参数，不读取事件或启动推理。

| profile | 用途 |
|---|---|
| `live` | 独立RAW/SDK进程、低延迟位置跟踪、目标5Hz发布；与原研究流程分开，详见LIVE_MODE.md |
| `baseline` | 保留138几何＋140运动＋144阻尼默认算法 |
| `research` | 上述基础上启用153 LM、156并行、无机体时跳过前端、154未建模曲线回退；未升为默认算法 |
| `rotor-experimental` | 再启用149惯性桨盘/条件推力代理试验；不承诺改善预测 |

各配置均保留原始事件时间戳。5ms只影响预览画面的事件累积，检测仍使用20ms。`live`新增低频发布调度与最新观测机制，尚未保证连续有效5Hz；原三种研究配置的更新方式保持。

## 输出

- `predictions.jsonl`：每个20ms时刻的观测、模型状态、预测和运行计时。
- `protocol.json`：输入范围、参数、代码指纹、权重SHA256和运行环境路径。
- `completion.json`：完成状态、来源不变检查与汇总耗时。
- `predictions.mp4`：仅指定 `--preview` 时生成；50fps是录像时间轴，不能当作在线处理速度。
- `gpu_worker.log`：CUDA工作进程诊断。

消费预测时先检查行的 `valid`，再取 `preferred_forecast` 指定的 `predictions` 项。其他项包括运动学/几何对照；不应把尚未发布的潜在系数当作有效预测。位置单位为原生图像像素，预测时域0.1/0.2/0.4秒。见 [流程与字段](ARCHITECTURE.md)。

`live`另有 `tracking.jsonl`；其预测行t为实际发布时刻，`observation_t`为事件截止时刻，与旧研究流程t语义不同。见 [LIVE_MODE.md](LIVE_MODE.md)，不要直接跨口径评分。

## 评分

人工框的标签格式为 `records` 对象；每项含 `t`、`state: "present"` 和 `[x1,y1,x2,y2]` 的 `box`。仅用于事后评分，不进入推理。

```powershell
& '<SDK Python>' research/stage144/score_dense.py `
  --labels '<人工框labels.json>' --run 'demo=<predictions.jsonl>' `
  --use-emitted --published-only --compare-key curved_rollout `
  --out 'runs/scores.json'
```

该命令适合research配置；baseline配置不加 `--compare-key curved_rollout`。统一使用 `--published-only`（起点 `valid=true`）和 `--use-emitted` 实际输出预测评分。缺失对也必须报告。示例成绩与局限见 [结果说明](RESULTS.md)。

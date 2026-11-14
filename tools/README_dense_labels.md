# 连续密集标注工具（事件相机）

给 RAW-only 项目补**连续位置真值**用的。稀疏框（约 0.67 s 一个）只能判"有没有发布、位置粗不粗"，
要算 ADE/FDE 就得有连续框；这套工具就是干这个的。交互设计参考了另一个工程里既有的无人机标注 GUI
（左键画框、右键平移、滚轮缩放、状态选择、显示原检测框、Enter 保存并下一对、原子写盘可续标），
针对"连续帧"加了模型预填、上一帧幽灵框和播放。

**全程只用原生事件计数，不使用重建 G；不访问、不修改另一个工程。**

## 两个脚本

### 1. 渲染待标帧

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\tools\annotate_dense_prepare.py `
  --raw 'E:\dataset\train\6\media\event.raw' --start 20 --duration 15 --fps 20 `
  --output 'output\dense_labels_train6_20_35' `
  --model-jsonl 'output\stage10_continuous_identity_20261017\strict10.jsonl'
```

- 顺序扫一遍 RAW（不来回 seek），每 `1/fps` 秒一帧，渲染成 PNG（暗底，ON 偏暖、OFF 偏冷，逐帧 99 分位归一）。
- `--model-jsonl` 可选：把既有运行的检测框按最近时刻贴到帧上，供 GUI 叠加显示（**不重跑检测**）。
- 输出目录非空会拒绝覆盖；`manifest.json` 记录 RAW 指纹、区间、坐标定义（`native_event_pixels`）。

### 2. 标注

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\tools\annotate_dense_gui.py `
  --case 'output\dense_labels_train6_20_35'
```

| 操作 | 键 |
|---|---|
| 画框 | 左键拖拽（自动记为"有目标"）|
| 平移 / 缩放 / 复位 | 右键拖拽 / 滚轮 / `0` |
| 保存并下一帧 | `Enter` |
| 上一帧 / 下一帧 | `←` / `→`（另有 ±10 帧按钮）|
| 状态下拉 | 有目标 / 无目标 / 无法判断，或按 `1` `2` `3` |
| **采纳模型框**（最省力）| `A`，采纳后自动跳下一帧 |
| 清除本帧框 | `D` |
| 撤销 | `U` |
| 显示模型框 / 上一帧幽灵框 | `M` / `G` |
| 播放 / 暂停 | 空格 |
| 跳到未标注 | 工具栏按钮 |

设计上的两点：
- **留空直接下一帧＝保持"未标注"**，不会被当成无目标（密集标注里把"漏标"写成"无目标"会污染指标）。
- 每次改动立刻原子写 `labels.json`，随时关掉都能续；重开会自动跳到第一个未标注帧。

### 3. 打分（标完就能出 ADE/FDE）

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage15\label_score.py `
  --run output\stage10_continuous_identity_20261017 --method strict10 `
  --labels 'output\dense_labels_train6_20_35\labels.json'
```

参考轨迹取每个 `present` 框的中心，horizon 由相邻标注帧线性插值（要求被标注夹住，间隔 ≤0.08 s）；
ADE 在 0.05 s 网格上取平均，FDE 取端点。模型沿用阶段14 的五个因果滤波器。

## 标签格式

```json
{"schema_version": 1, "raw": "...", "start_s": 20.0, "fps": 20,
 "coordinate_system": "native_event_pixels", "size": [1280, 720],
 "records": {"42": {"index": 42, "t": 22.1, "state": "present",
                    "box": [x0, y0, x1, y1], "decision_source": "operator_draw|model_prefill_accepted",
                    "complete": true}}}
```

## 自测

`event_drone_forecast\tests\test_annotate_dense_gui.py` 用隐藏窗口跑一遍增删改、采纳模型框、
断点续标与撤销，不显示界面、不解码 RAW：

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\tests\test_annotate_dense_gui.py output\dense_labels_selftest
```

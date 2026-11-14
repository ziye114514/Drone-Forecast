# Event Drone Forecast

**事件相机无人机跟踪与二维短时轨迹预测。**

只用一路事件相机的 ON/OFF 原始事件流，不做灰度重建，不预知目标参数（质量、桨叶尺寸、推力系数、风速都不输入），实时输出当前的整机观测位置，以及未来 **0.1 / 0.2 / 0.4 秒**的图像平面像素位置。

当前状态：**Windows + NVIDIA CUDA 的研究源码版**，不是 clone 即用的正式发行包。检测权重需另行获取，RAW 数据不随仓库分发。

---

## 目录

- [它做什么](#它做什么)
- [核心约束](#核心约束)
- [快速开始](#快速开始)
- [仓库结构](#仓库结构)
- [预测管线](#预测管线)
- [当前结果](#当前结果)
- [能力边界](#能力边界)
- [文档导航](#文档导航)
- [协作与贡献](#协作与贡献)
- [许可证](#许可证)

---

## 它做什么

| 能力 | 说明 |
|---|---|
| 输入端 | Metavision `.raw` 事件流，纯 ON/OFF 极性特征，三通道 `(ON, OFF, max(ON,OFF))` |
| 检测 | 单阶段 ONNX 检测器定位整机包围框；另有一条不依赖 NN 的桨叶支撑图前端 |
| 状态估计 | 位置/速度的因果滤波，无未来帧泄漏 |
| 预测 | 运动学外推 + 阻尼收缩，输出 0.1/0.2/0.4 秒三个未来点 |
| 输出 | 当前观测、三个未来预测、诊断视频、逐帧因果记录（可离线复算） |

三条研究分支并存，互不覆盖：

1. **NN 主体线** — 检测器 + Kalman + 运动学外推，当前默认的稳定路线。
2. **纯桨叶线** — 不读整机 NN 权重，从原始事件里恢复桨盘残弧、五圆结构、侧视退化，配合可观测/不可观测二力模型。
3. **论文复现线** — `src/event_drone_forecast/` 下的可安装 Python 包，独立于根目录脚本。

---

## 核心约束

这些是整个项目的前提，不是可选项：

- **只有一路二维事件相机画面。** 没有第二视角，没有深度，没有 IMU 同步。
- **不做灰度重建。** 所有特征直接从事件极性构造。
- **目标参数不可预知。** 不预输入质量、桨叶尺寸、推力系数、风速。
- **没有飞控 / RPM / IMU 真值。** 无法补采，因此所有"物理量"只能是像素域的有效代理，不能声称为真实牛顿推力或真实转速。
- **在线可更新的只能是像素域有效参数。**
- **相机内参未标定。** 项目只要求二维像素位置，不需要相机标定。

---

## 快速开始

### 1. 环境

```powershell
# Python 3.11 + Metavision SDK + CUDA
D:\drone forecast\.venv-event311\Scripts\python.exe --version
```

依赖清单见 `requirements-lock.txt`（锁定）与 `requirements-counts.txt`（计数用）、`requirements-report.txt`（报告用）。

### 2. 配置

```powershell
Copy-Item configs\runtime.example.json configs\local.json
# 编辑 configs\local.json，填 Python 解释器路径与检测权重路径
```

配置里的相对路径以配置文件所在目录为基准。

### 3. 自检

```powershell
python core\forecast.py doctor --config configs\local.json
```

`--help` 不需要 SDK 或 CUDA，可以先跑这个确认装好了：

```powershell
python core\forecast.py --help
```

### 4. 跑一段

```powershell
python core\forecast.py run --config configs\local.json `
  --raw '<你的 event.raw>' --output '<新的输出目录>' `
  --start 22 --duration 2 --profile research --preview
```

### 5. 主预测入口

需要显式给权重路径和它的 SHA256（脚本会校验，防止拿错模型）：

```powershell
python core\run_optimized_forecast.py `
  --raw 'E:\dataset\train\6\media\event.raw' `
  --output 'D:\drone forecast\output\run_001' `
  --start 110 --duration 2 `
  --detector-weights 'models\live_position_accumulated_v1.onnx' `
  --detector-sha256 ed8016494a798df884573ffbbcec0e69d1331886da3222909e90135048c0f7a4
```

实测这一段：100 帧、中位处理耗时 **73.8 ms**，其中前端 42.5 ms、几何 46.4 ms、GPU 11.2 ms、模型 0.33 ms。阻尼外推默认开启（`damped_forecast=true`，`alpha=0.9`）。

### 6. 测试

```powershell
python -m unittest discover -s tests
```

25 个测试，约 1.4 秒跑完。注意：**环境里没有 pytest**，只能用 unittest。

---

## 仓库结构

```
event_drone_forecast/
├── core/                  # 主代码：340 个平铺 Python 模块
├── src/                   # 可安装包 event_drone_forecast（论文复现线）
├── tools/                 # 标注、评测、发布、诊断工具（900+ 文件）
├── research/              # 158 个阶段目录，每阶段独立可跑
├── native/                # CUDA / C++ 加速内核
├── models/                # 检测权重（ONNX）与物理参数（JSON）
├── configs/               # 运行时配置
├── docs/                  # 88 篇设计文档与阶段交接
├── tests/                 # unittest 测试集
├── release_templates/     # 发布模板
├── WORKLOG.md             # 完整实验记录（579 KB，150+ 阶段）
└── README.md              # 本文件
```

### 关于 `core/`

`core/` 里是 **340 个平铺模块**，没有子包嵌套。这是刻意的：

- 模块之间大量互相 `import`（如 `import live_decode`），平铺命名空间让这些 import 一行都不用改。
- 入口脚本用 `python core\xxx.py` 调用，`core/` 内部自行把仓库根和 `core/` 加进 `sys.path`。
- 340 个文件里 **76 个**有 `if __name__ == "__main__"`，是可直接运行的入口；其余是库模块，跑 `--help` 会失败，这是正常的。

常用入口：

| 脚本 | 用途 |
|---|---|
| `core/forecast.py` | 主 CLI（`doctor` / `run`），带 profile 切换 |
| `core/run_optimized_forecast.py` | 当前默认预测管线，几何 + 运动 + 阻尼 |
| `core/run_fast_structure.py` | `--rotor-only` 纯桨叶主线 |
| `core/run_rotor_only.py` | 桨叶专用入口 |
| `core/run_paper.py` | 论文复现线入口 |

### 关于 `research/`

158 个 `stage*` 目录，每个是**一次独立实验**：有假设、有对照、有结论，包括大量**失败结论**。这些没有删，是因为它们记录了哪些路走不通——比如四轮物理/几何分支、速度守门、自适应阻尼，全部实测无预测增益，留在那里防止后来人重走。

完整叙事在 `WORKLOG.md`。

---

## 预测管线

```
RAW 事件流
   ↓
ON/OFF 三通道特征 (ON, OFF, max)
   ↓
┌──────────────────┬──────────────────┐
│  NN 检测器        │  桨叶支撑图前端   │
│  (ONNX, 整机框)   │  (无 NN, 残弧)   │
└──────────────────┴──────────────────┘
   ↓
几何后端（五圆结构 / 侧视退化 / 共享平移）
   ↓
运动后端（可见位移 / 结构拟合）
   ↓
因果状态估计（位置、速度）
   ↓
运动学外推 + 阻尼收缩 k(h) = 1 − αh²
   ↓
0.1s / 0.2s / 0.4s 像素位置
```

### 阻尼外推

这是目前唯一被两份独立人工标注同时验证有效的修正。

原始外推在**沿航向**方向上系统性过冲，0.4 秒中位偏 **+24.5 px**，横向只有 +0.4 px。原因是匀速模型假设速度恒定，而真实无人机在持续减速。

修正方式是按预测时域收缩位移：

$$k(h) = 1 - \alpha h^2, \quad \alpha = 0.9$$

对应 `h = 0.1 / 0.2 / 0.4` 秒时 `k = 0.991 / 0.964 / 0.856`。

效果（0.4 秒平均误差）：

| 数据集 | 修正前 | 修正后 | 变化 |
|---|---|---|---|
| train6 侧视 (人工标注, n=66) | 55.09 px | **50.03 px** | −9.2% |
| train2 (跨录像留出, n=36) | 66.65 px | **59.30 px** | −11.0% |

单参数、无拟合、跨录像有效。已设为默认（`--no-damped-forecast` 可关）。

---

## 当前结果

### 主口径：人工标注真值

| 数据集 | 0.1s | 0.2s | 0.4s | 样本 | 耗时 |
|---|---|---|---|---|---|
| train6 侧视 110.0–119.5s | 9.89 px | 18.40 px | **50.03 px** | 66 | 60.9 ms |
| train2 40.70–46.00s | 9.46 px | 20.42 px | **59.30 px** | 36 | 56.0 ms |
| train6 清晰段（自洽） | 6.00 px | 10.97 px | 25.60 px | — | 101.7 ms |

前两行是**人工逐帧标注**的真值，第三行是模型自洽基准（用未来帧的观测中心当参考），口径不同，**不能混用**。

### 检测能力

以人工框为参考：

| 数据集 | 检出率 | 中心误差 0.1/0.2/0.4s | IoU | 宽比 / 高比 |
|---|---|---|---|---|
| train6 侧视 | 77.9% | 5.7 / 8.8 / 23.5 px | 0.67 | 1.10 / 1.33 |
| train2 | 61.0% | 6.0 / 9.9 / 13.7 px | 0.58 | 1.18 / 1.47 |

### 关于误差来源

早期把"相对二次拟合残差 36 px"当成观测噪声，据此判定瓶颈在观测——**这个结论是错的**。检测框中心实际只差约 6 px。那 36 px 主要是**真实轨迹的曲率**。

0.4 秒误差的大头来自目标在转弯，不是看得不准。这也是为什么纯运动学方法的天花板在这里：完美瞬时速度的匀速外推地板是 68.1 px，而实际误差 55.8 px——阻尼修正已经把这块吃掉了。

---

## 能力边界

明确没做到的事：

- **30 px / 0.4s 未稳定达成。** 侧视 50.03 px，train2 59.30 px。
- **持续 30 FPS 未达成。** 实测中位 73.8 ms/帧（约 13.5 FPS）。清晰段更慢，109 ms。
- **真实 RPM / 牛顿推力未辨识。** 只有像素域的有效代理，不声称物理量已标定。
- **新录像泛化未通过。** FRED 系列新录像上 0/25 有效输出。
- **侧视姿态未可靠恢复。** 侧视段逐桨频率质量 0.375、相位相干 0.117。
- **相机内参未标定。** 项目只要求二维像素位置，不要求标定。

已经被证伪、写进代码留作对照的路子（都在 `research/` 里有记录）：

| 方案 | 结果 |
|---|---|
| 速度守门（鲁棒斜率 + 分歧门） | 稀疏端点看着好 48.8→42.4，人工真值全输，且 train1 硬加速段 84→327 翻车 |
| 四路物理修正（线法向 / 线方向 / veto） | 全部比 baseline 差 0.4–2.7 px |
| 平移/摆动分离 | 侧视 50.03→50.78~70.64，全退化 |
| 自适应阻尼（按加速度分组） | 最优 α 无单调关系，不成立 |
| CTRV / IMM 运动模型 | 0.1–0.4s 无收益；拉到 1.0s 才反转 |
| 长记忆桨叶关联 | 按位置长记忆使错误率 1.99%→7.05% |

---

## 文档导航

**入门**
- [快速开始与评分](docs/QUICKSTART.md)
- [流程图、输出字段与代码导航](docs/ARCHITECTURE.md)
- [结果与当前缺陷](docs/RESULTS.md)
- [协作邀请指南](docs/COLLABORATION_INVITE.md)

**按主题**
- [低延迟位置通道](docs/LIVE_MODE.md)
- [持续五圆结构](docs/CONTINUOUS_FIVE_CIRCLE.md)
- [纯桨叶原生管线](docs/ROTOR_NATIVE_PIPELINE_20261101.md)
- [侧视响应](docs/SIDE_RESPONSE.md)
- [FRED 跨域迁移](docs/FRED_TRANSFER.md)
- [检测模型来源](docs/MODEL_CARD.md)
- [下一步路线](docs/ROADMAP.md)

**历史**
- [完整实验记录 WORKLOG.md](WORKLOG.md) — 579 KB，150+ 阶段
- [README 历史状态横幅存档](docs/README_HISTORY_BANNERS_20261114.md)
- [README 历史（截至阶段 156）](docs/README_HISTORY_THROUGH_STAGE156_20261025.md)
- [项目状态快照 2026-10-22](docs/PROJECT_STATUS_20261022.md)

---

## 协作与贡献

三条规矩，写在 `CONTRIBUTING.md` 里：

1. **不改冻结产物。** 已发布的模型、协议、发布包保持原样。
2. **不美化数字。** 退化的照实写，口径不同的不能混着比。
3. **权重与数据不入库。** ONNX 和 RAW 都在 `.gitignore` 里。

模型权重的 SHA256 清单见 `MODELS.md`。

研究记录只写仓库内那份 `WORKLOG.md`（曾经在上级目录有一份，已废弃）。

**想参与但没有数据？** 下面这两条不需要 RAW、权重或 GPU：

```powershell
python tools\demo_synthetic.py --output D:\tmp\demo   # 合成控制，验证几何/相位逻辑
python -m unittest discover -s tests                  # 25 个单测，约 1.4 秒
```

具体怎么找到人、怎么给新人派第一个任务，写在 [协作邀请指南](docs/COLLABORATION_INVITE.md) 里。

---

## 许可证

自有代码 MIT。第三方文件保留各自许可证，逐项见 `THIRD_PARTY_NOTICES.md` 与 `NOTICE.md`。

本项目是已有方法的工程整合与实验实现，**不是所参考论文的官方代码**。

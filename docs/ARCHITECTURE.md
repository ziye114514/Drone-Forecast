# 当前流程与代码导航

```text
Metavision RAW（原始微秒时间戳）
  ├─ 活动噪声过滤 → 原生ON/OFF帧 → 固定检测器 → 当前整机框
  └─ 当前ON/OFF计数 → count/occupancy两个几何前端
                         → 边界完整性检查 → 分来源机架模型
  整机框 + 当前合格几何运动 → 位置/速度估计 → 稳健加速度外推
                         → 阻尼预测 / 可选曲线回退 → valid发布门控
  可选桨盘试验 → 残弧/粘连/时序身份 → 条件周期 → 相对推力代理
```

## 两层检测

整机检测提供位置。逐桨几何用于结构和运动信息；看不清全部桨叶不应阻止位置分支输出。当前检测器对每时刻取最高置信度的整机框，并不是多目标身份管理系统。

机架采用四个共面等半径圆盘与中心的简化模型。部分桨盘推断和惯性续接不等于当前观测；不应把活动区域的视觉ID说成真实电机ID。

四桨学成结构与当前姿态有效是两个状态。`structure_supported` 是既有结构记忆，不是可靠的正/侧视分类器。当前 `research` 的曲线回退只用于尚未学成结构的状态；这仍有改进空间。

## 主要代码

| 文件 | 职责 |
|---|---|
| `forecast.py` / `forecast_runtime.py` | 统一启动、自检、可配环境与权重路径 |
| `run_optimized_forecast.py` | RAW循环、分支拼接、输出与子进程清理 |
| `research/stage47/worker47.py` / `stage45/graph45.py` | 共享内存当前帧、ORT CUDA图检测 |
| `research/stage138/tiles138.py` / `objective138.py` | 共享ON/OFF支持、几何拟合 |
| `research/stage129/boundary129.py` | 桨盘传感器边界检查 |
| `research/stage140/visible140.py` / `stage135/predictor135.py` | 几何运动与整机运动回退 |
| `research/stage153/lm153.py` | 既有目标函数LM求解，失败回原TRF |
| `research/stage154/turn154.py` | 运动学曲线外推，未使用转速或推力 |
| `research/stage156/parallel156.py` | 独立几何状态安全并行，合并后预测 |
| `research/stage146`—`stage149` | 阈值、弧拆分、时序关联、惯性/尺度与条件物理试验 |
| `research/stage144/score_dense.py` | 事后框中心预测评分 |

完整research模块仍有传递导入，历史源文件保持以便复跑。live的位置更新/外推已机械迁入可安装`src/event_drone_forecast/position`，只从包内固定FilterPy资源加载作者函数，不再读取历史研究目录；632条实际输入在源码、迁移wheel和独立安装目录中逐项保持。根目录live数学文件只是兼容导入。

旧`run.py`、`run_paper.py`和原控制台CLI仍为历史流程；新的position Python API不等于完整RAW/CUDA链已打包。详见[核心范围](POSITION_CORE.md)。

## 预测行

`t` 是录像秒时间戳；预测坐标为原生1280×720像素。`max_event_t` 的单位为微秒，不能晚于当前截止时刻。

- `body`：当前整机观测，`detected=false` 不能解读为当前看到了机体。
- `geometry` / `models`：当前几何证据与模型状态，二者区分观测和记忆。
- `valid`：是否准许发布当前预测。
- `coefficients`：内部状态/外推系数；存在不代表允许发布。
- `predictions`：按方法和时域保存的未来像素位置。
- `preferred_forecast`：本配置选择的方法键；仍需先检查 `valid`。
- `rotor_branch`：实验观测/条件频率与推力代理；默认不启用。

## 时间与实时目标

事件原始时间分辨率、20ms处理切片、几何刷新率、预测发布率与视频FPS是不同的概念。当前是顺序读取录像的离线处理器；并行只发生在同一时刻的两个独立几何臂，未实现跨帧堆积或实时丢帧调度。

新增独立 `live` 位置通道已实现目标5Hz发布、最新观测和延迟补偿。其SDK/RAW在独立进程，原生20ms持续处理，位置跟踪目标25Hz；不执行桨盘几何/推力预测。尚未达到连续有效5Hz或实时相机验收。原研究流程尚未加入低频几何调度，详见 [LIVE_MODE.md](LIVE_MODE.md)。

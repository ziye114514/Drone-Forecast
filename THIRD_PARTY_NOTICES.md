# 第三方来源与分发范围

本项目自有源代码沿用根目录MIT。第三方文件保留原许可证，不受本项目MIT重新授权。

## 当前运行链实际使用的随附源码

| 目录 | 来源/固定提交 | 随附许可与用途 |
|---|---|---|
| `external/filterpy_linear_upstream` | [FilterPy](https://github.com/rlabbe/filterpy), `3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33` | MIT；保留完整作者文件和LICENSE。107通过AST执行原predict/update和reshape_z函数，未改写其数值计算 |
| `external/filterpy_process_upstream` | 同一FilterPy提交 | MIT；未改动discretization.py与LICENSE。可选连续时间Q调用作者函数，跨录像收益混合，不是默认算法 |
| `external/circular_center_calibration_168b85c1` | [circular-center-calibration](https://github.com/xiahaa/circular-center-calibration), `168b85c1cb241937a0c3e563ef382c20ad0be456` | Apache-2.0；保留LICENSE、NOTICE、快照和已有Python源码。58导入作者2D函数；已标定圆心校正只在满足已知K条件时使用，不能冒称自采RAW有标定 |

上述目录中的 `UPSTREAM_SNAPSHOT.json` 记录文件来源与哈希。当前本地快照没有AAMED子模块；上游NOTICE提到的可选C++/AAMED并未随本项目候选包引入。

0.3.1可安装位置核心另将上述固定FilterPy线性/过程噪声快照原样放入`src/event_drone_forecast/_vendor`作为包内资源，逐份保留LICENSE与快照指纹，未改动作者文件。历史external目录仍用于旧研究流程；包内镜像用于脱离研究目录运行，第三方许可不变。

## 独立安装的运行依赖

Metavision SDK、NumPy、SciPy、OpenCV、PyTorch、ONNX Runtime、Ultralytics及其依赖由用户独立安装，遵循各自条款；源码包不附带这些软件、SDK、CUDA运行库或安装器。GPU检测确实使用Ultralytics预处理/NMS；不能写成“运行时不使用任何第三方研究或检测库”。

## 权重与数据

检测器来源和SHA256见 [模型说明](docs/MODEL_CARD.md)。当前候选包不含权重，不宣称上游权重可用本项目MIT重新分发。自采RAW、FRED下载数据和人工原始标注不放入源码包。

## 历史可选对照

为保留历史源码兼容，候选包还保留 `external/filterpy_imm_upstream`、`external/filterpy_mmae_upstream` 和 `external/evperiscope` 的已有小型源码快照，均保留原MIT LICENSE；它们不是当前推荐配置新增的算法贡献。它们的既有README/快照文件也一并保留。

独立下载的 [EEPPR](https://github.com/JackPieCZ/EEPPR) 是GPL-3.0项目；历史基准脚本引用用户单独下载的目录，未进入当前默认推理，也未复制进当前候选包。其他下载的整仓库不随本项目打包。

## 方法参考

- [Relative State Estimation using Event-Based Propeller Sensing](https://arxiv.org/abs/2604.18289)
- [Event-Only Drone Trajectory Forecasting with RPM-Modulated Kalman Filtering](https://arxiv.org/abs/2603.01997)
- [effqpdet](https://github.com/radimspetlik/effqpdet)
- [HelixTrack](https://github.com/radimspetlik/helixtrack)

本项目不是这些工作的官方代码，不以其论文成绩作为自己的复现成绩。历史方法记录保存在 `NOTICE.md`；当前分发以本文件和各第三方原始许可为准。

本机另有`research/helix_reference`适配器，直接包含用户独立下载的HelixTrack头文件进行编译执行。作者仓库文件未改动；本机未发现完整上游许可证文件，故作者代码、编译DLL与该本地适配目录都明确排除公开源码候选包。该实验不等于已获得重分发授权，也不是完整论文复现。

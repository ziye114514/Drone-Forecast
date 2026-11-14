# 88：固定 FRED 子集的因果事件输入对照

- [FRED 官方仓库](https://github.com/miccunifi/FRED)提供事件、位置框和划分。复用第7阶段固定提交 `2bf89c5376eda528431b62d6c60f2c13d8f95ab4` 的划分，并重新检查本地压缩包。旧版部分框缺 ID；不把不同框自动拼成单目标轨迹。多框同时间、越界、无效框、时间缺口必须明示。
- [Event-Only Drone Trajectory Forecasting with RPM-Modulated Kalman Filtering](https://arxiv.org/html/2603.01997v1)以过去标注框隔离预测层，并以事件周期调节过程噪声。借鉴其分离检测误差与预测误差的评测方法；本实验不是作者代码复现或官方完整成绩。
- [Residual Kalman Dynamics for Event-Based UAV Forecasting](https://arxiv.org/html/2609.00839v1)已经研究框历史/KF加事件加速度残差，并用无事件模型和去相关子集检查运动先验捷径。不能再把这一总体组合视为新颖。借鉴事件独立贡献的诊断要求；没有运行其作者代码。
- [FilterPy MMAE](https://filterpy.readthedocs.io/en/latest/kalman/MMAEFilterBank.html)：本轮直接复用项目已固定的作者滤波核心、77适配和84批量投影。几何、频谱、RLS也复用既有模块，不重写滤波器。

实验分两类：FRED用过去提供框作为位置/ROI输入，明确不是RAW-only部署；自采数据保持78的实际自动检测和事件观测，新增仅使用过去样本的打乱/延迟输入。两类不能混成同一精度数字。未来框只在预测存盘后的评分过程读取。

时间打乱使用因果过去样本库，至少落后0.2秒、最多2秒；不是整段循环移位或随机置换，后两者会把未来输入带回当前。恒定、零修正、延迟和打乱控制共享真实输入的发布资格时钟，保留同样输出覆盖。控制较差也不能删除。

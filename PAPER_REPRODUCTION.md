# v0.3：桨盘姿态—转速—推力—未来位置

本轮对照用户提供的 `D:/Downloads/2604.18289v1.pdf`，即 **Relative State Estimation using Event-Based Propeller Sensing**。这是按论文结构编写的独立实现，并增加二维在线适应和多步预测；不是作者代码，也未复现作者表格成绩。

你的判断成立：有当前状态、姿态和推力，就可以在动力学假设下积分得到未来位置。论文式 (9)–(10) 本身就在做下一步状态预测。更远的未来取决于尚未发生的电机控制，因此这里保持当前转速，并让角速度指数衰减，输出 0.1、0.2、0.4 秒的预测；不把未来转速偷偷输入模型。

## 运行与结果

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_paper.py' --raw 'E:\dataset\train\6\media\event.raw' --start 22 --duration 6 --output 'D:\drone forecast\output\my_paper_prediction' --preview
```

传感器输入只有 RAW。参数配置可选，不需要人工框、重建视频或未来轨迹。结果位于 `D:/drone forecast/output/paper_reproduction_v030`。完整评分与视频入口为该目录的 `查看论文复现实验.html`。新命令为 `run_paper.py`，保留的 `run.py` 仍是 v0.2 对照入口。

## 与原论文逐项对照

| 环节 | 原论文 | 本轮实现 | 尚有的差距 |
|---|---|---|---|
| 分批检测 | 10ms，STC 过滤，CC 或 HDBSCAN | 10ms，重复双极性事件、形态学、CC 轮廓 | 不是作者 STC，未实现 HDBSCAN；稀疏小桨盘有漏检 |
| 分桨身份 | Norfair 关联与短期预测 | 匈牙利关联、速度预测、过期和重新初始化 | 未调用 Norfair；相互靠近时仍可能换 ID |
| 转速 | 子象限计数、滑窗 FFT、KF | 160ms 窗、100us 分箱、正负极性 FFT、每桨 KF | 窗长和噪声等是公开文本不足处的工程选择 |
| 谐波 | 提取叶片通过率并换算转速 | 加入去平移后的转子平面相位一致性、候选半频/倍频 | 是受 HelixTrack 启发的简化实现，没有其 GN 同伦位姿跟踪器 |
| 桨盘姿态 | 最高事件数桨盘椭圆，经相机 K/P1E 反投影 | 名义分支选最高活动桨；增强分支融合一致桨盘；可选 K 的圆锥特征分解与畸变校正 | 不是完整 P1E；缺真实 K 时默认弱透视轴比，不能当作已验证的三维姿态 |
| 姿态传播 | 机体系左右电机差分、坐标变换、单位推力轴 KF | 三维单位轴 KF；图像内轴角速度、左右频率差经验增益 | 图像左右不是机体左右，默认不具备已标定的 yaw/电机对应；没有完成真实机体系 roll/pitch 力矩辨识 |
| 推力 | T=cf ΣΩi² | 逐桨频率平方和，相对参考推力；另有物理单位核 | 默认不知道 cf、质量、尺度；估计有效像素增益，不输出牛顿 |
| 位置 | 已知机体对角线、K、观测机位姿，世界系位置/速度 KF | 二维位置/速度 KF，以推力方向和强度作为加速度输入 | RAW 主入口没有真实米制位置或观测机位姿变换 |
| 向前预测 | 动力学单步状态预测 | 保持当前转速、衰减角速度的多步积分 | 0.4 秒内突然改变控制、阵风或相机运动仍可能造成误差 |
| 不理想几何 | 主要是四桨可见的俯视试验 | 残弧历史约束拟合、近侧视轴线、观测质量门控 | 短弧/直线不能唯一恢复全部姿态；漏桨后不能假装已测到四桨总推力 |
| 在线适应 | 原文未提供同样的未知参数自适应模块 | 延迟残差、受约束 RLS、残余加速度、因果模型融合 | 额外工程模块；还没有证明优于基线或能恢复真实风速/质量 |

`paper_physics.metric_acceleration` 和 `metric_forward` 实现物理单位下的 T/m·n+g 与三维积分，并有悬停/倾斜推力测试。**它们不等于真实 RAW 已完成三维米制复现**。真实 RAW 主入口调用的是像素域 `ProjectedThrustModel`。`PaperConfig` 中可选的质量/推力系数/对角线字段目前并不启用完整米制管线。

默认画面上方作为重力反方向、两叶桨、200px/s² 名义重力，都明文记录为假设；未伪造相机内参。检测目录中的事件→RGB 单应矩阵 H 不能替代相机 K。拍摄时相机移动会混入目标运动，目前不做完整相机自运动补偿。

## 二维模型与未知参数

像素域模型为 `a = κ·s·n_xy − g_px·image_up − diag(drag)·v + residual`，其中 `s = Σ f_i² / reference_sum`。名义模型只使用默认先验和首次参考平衡；自适应模型在过去预测到期后，根据新到的 RAW 检测调整有效增益与残差。四桨、几何、周期质量不足时冻结学习。

这能降低必须提前知道机型参数的依赖，但不能凭一条单目轨迹分别辨识质量 m、推力系数 cf 和尺度：加速度首先包含 cf/m，乘以相同倍数的 m、cf 会给出相同加速度。风、阻力、相机运动和投影变化也可能互相混淆。因此日志写“有效增益/残余加速度”，不写“已识别质量/风速”。

## 八个输出模型

`cv`、`ca`、`adaptive_history`、`adaptive_rotor`、`adaptive` 是同一检测输入上的对照；`paper_nominal` 为名义推力模型，`paper_adaptive` 为在线适应推力模型，`paper_fused` 为 CV/两物理模型的因果加权组合。

`physics_active=false` 时后三个输出可能使用 CV。评分必须同时查看“全部可预测样本”和“物理观测有效样本”，否则会把回退预测误算成物理预测成绩。四椭圆子集按预测发布时的观测选取，不能按未来误差挑片段，也不等同人工确认的四桨真值。

## 日志与验证

- `observations.jsonl`：原始检测桨盘 `detected_rotors`，带稳定 ID 的 `rotors`，四桨/四椭圆/四频率标记。真实频率质量、相位相干和谐波诊断以每桨内的 `frequency_diagnostics` 为准；顶层同名摘要中的相位字段在 v0.3 保留为空，不影响推理。
- `forecasts.jsonl`：八个已发布预测、推力轴、有效参数、学习次数、物理有效性及模型权重。参数单位是像素域。
- `predictions.csv`：各预测时刻/时长/模型的未来位置。不是观测到未来后回填的轨迹。
- `source_snapshot`、`experiment_protocol.json`：冻结推理代码、测试及指纹；旧版已归档，开发试跑不混入最终成绩。
- 四段完整 RAW 是以前看过的数据，属于工程回归，不能称新的未见测试集。稀疏人工框只评分实际标注时刻的未来端点，不插值制造密集真值；没有实飞 RPM/姿态真值。
- 原生 RAW 4 秒前缀与完整运行前缀逐行比较，用于检查因果一致性。单进程高事件率计时与并行批处理耗时分开。
- 25 项测试覆盖：动力学、标定圆反投影、已知事件频率、遮挡平移、稳定 ID、残弧、侧视轴线、未来不泄漏等。合成旋转边缘不是 ESIM/v2e，也没有模拟完整光度物理。
- `controlled_flight` 是已知姿态/频率观测下的滤波与积分实验，**不测试 RAW 识别**；不能代替实飞效果。

## 复跑本机实验

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\tools\paper_experiment.py' run
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\tools\paper_experiment.py' evaluate
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\tools\paper_experiment.py' prefix
```

完整运行已经存在则跳过；不覆盖同名运行。修改运行代码会使冻结指纹检查失败，应另开实验目录，不能替换旧成绩。`tools/build_paper_report.py` 生成最终报告、图与索引。

## 下一阶段的优先顺序

先用台架/公开带真值桨叶数据验证每桨频率、谐波和姿态；再在固定相机、四桨清晰的独立录制上测多时长预测和消融。收集真实 K 与同步状态仅用于验证/标定研究分支，部署入口仍可保留 RAW-only。随后处理几何退化时的状态切换、相位跟踪恢复和相机自运动，最后再评估哪些组合形成新的、有证据的贡献。不要仅凭“用了在线学习/椭圆补全”就宣称首次创新。

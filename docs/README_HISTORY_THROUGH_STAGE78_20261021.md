# Event Drone Forecast：只输入RAW的二维无人机预测工程

2026-10-21最新完成76–78：[状态隔离/未来修正与六段RAW视频](</D:/drone forecast/output/stage78_RAW_separated_forecast_20261021/第七十六至七十八阶段状态隔离与预测修正报告.html>)、[新入口说明](research/stage78/README.md)。旧融合误差拆解后，固定十路结构对照，全部49路和实际RAW前缀验证；2185父/4370原生窗口、1850视频帧。原侧视0.4s85.767→63.383px、3号143.039→97.443，但恒定输入多数近似或更好；额外三个固定录像未证实普遍收益，2号回历史反而差于CV、4号差于历史。新实验 `run_separated_forecast.py`，默认54仍保持，端到端仍非实时。下面旧状态均为历史。


2026-10-21最新完成68–75：[框CV与事件物理双分支报告/视频](</D:/drone forecast/output/stage75_RAW_parallel_forecast_20261021/第七十四至七十五阶段双分支融合报告.html>)、[RAW入口与复用说明](research/stage75/README.md)、[推力接线/时间扩展报告](</D:/drone forecast/output/stage73_temporal_extension_20261021/第六十八至七十三阶段事件推力预测报告.html>)。真实RAW验证39路预测及两独立前缀，1435父/2870原生窗口、1100视频帧；作者FilterPy IMM固定提交/MIT保留。清晰0.4s原CV24.292→融合14.746px，但侧视/3号仍未稳胜强历史，端到端不实时。最新实验入口 `run_hybrid_forecast.py`，默认54保持。以下旧“未接推力/最新62/暂停”为历史记录。


2026-10-21完成64–67：[运动补偿、投影方向预测与无桨心局部周期报告](</D:/drone forecast/output/stage67_multipeak_periodicity_20261021/第六十七阶段多峰周期与本轮预测总览.html>)。64真实RAW过去速度补偿及前缀通过；65缓存未来实验仅3号138.262→138.107px、侧视不变，没有明显增益。66/67无需椭圆即可提局部周期，但多峰规则在额外有色非周期控制中误接受31/40、照明全部通过，未进入推力学习。最新完整RAW入口仍62、默认仍54；以下历史记录保留。

2026-10-21完成63不依赖四桨锁定的整机轴观测诊断：[63报告与固定画面](</D:/drone forecast/output/stage63_body_axis_observation_20261021/第六十三阶段不依赖四桨的整机轴观测报告.html>)、[说明](research/stage63/README.md)。14秒RAW、700当前时刻，直接复用原6 PCA/LSD。侧视139/250方向候选没有补上62失效区间，3号全事件与双极分别补135/4次；尚未验证这些线是桨盘平面方向，未接入新的未来预测。下面62是最新完成的完整RAW预测实验。

2026-10-21完成62方向与转速准入分离：[62报告/完整视频](</D:/drone forecast/output/stage62_independent_axis_20261021/第六十二阶段方向与转速独立使用报告.html>)、[使用说明](research/stage62/README.md)。实际RAW1025父/2050子、740视频帧及两个独立RAW前缀通过，完整15预测与冻结因果回放逐项相同。侧视轴向0.4s误差122.894→113.917px，清晰15.736→14.004px，3号136.693→138.262px变差；纯历史侧视81.913px仍更好，不升级默认。新试验入口run_axis_forecast.py保留全部15专家，视频紫色显示轴向分支。没有提高四桨提取率或独立验证RPM/姿态。61完整两路也已完成：[稳定线段与配对正式结果](</D:/drone forecast/output/stage61_preellipse_lines_20261021/稳定线段与配对完整预测报告.html>)；无实质未来增益，配对不默认。以下61旧“尚未评测”为当时模块阶段历史记录，以上最新完整结果为准。

2026-10-21完成61当前测量修复小对照：[先判线再拟合椭圆/RNG控制](</D:/drone forecast/output/stage61_preellipse_lines_20261021/第六十一阶段先判线再拟合椭圆报告.html>)、[61说明](research/stage61/README.md)。60完整预测暴露当前OpenCV5对真实共线轮廓fitEllipse依赖RNG的尺度跳变；同一11px点集产生约9.86–13.23px长度。61将薄轮廓在椭圆拟合前交给已有线段测量，300不同RAW时刻、两种RNG历史共600窗口当前结果精确一致。61尚未完成未来位置评测，默认仍54；60旧完整预测停在严格回归失败、未评分，保留为失败尝试，不能宣称其融合已改善精度。

2026-10-21完成58投影圆心/共同圆锥诊断及59侧视线段小试验：[59报告与画面](</D:/drone forecast/output/stage59_side_lines_20261021/第五十九阶段侧视线段中点小试验.html>)、[58报告](</D:/drone forecast/output/stage58_projected_centers_20261021/第五十八阶段投影圆心与共同桨盘报告.html>)。中等倾角合成圆心有改善，极端侧视仍退化；450真实RAW时刻原57几何精确，单独提高事件密度阈值未稳定改善线段可用性。端点不完整或多个桨盘粘连时，线段中点不能视为真实桨心。正视椭圆/侧视线段应为共享运动状态的两类观测，尚未新增未来位置成绩，默认仍54。具体执行/交接见根目录最新状态及output/checkpoints/latest_handoff.json。

2026-10-21完成57[完整轮廓候选ROI与阈值诊断](</D:/drone forecast/output/stage57_candidate_roi_20261021/实验结论与密度阈值诊断.html>)：[使用/局限](</D:/drone forecast/event_drone_forecast/research/stage57/README.md>)。清晰几何和0.4s物理误差从55退化17.20px恢复原14.22px；侧视122.45px、3号136.69px仍无新收益。1715父/3430子时刻、独立RAW前缀和360行评分通过；额外400时刻精确旧几何的阈值探针确认硬裁使每帧阈值比值中位1.315。54默认保持。下一步核查共面桨盘圆锥约束/投影圆心，不继续重复框余量微调。

2026-10-21完成56[共同机架关联复用小对照](</D:/drone forecast/output/stage56_roi_layout_association_20261021/第五十六阶段ROI机架关联复用报告.html>)：2600缓存当前时刻和720独立前缀通过，全部新旧关联输出相同；ROI开启后清晰段原初始化合格时刻60→3，侧视最长连续仍不足。没有重算相同RAW频率或声称新预测收益。下一步57验证候选筛选ROI，保留全图阈值与完整轮廓；55完整预测结果见下。

2026-10-21完成55整机框引导桨盘对照：[结果图与视频](</D:/drone forecast/output/stage55_body_guided_rotors_20261021/桨盘提取对照结果与固定时刻画面.html>)、[55使用与局限](</D:/drone forecast/event_drone_forecast/research/stage55/README.md>)。1715父/3430桨叶时刻、真实RAW前缀与360行未来评分通过；0.4s物理侧视122.46→120.94px、清晰14.22→17.20px、3号不变，联合成功不变，尚无稳定提升。54完整基线保留。下一步检查共同机架关联与连续初始化，不把四候选数当作真实四桨准确率。长期目标active。

2026-10-21继续，已完成54完整物理部署。可用[run_deploy_forecast.py](</D:/drone forecast/event_drone_forecast/run_deploy_forecast.py>)保留全部15模型、单次RAW解码与独立桨叶进程；[结果图/校验说明](</D:/drone forecast/output/stage54_single_decode_deploy_20261021/部署结果图与前缀校验说明.html>)、[报告视频](</D:/drone forecast/output/stage54_single_decode_deploy_20261021/第五十四阶段单次解码完整物理部署报告.html>)及[最新交接](</D:/drone forecast/PROJECT_STATUS_20261021.md>)。预测精确53，尚未实时；单解码单步未稳定胜双解码，不能宣传固定加速。53完整研究对照与旧入口均保留。

2026-10-19按用户要求收尾暂停，已完成至48，无后台任务，不自动续跑。先读[最新交接状态](</D:/drone forecast/PROJECT_STATUS_20261019.md>)、[实验总报告](</D:/drone forecast/output/research_overview_20261018/index.html>)和[WORKLOG](</D:/drone forecast/WORKLOG.md>)。以下较早记录和旧默认保留。

保留的位置/历史专用入口是[run_shared_body_forecast_v2.py](</D:/drone forecast/event_drone_forecast/run_shared_body_forecast_v2.py>)：原生ON/OFF、固定适配权重、共享内存及常驻本地GPU；用法/视频见[47](</D:/drone forecast/output/stage47_shared_raw_bridge_20261019/第四十七阶段共享内存RAW入口报告.html>)。完整几何FP32单步74.52ms，仅整机位置FP16为31.05ms，两者仍超过20ms预算；后者关闭桨盘几何，不等于完整物理链。原v1失败保留，使用v2。

整机检测明显改善，未来位置仍有较大误差。46完整1550帧回归保持覆盖和联合成功；48在同一更准位置上加入桨叶物理，尚无稳定额外收益。侧视换槽20次重置导致参数实际更新0，是下一步应优先处理的瓶颈，阶段49未启动。没有实飞姿态、机械RPM或推力准确性证明。

旧完整桨叶物理RAW基线为[run_adaptive_physics.py](</D:/drone forecast/event_drone_forecast/run_adaptive_physics.py>)，见[30](</D:/drone forecast/output/stage30_raw_adaptive_physics_20261018/第三十阶段RAW自适应物理预测报告.html>)。48为缓存消融；53已把新整机位置和修复后的完整物理路径接到实际RAW并通过回归，但尚未显示稳定优于纯历史的精度。不用G，不访问/修改yolov10-main；旧run.py、42入口及冻结失败保留。

2026-10-17第九阶段：[整机状态保护与重新确认](research/stage9/README.md)。侧视邻域分支输出121→164/200，共同稀疏3标签当前误差21.47→15.42px；清晰段保持，FRED无收益，保留旧默认预测器。报告在`D:/drone forecast/output/stage9_state_recovery_20261017`；本轮仅验证检测，未证明未来轨迹提升。

2026-10-17第八阶段：[纯ON/OFF桨盘提取对照](research/stage8/README.md)。四组因果特征对比已完成：清晰盘有局部改善，侧视与FRED未获一致提升；保持原预测入口。输出`D:/drone forecast/output/stage8_onoff_rotors_20261017`。不使用G，不访问或修改yolov10-main；DSH→DeepSeek仅作已复核的文本审查。

2026-10-16第三阶段：[多桨法向与残弧置信度对照](research/stage3/README.md)。新增`run_normals.py`研究入口，共享位置与转速比较三种轴向/准入方案。已标定仿真有所改善，真实二维预测尚无稳定收益，未替换默认分支。报告位于`D:/drone forecast/output/stage3_normal_confidence_20261016/第三阶段实验报告.html`。

2026-10-16第二阶段：[模块验证与回退实验](research/stage2/README.md)。新增`run_guarded.py`实验入口，在原桨盘检测基础上增加受身份与位置约束的短时接续。无约束`run_hybrid.py`在自有录像出现背景误报，保留为失败对照，不作为默认升级。报告位于`D:/drone forecast/output/stage2_module_validation_20261016/第二阶段实验报告.html`。

2026-10-16新增独立研究实验：观测错误、在线学习污染与无同步日志时的公开数据验证。见 [实验说明](research/stage1/README.md)；报告位于 `D:/drone forecast/output/stage1_observation_errors_20261016/第一阶段实验报告.html`。包含负结果与覆盖率，不将仿真或伪标签当作实飞参数真值。

**新增 v0.3 论文结构分支**：逐桨稳定 ID、周期/相位一致性、频率平方和、推力轴 KF、受推力驱动的位置 KF，以及 0.1/0.2/0.4 秒前向积分。运行入口为 `run_paper.py`，详见 [论文复现范围与操作说明](PAPER_REPRODUCTION.md)、[相关论文和代码核查](RELATED_WORK_AUDIT.md)。最新结果为 `D:/drone forecast/output/paper_reproduction_v030/查看论文复现实验.html`。

v0.3 是对论文结构的独立实现与像素域适配；没有宣称已验证实飞姿态/转速或超过论文。以下保留 v0.2 的使用方式和既往结果；`run.py` 仍用于旧分支。

这是可运行、可评分、带基线和因果性测试的研究项目。输入一份 Prophesee RAW，自动检测和跟踪主要无人机的桨盘，提取椭圆及周期特征，在线学习图像中的有效运动关系，输出未来 **0.1、0.2、0.4 秒**的位置。

**当前输出是 1280×720 原生事件图像中的像素位置。** 不输入质量、机体大小、桨盘尺寸、推力系数或风速；也没有声称自动恢复这些量的真实物理值。没有相机标定和距离信息时，不输出伪造的米制三维轨迹。

## 已实现

1. SDK 流式 RAW 解码，按 20 ms 截止时刻发布结果；不读取后续事件。
2. 重复双极事件抑制背景，全图自动发现目标；支持完整椭圆、残弧与近线段三种观测。无需人工框或点击初始化。
3. 残弧拟合排除部分遮挡切边，记录可见弧范围、拟合残差、中心和轴向不确定性；近线段只提供轴方向。部分遮挡时用历史多桨布局推断整体位置，单桨关联有严格时间限制。
4. 100 ms 有界历史窗口中的分桨局部 FFT，附加半窗相干和谐波歧义诊断；频谱峰不自动等同于真实机械转速。
5. CV、CA 状态滤波；递归最小二乘在线学习有效图像加速度，带扰动残差和平滑、预热与失效门控。
6. 同步比较纯历史与加入桨盘轴向/相对频率平方的在线学习版本；几何/频率质量不足时冻结桨叶分支的学习并回退到历史模型。轴向特征对法向正负翻转不变。
7. 检测失效超时停止发布；重新发现目标后重建状态，保留同一目标序列中已学到的参数。
8. JSONL/CSV 数值输出、三模式彩色视频、独立人工框评测、新旧版本共同样本比较、真实 RAW 前缀因果性复核。输出预测不确定性诊断半径，但未校准概率覆盖率。

完整模式采用自适应组合，是为了在桨叶特征不可靠时仍能回退到运动模型。它不保证一定优于 CV/CA，更不保证未来任意飞控动作可预测。是否改善由留出录像实测决定。

## 一条命令使用

本机环境已在 `D:\drone forecast\.venv-event311` 建立。PowerShell 中：

```powershell
cd 'D:\drone forecast'
& '.\.venv-event311\Scripts\python.exe' '.\event_drone_forecast\run.py' --raw 'E:\dataset\train\6\media\event.raw' --output '.\output\my_prediction' --preview
```

只有 `--raw` 是传感器输入，`--output` 指定保存目录。默认从文件开头处理到结尾。快速查看：

```powershell
& '.\.venv-event311\Scripts\python.exe' '.\event_drone_forecast\run.py' --raw 'E:\dataset\train\6\media\event.raw' --start 108 --duration 5 --output '.\output\my_short_prediction' --preview
```

也可以使用包装脚本：

```powershell
.\event_drone_forecast\predict.ps1 -Raw 'E:\dataset\train\6\media\event.raw' -Start 108 -Duration 5
```

换 RAW 不需要提供人工框、RGB、离线重建或相邻 manifest。算法配置可选；没有任何目标质量/尺寸/风速参数。

## 查看本机整合结果

最终结果在 `D:\drone forecast\output\raw_only_project_integrated\查看整合结果.html`，包含新旧版相同时间段的视频、覆盖率、相同可用标签时刻的误差、失败样例和速度检查。

旧版结果保留在 `output/raw_only_project`；整合期间的 `raw_only_project_v020` 是已停止的中间试验，不能与最终成绩混合。最终配置和 7 个运行模块的 SHA 保存在 `experiment_protocol.json`。

本轮实测是功能覆盖提高、精度尚未提高：0.2 秒覆盖率从 41.4% 到 48.5%，双方共同可预测样本的平均误差从 45.72 到 53.97 像素。高事件率场景也仍不实时，因此保留旧版，不把 v0.2 当作精度升级。旧版源码在最终结果目录的 `baseline_v010/event_drone_forecast`，可用同一 Python 环境运行其 `run.py`。

## 一键复跑本机实验

```powershell
.\event_drone_forecast\reproduce.ps1
```

依次冻结/核对配置，处理四段 RAW，独立评分，生成视频，检查真实 RAW 前缀一致性，测量单进程耗时，生成报告并验证交付文件。默认跳过已完成且版本一致的推理，不覆盖中间运行；若运行代码已变化，会要求使用新的 `-Output` 目录。

此脚本面向当前数据目录；换数据做单份预测用 `predict.ps1`。只需运行预测时，不必安装 matplotlib；完整报告需它，本机已装到项目专用环境。

在新环境安装完整依赖：`python -m pip install -r event_drone_forecast/requirements-report.txt`。Metavision SDK 仍需由相机厂商安装。

## 环境

使用与已安装 Metavision SDK 匹配的 Python。本机 SDK 是 Python 3.11 二进制，Python 3.13 不能直接加载它。新环境可在工作区建立：

```powershell
py -3.11 -m venv --system-site-packages .venv-event311
& '.\.venv-event311\Scripts\python.exe' -m pip install -r '.\event_drone_forecast\requirements-lock.txt'
```

`--system-site-packages` 用于访问本机已安装 SDK。SDK 由相机厂商安装，不包含在此项目，也不由 requirements 安装。若使用其他 SDK 版本，应先验证 Python 版本兼容。

代码包亦可本地安装；不安装也能通过 `run.py` 运行。

## 输出文件

| 文件 | 内容 |
|---|---|
| `prediction.mp4` | 原生事件、已观测桨盘椭圆、历史轨迹、CV 与完整自适应预测 |
| `observations.jsonl` | 检测框、桨盘几何、叶片通过频率、测量置信度 |
| `forecasts.jsonl` | 每次发布时各模型的状态、有效加速度、更新次数和组合权重 |
| `predictions.csv` | 每个时刻、模型和预测时长的未来像素坐标 |
| `run_manifest.json` | RAW 路径、尺寸、唯一外部传感器输入、代码指纹和解释范围 |
| `config.json` | 完整运行参数 |
| `summary.json` | 覆盖、事件量、耗时、在线更新次数 |

逐时刻日志会保留没有目标或停止预测的情况。不会用未来真值回头修正已经发布的预测。

## 评测与复跑

`tools/prepare_evaluation.py` 单独读取现有人工事件框，已导出到 `output/raw_only_project/evaluation_data`。预测模块不导入该脚本。整合回归复用相同标签，不重新选择标注时刻。

```powershell
& '.\.venv-event311\Scripts\python.exe' '.\event_drone_forecast\tools\evaluate.py' --run '.\output\raw_only_project_integrated\runs\train2' --labels '.\output\raw_only_project\evaluation_data\train2.json'
```

评测只在真实人工标注时刻打分，不插值制造密集真值。对标注时刻 `t`，选择 `t-h` 之前最后一次已经发布的预测，外推到 `t`；实际时长为请求时长至请求时长加一个处理步长。报告未来端点误差、覆盖率和大误差尾部。**这不是有密集真值的轨迹 ADE。**

开发用 train/6；train/2、3、4 此前已经检查或评分，本轮称固定配置的工程回归，不称新留出测试。它们在原数据集中仍叫 train，不能宣称官方独立测试集或跨机型验证。新版另外报告与旧版都能预测的相同标签时刻，避免覆盖率变化混淆误差比较。

## 方法表达

观测量包括弱透视下桨盘法向投影的无向轴代理和相对频率平方。法向正负无法单靠椭圆确定，因此使用双角特征，近线段不输出完整倾角。在线模型学习的是

`image_acceleration ≈ learned_projection_features + residual_disturbance`

它不是已标定的 `a = (c_f/m) ΣΩ² n + g` 三维模型：图像加速度还受到深度变化、投影和相机运动影响。权重不是质量、真实推力系数或风速。椭圆倾角也不是完整三轴姿态，正负法向和偏航仍有歧义。

采用延迟预测残差更新时，只会用已经到达的 RAW 测量，随后发布的预测才使用新参数。组合器同样只比较已经结束的历史预测，不能读取未来 RPM、RGB 或人工真值。

## 测试

```powershell
& '.\.venv-event311\Scripts\python.exe' -m unittest discover -s '.\event_drone_forecast\tests' -v
```

覆盖恒速/加速度单位、改变未来输入不影响历史输出、学习延迟、目标丢失停止发布、空事件不误报、无框自动检测和已知周期信号，以及遮挡残弧、短弧不强行恢复完整形状、近圆位置保留、近线状四桨自动发现、法向正负不变和坏几何冻结学习。当前 16 项测试。

## 当前边界

- 单个主要目标；持续误关联、多个相似目标和旋翼完全不可见时不能保证识别。
- 只支持已安装 SDK 可读的 RAW，不是任意普通视频或其他相机封装。
- 不能由频谱峰直接证明真实 RPM；尚无当前数据中的逐桨测速真值。
- 残弧仍可能混入叶片自身边缘、遮挡或拖影；输出的补全不能证明真实几何正确。线状碎片提供的是候选轴方向，不保证它必然等于整个桨盘的轴。
- 局部协方差与预测半径未做概率校准；不宣称 95% 置信覆盖。
- 当前没有姿态、风扰、米制位置真值，不对这些量给虚假精度。
- 原生事件量与聚类负载会导致运行耗时波动。MP4 的 50 FPS 是播放时间轴，不是推理速度保证。
- 未知目标、未知尺寸、自由扰动同时存在时有可辨识性限制；看起来拟合的参数不一定是真实物理量。
- 因果预测器不使用原数据集离线 PCBS/重建；评测的现成人工框曾在重建图上绘制，存在边界与观测定义的不确定性。

论文与项目来源见 [NOTICE.md](NOTICE.md)。完整整合结果和失败分析由 `output/raw_only_project_integrated` 的报告给出，不能把本 README 当成结果承诺。

## 整机侧影研究入口（2026-10-17）

新增独立实验入口 `run_body_axis.py`：四点建立等效机架后，用整机事件支持与线段观察二维侧影方向，不要求持续四桨可辨。输入只需 RAW；初次有效四点出现前保持未建模。

运行方法和边界见 [research/stage6/README.md](research/stage6/README.md)。报告位于 `../output/stage6_body_axis_v2_20261017/第六阶段整机侧影实验报告.html`。该入口不替换原预测器；方向轴没有独立姿态真值，运动反推的力幅值不等于实际推力。


### 第七阶段：FRED定位/预测对照与RAW运动基线

见 `research/stage7/README.md`。结果：`D:\drone forecast\output\stage7_localization_budget_v3_20261017`。新增 `run_motion_baseline.py --raw ... --output ... --front activity`，使用实际OpenCV卡尔曼及既有修复版活动跟踪。11条RAW各20秒；8/9仅官方测试短子集。此入口是二维运动参考，非完整姿态/推力复现，未替换默认物理入口。

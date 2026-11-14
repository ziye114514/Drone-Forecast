# 177—180：孤立原轮廓、单桨相位入口与作者核心适配

用户：“好，现在回顾一下卡点，然后一个卡点一个卡点去攻克吧”。本轮沿176的输入卡点推进；没有NN替代、标签调参、扩大发布包或修改作者仓库。**当前生产默认仍176。研究目标未完成，无后台任务。**

先读[图文报告](../../output/rotor_inputs177_180_20261107/index.html)、[177完整预测对照](../../output/isolated177_20261107/integrated/scores.json)、[C帧局部相位](../../output/helix_bridge180_20261107/review_C_ring_phases.json)、[作者核心C短跟踪](../../output/helix_bridge180_20261107/native_pilot_C.json)。176的参考/二力语义见[前一交接](HANDOFF_20261107_FIXED_REFERENCE176.md)。

## 卡点顺序与本轮结论

1. **活动核不是物理桨盘边界。** 已补孤立轮廓重拟合入口；粘连双弧147/174仍在。43.6s右侧10.1px活动核恢复为25.0px原生轮廓，另43.4s左侧18.5→27.1px。当前只恢复几何候选，不增加桨数或周期票。
2. **轮廓亮不等于可测转速。** 三个坏时刻每个轮廓内部只有0—2个当前严格时序通过像素；旧转速区域入口需8个。平移补偿后最多0—3个，不能靠它解决。不是证明桨叶无周期，而是现有逐像素检验器/初始化没有取得足够合格证据。
3. **单桨相位不应等待完整五圆。** 已加单个非扁椭圆的局部空间相位初始化，前半窗口选频率/旋向，后半验证；C帧候选3合格，约263.58Hz事件相位。不依赖四个频率或完整模板；但没有机械RPM真值。
4. **持续性仍卡住。** 本机HelixTrack作者核心已通过独立C接口编译、调用；C同旋向跟踪约258—265Hz，但后续三包预更新相位一致性只有.25—.41，未持续达到现有.5门槛。不足以送入已知推力。
5. **推力尺度/未知力、0.4s轨迹与RAW30FPS还没解决。** 不把混合频率变化代理说成可测推力，不凭一个C帧宣称预测有收益。下一步先做当前逐桨几何更新/惯性复锁与局部相位连续资格，不扫k/tau/标签。

## 177：补小核外的孤立原轮廓

文件 `isolated_rim_visibility.py`；`FastStructure.isolated_rim_refit=False` 默认关闭，控制保持176。

- 复用`curvature_visibility.masks/contours`和147/171的`fit_arc_exact`，最多3个父轮廓；仅一个当前合格核、最多1处凹点的父轮廓可重拟合。
- 现核填充若已解释≥60%父轮廓，不重做；新拟合需原点残差、弧覆盖/条件数/中心不确定性、角度不确定性≤15°、非扁率>.16、原像素占据≥.60、扩大后解释更多原像素且包含现核；ROI/传感器裁边拒绝。
- 多核父轮廓留给双弧拆分，无核轮廓不自行变成桨叶；这次仍拒绝44.0s右侧无合格核的弱轮廓。
- 新center_sigma/协方差/normal同步重算；活动核中心/尺寸来源另存；不变更时间、原像素和频率投票来源。`mode=arc/inferred_shape=True`，须后续几何资格，不是完整物理桨盘真值。

实际200包GPU缓存前端（side133/train2 67），三臂共享同一GPU结果；标签文件审计钩子禁止推理中读，输出后统一固定时刻0.4s评分：

|路线|侧视均误差px，17/19|另一段，7/7|
|---|---:|---:|
|176控制|38.616605|87.080605|
|仅孤立轮廓177|38.616605|87.115445|
|174双弧＋177|39.283364|87.117098|

侧视0次孤立重拟合，另一段10次；侧视3个当前布局/2次176更新，另一段仍0布局/0更新。有效预测115/54不变；控制复原176位置差0。没有因为轮廓更好看就提升几何权力。三桨三个位置/尺寸/扁率依然不足以确认共同平面；旧43.6三个约10px核的几何度量能偶然通过，不能因此称为三个真实桨盘。

实际快照/输出：`output/isolated177_20261107/integrated`，`protocol.json`、`completion.json`、`scores.json`、`three_audit.json`、`source_snapshot/`。纯机制图和JSON在该177目录上层。

## 178/179：过去速度对齐及实际周期票数

复用`native_event_alignment.past_warp`（项目21原实现），不新写对齐算法、不得使用窗口开始之后的速度。原生未对齐计数与174保存20ms计数最大差0。

- 三时刻43.4/43.6/44.0：完整窗口开始前的原生位移速度，不用标签/未来/优化速度；原始EventCD缓存，不重建G。
- 只对齐几何：候选数2/3/2→2/2/2，没有解锁三桨。
- 再将对齐事件送入完全相同CUDA周期检验器：各父轮廓内部严格周期票从 `[1,0] / [2,0,0] / [1,0]` 到 `[1,0] / [0,0,0] / [3,1]`。不可把普通占据/活动像素当作新增周期像素。
- 保存`output/alignment178_20261107/probe.json`、`period_seeds.json`、`aligned_period179.json`、源图/周期图NPZ。
- 不启用对齐为默认，不做新RAW FPS或独立RPM精度声明。这仅排除了“简单平移补偿足以解决此卡点”。

## 180：单环相位资格，复用作者核心

### 自动当前窗口局部初始化

`observed_ring_phase.ring_phase_prior(events,ellipse,cutoff_s)`：

- 单个当前非扁椭圆，.2—1.5径向门，沿用SharedPhaseFFT的`exp(2j*theta)`、环权重；明确两叶/弱透视测量假设。
- 频率网格直接用`predictive_harmonics.BASE_FREQUENCIES`120—960Hz步15。前半窗口发现频率和旋向；用SharedPhaseFFT同类有界log功率抛物线细化频率，只用发现半段。
- 细化是必要的相位量化修正：15Hz网格误差可使后半段积累超过1rad，不能用放宽相位门槛替代。频率及旋向随后冻结，验证不参与选择。
- 训练/验证浓度≥.5，验证旋向功率对比≥.5（3:1），衔接余弦≥.5，至少9扇区/每半≥32事件，阈值沿用空间相位既有约定，没有按轨迹标签搜参数。
- 几何同窗共享，邻近事件不独立；不称统计概率界、真实RPM、真实推力。
- 独立生成恒角速195Hz事件：194.994Hz，训练/验证约.999；同样空间点云打乱时间则拒绝。追加未来事件结果不变，改变验证段不改变发现频率。
- train2 43.6孤立轮廓仍拒绝：候选约−537Hz，验证浓度.141、衔接.104。
- C112.02四个当前候选，只有候选3通过：263.5826Hz，前半.660左右/后半.648/衔接.638/12扇区。其他三拒绝。此例不是实时全段逐桨召回或RPM精度。

### 作者核心本地编译与适配

`D:\drone forecast\helixtrack-main` 作者仓库未改；没有复制作者头文件进工程/发布包。`native/helixtrack_bridge.cpp`用include直接调用作者`HelixTrackTracker`，C ABI向Python批量输入x/y/t/p、获取频率/角加速/投影。SDK读取器/GT/YAML/HDF5/GUI均不进入桥接。

- 编译 `tools/build_helix_bridge180.ps1`，使用VS18 Community MSVC、`D:\Prophesee\third_party`已有Eigen/OpenCV；生成仅在`output/helix_bridge180_20261107`。
- 不装依赖，不改系统PATH/作者库/两处yolov10-main/RAW/发布包；编译器仅有Eigen字符代码页warning，无编译/链接错误。
- `helixtrack_adapter.HelixCore`要求显式正事件相位先验和当前非扁椭圆；0/NaN先验拒绝，不回退作者GT。重复/乱序/未来事件拒绝；所有输出`qualified_for_force=False`，需外部资格与持续性，不是直接力传感器。
- 椭圆仅确定锥线，不唯一确定材料点相位。默认弱透视轴白化再交作者核心，固定16px归一化尺度仅单位约定。`author-conic`仅匹配锥线的投影参数控制可能静态锁至近0Hz，负结果存`native_pilot_author_conic.json`，不能当真实转速。
- 独立**正交投影恒角速**合成200Hz（不是用适配器H自造同模型数据）：200.0021Hz，投影轮廓误差约3.3e-7px；`checks_initial_core_geometry.json`是早期仅ABI/同投影自洽检查，不应代替这一独立生成检查。
- 作者每2000个有效批次事件允许求一次姿态；两种旋向/phase-only和joint作为诊断对照，不作为生产四支并行。使用默认GN步长，不称完整官方config/标定条件复现。

### 两个真实短跟踪

- train2：自动新轮廓，但先验来自既有混合180Hz；弱透视后输出虽158—196Hz，下一包预更新相位浓度仅.008—.052。**数字像转速仍然拒绝**。
- C：先验由当前窗口局部相位263.58Hz自动给出；同旋向约258—265Hz，反旋向降向103—168Hz。0.1s后续三个包在更新之前核验，同旋向浓度仅.25—.41，未稳定过.5。不能说持续测出真实RPM/可测推力。
- C各源cutoff112.02/112.053333/112.086666/112.119999，当前原生body取不晚于cutoff的现有输出（约20ms旧），速度源严格早于完整窗口开始。当前body用于测量归一化，不是用未来标签预测轨迹；输出记录native_body_source_t/velocity_source_t。初始过去窗口重放用于warm，不把warm残差当未来预测。
- 单桨桥接每包仅事件过程计时，未包括读RAW/GPU/所有桨提案/相位初始化/渲染/轨迹。不可换算30FPS。原始输出elapsed_ms全部保留。

## 下一步唯一优先项

**把当前单桨几何/局部相位做成真正随时间更新的观察器，接到live候选流，先让C的合格初始化跨包保持资格。**

1. 当前pilot的椭圆来自保存的一个当前帧，随后仅原生body平移＋作者局部批次姿态，并未将每包新的原弧/当前椭圆送回；这正是下一处具体工程缺口。不要把单帧初始化再写成永久成功。
2. 复用现有5圆关联/短惯性规则给可见候选局部track id；它们不是机头/真实电机标签。允许复锁；当前观测过期就降权，不补隐藏几何/RPM。单桨相位资格不等待完整四桨；侧线仍走侧线规则，环相位对扁线不适用。
3. 自动初始化只接受当前局部相位资格；弱频率代理不可冒充已知力。下一包预更新核验失败时收回可测贡献，隐藏/失效贡献归未知项，保持净加速度连续。
4. 同一实际源流只做一次有/无本贡献对照，再看0.4s；资格还不连续前不扫k/tau/反复评分。FPS最后对完整RAW入口，不能用局部C++循环代替。

## 文件、重放与检查

新文件：`isolated_rim_visibility.py`、`observed_ring_phase.py`、`helixtrack_adapter.py`、`native/helixtrack_bridge.cpp`；工具177—180见报告delivery_snapshot。`fast_structure_front.py`增加默认False孤立轮廓可选入口和诊断；`run_rotor_only.SOURCE_FILES`加isolated模块；生产预测/二力/NN/发布包未替换。

在项目根目录使用SDK Python：

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'event_drone_forecast/tools/check_isolated_rims177.py'
& 'D:\drone forecast\event_drone_forecast\tools\build_helix_bridge180.ps1'
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'event_drone_forecast/tools/check_helix_bridge180.py'
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'event_drone_forecast/tools/check_ring_prior180.py'
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'event_drone_forecast/tools/review_ring_prior180.py'
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'event_drone_forecast/tools/pilot_author_phase180.py' C
```

177实际GPU试验脚本`tools/live_rims177.py`的OUT使用exist_ok=False，重跑应另设新目录，不覆盖旧快照。新报告`output/rotor_inputs177_180_20261107/index.html`及最终delivery_snapshot不同于177实际启动快照，区分记录。

来源：[HelixTrack作者](https://github.com/radimspetlik/helixtrack)、[AAMED作者](https://github.com/Li-Zhaoxi/AAMED)、[Gallego CVPR2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gallego_A_Unifying_Contrast_CVPR_2018_paper.pdf)。作者授权/许可需核对后才能把含作者实现的DLL纳入GitHub；本轮只本地研究，未动发布树。无新盲测，无同步RPM/飞控真值，不再要求用户补采日志。

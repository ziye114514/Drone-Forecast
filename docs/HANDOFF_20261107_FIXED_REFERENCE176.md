# 176：固定图像输出参考，布局中心不再直接替换轨迹

用户最新：“那就继续修吧，一个一个问题解决”。本轮按174交接只处理位置参考及几何权力，不扫检测阈值、推力系数或tau。主线仍纯ON/OFF桨叶、五圆/可见性、可见贡献＋未知项单状态；NN基线/发布包保留。

先读[图文报告](../../output/layout176_20261107/index.html)、[全部数值](../../output/layout176_20261107/final_summary.json)、[公共入口检查](../../output/layout176_20261107/checks.json)。当前默认是176固定参考，不是173，也不默认启用174原弧扩大。

## 已改什么

### 175：模板的body锚点必须同步

旧前端在FastStructure.step后把body.center换成几何中心，但跨帧弧引导、刚体core.anchor、Continuous.body及pending确认仍保存旧body值，下一帧可能多加参考差。FastStructure.rebase_body_reference只移动这些body锚点，不移动图像桨心、beta、张量、协方差。模拟旧31.62px幽灵位移→0，真实平移保留。实际单独175侧视38.7329→38.7434px，另一段不变；不能报成准确率进步。

### 176：p、g和b分开

- 输出p是与初始原生支撑对齐的图像参考，不声明真实COM。当前桨盘布局中心g单独进入观测，定义g=p+b；b为未知图像参考偏置，未读标签、未加固定20px。
- 第一次g仅初始化b=g-p及它与p的负交叉协方差，不改p/v/d/k，禁止首次布局绝对换轨迹点。之后g通过H=[I_pos,I_bias]约束同一状态。
- 新LayoutReferenceObserver继承已验证ShortGapObserver，加2个偏置状态，保留7维运动/力、2维前位置clone，共11维。原生位移与布局使用对半信息权重，避免盲目独立堆叠同一事件来源；协方差仍工程代理，不称统计校准。
- 原生支撑部分出画→更完整支撑的初始化修正：p/前位置clone加δ，b减δ，保持p+b及图像桨心不动；同步模板body参考。可见/未知力分工不变，不补隐藏RPM。
- 只接受当前four/three/relock_pending且joint_center_is_current_measurement的布局；轴线、两点、coasting先验不能成为完整中心。布局资格有效期沿用.4s，NIS沿用9.21034，位置协方差底限沿用6px，未搜参数。
- 布局观测当前仍需有效原生位置/相对位移；尚不单独救全部native丢测，173短缺测处理保留。常量b是图像参考定义/假设，不是已识别机体与真实质量重心偏移。
- JSON与时钟publisher均标出output_reference和非COM/非布局点含义；g仍在structure/motion中单列。没有多一条生产预测轨迹。

## 当前默认与旧控制

`run_rotor_only.py`及`run_fast_structure.py --rotor-only`在`--rotor-relative-observation`且当前visibility路线下默认`reference_mode=fixed`、gap=short。默认174的`reconsider_small_cores=False`保持。

- `--rotor-reference-mode legacy --rotor-gap-prediction short`：173控制，旧绝对几何中心替换。
- `--rotor-reference-mode fixed`：当前176。
- gap legacy/coherent/short均支持固定参考；对应LegacyGapLayoutReferenceObserver/CoherentLayoutReferenceObserver/LayoutReferenceObserver。显式fixed要求joint relative，错误组合在启动GPU前拒绝。
- `--rotor-legacy-visibility`未指定reference时仍默认legacy，保完整历史控制。旧NN入口仍显式旧visibility，发布目录/ZIP不动。
- 旧170/173/175实验驱动已显式选自己的旧reference/gap，避免默认更新悄悄改控制。当前176实际驱动也已整理为公共工厂，试验启动快照仍保留手工组装的等价配置。

## 真实结果与尾部代价

两段原生EventCD缓存运行实际GPU周期前端/几何/状态；不是新RAW解码FPS。推理禁读labels.json，输出落盘后评分，所有录像仍历史开发。源时刻固定0.4s：

|片段|173控制|176默认|可评分/计划|
|---|---:|---:|---:|
|侧视|38.7329|38.6166|17/19|
|另一录像|87.0806|87.0806|7/7|

**收益很小且不逐点稳定**：侧视P95 68.8076→69.4512，最大69.9755→72.4939；112.2、112.8、113.0等退化，112.4/112.6改善，全部保留。默认采用参考语义修复，不声称稳定高精度。侧视115、另一段54有效预测不变，控制p/v/a复原173差0。

侧视仅3次当前布局，1次偏置初始化、2次后续更新；第一布局造成的绝对参考更换1→0。另一段67包0次合格布局/0更新，因此本轮不可能降低其主要变向误差；43.6/44.0仍140.5/193.7px。

同输入来源对照（固定参考、相同候选资格/协方差）：实际布局38.6166px，用原生位移替代布局数值38.7518px；均2次更新。差约.135px，只有一个片段两次相关更新，不能称显著普适/创新成功。控制是离线验证，未加进生产多分支。

174新原弧＋固定参考：侧视39.2834px，来源替代控制38.7738px；另一段87.0823px，不优于默认。继续保留候选API，不能因为图里椭圆变大就扩大位置权力。

## 检查与快照

- 无布局时复原旧相对观测器；首次布局不改p/v/a；全布局加[70,-110]任意参考偏移，输出运动差约2e-12；参考修正后p+b、v/力保持；PSD数值检查。
- 前端mock验证g和p分开，24px参考差不替换轨迹。隐藏贡献转入d不改净运动及偏置。
- 公共工厂对真实固定输入重放p/v/a差0；模式工厂、旧visibility默认控制、两个CLI help通过。
- 实际：`output/reference175_20261107/integrated`、`output/layout176_20261107/integrated`；来源对照`.../source_control`；机制JSON与图在176根目录。
- 175最初检查试图deepcopy含SDK EventCDBuffer的前端失败，只改测试为复制数值量，未改算法门槛；initial_test_note保留。
- 实际176开始快照/结束守卫通过；随后只整理公共工厂/分派、旧试验显式选项、输出参考元数据及publisher标签。实际算法手工组装与当前公共工厂重放相同，但不称所有源字节相同；当前交付快照另存delivery_snapshot。
- 备份`backup/20261107_pre_reference175`、`backup/20261107_pre_bias176`。无依赖安装、训练、作者代码复制、上传、原始RAW修改或代理；两处yolov10-main/权重/发布包不动。没有后台作业，没有创建/暂停目标，研究未冻结。30px/持续RAW30Hz未完成。

## 下一步只攻一个输入卡点

参考维护已修，先不要重新扫滤波器、参考偏移、tau或这些旧标签。另一录像没有合格布局，三小活动核不等于三真实桨盘，重点是：

1. 在已有174原弧候选基础上，检查孤立小核的外围原弧，而不只处理两个粘连桨；先给每个候选正确的原弧/局部核心角色和不确定性。
2. 三桨只在几何关系或近期模板充分支持时初始化/续接；保留多义性，不补第四个“实测”或RPM。两点/侧线继续不造完整当前中心。
3. 需要周期/姿态耦合时，先复用本机已有作者实现，解决自动初始化与事件格式，不再重复写同类相位跟踪。
4. 用一次机制测试＋一次有控制的小片段检查新输入是否真正进入176状态，再讨论30px目标。不能继续用无当前几何的滤波变体代替用户的核心。

## 已找到本机作者代码，后续不用再找路径

`D:\drone forecast\helixtrack-main`已存在，未改。已读README、recording_setup.cpp、main.cpp相关初始化和config_tracker.yaml。

- `recording_setup.cpp:139`查相邻`tracker_initial_positions/*.csv`，读取圆心、旋向、扁率、半径、psi；不能把这种提供的初始化当我们自动检出成功。
- `main.cpp:219`附近：tracker_init_rpm>0用配置先验，否则回退ground_truth_rpm.at_us(start_us)。当前yaml的tracker_init_rpm为0；接入RAW-only必须显式用正先验/过去事件估计，不能使用GT回退。
- SDK/CMake/Eigen/OpenCV/HDF5/yaml-cpp等条件与许可仍需核对后才编译或纳入发布。未复制作者代码到工程/ZIP；先做独立接口/初始化适配。
- 正式[HelixTrack仓库](https://github.com/radimspetlik/helixtrack)、[CoHelix共享几何论文](https://arxiv.org/html/2610.31384v1)与[stochastic cloning原始记录](https://authors.library.caltech.edu/records/enrkd-zam24)供复查。新参考偏置维护不是新的滤波理论，论文RPM锁定率不能直接比本项目轨迹px。

<!-- collaboration523-current:start -->
> 当前协作版本（2026-11-13）：517纯桨叶研究线，原102开发点实际RAW均29.997218px；FRED11新录像0/25有效输出，泛化未通过，持续30FPS未通过。[最新交接](docs/HANDOFF_20261113_COLLABORATION.md)及[独立源码包](../output/collaboration_20261113/event-drone-forecast-collab-20261113.zip)。下方为历史。
<!-- collaboration523-current:end -->

<!-- target30p5-current:start -->
> **当前研究（2026-11-12）**：同原102起点0.4s均值31.955→**31.471px**，中位25.379；30.5目标未达。当前实验入口`run_rotor_local_contrast_candidate.py`＋原`models/native_response_plane_v1.json`。局部事件能量归一化改善桨叶位移匹配，纯ON/OFF＋五圆＋可测代理/未知合力主线保留，无NN/G；不是已测准真实推力。4/6开发源改善、2/6回退，新录像盲测仍保留，FPS后置。见[报告](../output/target30p5_20261112/index.html)与[交接](docs/HANDOFF_20261112_TARGET30P5.md)。以下为历史。
<!-- target30p5-current:end -->

<!-- target31-current:start -->
> **当前研究（2026-11-11）**：同原102起点0.4s均值 **32.600→31.955px**，中位26.596；31目标未达到。推荐`run_rotor_joint_only_candidate.py`，纯ON/OFF桨叶＋五圆＋可测/未知力，不用NN/G。收益主要来自位移和不确定度校正，不称推力已辨识。六源均已暴露开发，新录像未验证，FPS后置。见[报告](../output/target31_20261111/index.html)与[当前交接](docs/HANDOFF_20261111_TARGET31.md)。以下为历史。
<!-- target31-current:end -->

<!-- accuracy453-current:start -->
> **当前研究状态（2026-11-10）**：`run_rotor_accuracy_candidate.py`为原生桨叶＋五圆＋可测/未知力主线，无机体NN/G。同实际网格原102起点完整447 **33.361→33.199px**；中位回退，4/6改善，30px未达，新录像未验证，FPS后置。RAW4共181包复现差0。当前逐桨输入稀少，推力增益在线更新仍0，不称推力创新已成功。局部几何454退化未接默认；原入口和公开NN发布包保留。先读[权威交接](docs/HANDOFF_20261110_ACCURACY453.md)与[图文报告](../output/accuracy_progress453_20261110/index.html)，下面为历史。
<!-- accuracy453-current:end -->

<!-- projection440-latest:start -->
> **当前原生桨叶候选（2026-11-10）：** `run_rotor_projection_candidate.py`补共享帧噪声与椭圆/侧视法线表示切换，仍为五圆＋可测/未知力主线，无机体NN/G。同原102起点34.607→**34.295px**；实际六段RAW同101起点公平旧/新34.162→**33.396px**，原102少1输出，不能混淆分母。11/4有回退，30px尚未达，新录像未验证，FPS后置。跨帧提早出生未稳定改善精度，保留负例未推广。原429/432入口和公开NN包保留。见[新报告](../output/projection_accuracy_20261110/index.html)与[权威交接](docs/HANDOFF_20261110_CONDITIONAL_RESPONSE.md)。以下为历史。
<!-- projection440-latest:end -->

<!-- observation430-latest:start -->
> **当前原生桨叶精度候选（2026-11-10）：** 同6段、同102原起点0.4秒平均误差36.463→**34.607px**，30内51→56。新增已有桨叶身份的当前位移续接、弱观测传播一致性、弱支撑参考点资格分工；不更新隐藏RPM/可测推力。4号原RAW181包零跳包复现差0px。11号有退步，4号参考资格小修相对426有回退，逐段全部公开；无按录像选择。主线仍为ON/OFF桨叶、五圆可见性与可测/未知力，无机体NN/G。**30px未达，新录像未验证，先精度后FPS，P95仅附带记录。** 新入口`run_rotor_observation_candidate.py`，旧入口/公开NN包保留。见[当前报告](../output/observation_accuracy_20261110/index.html)与[权威交接](docs/HANDOFF_20261110_CONDITIONAL_RESPONSE.md)。以下阶段为历史。
<!-- observation430-latest:end -->

> 2026-11-10 当前原生桨叶候选：同6段102固定起点，0.4秒均值43.32→36.46px，P95 102.27→75.74px，35px内49→59、30px内45→51。所有6段排除拟合，单一186教师模型，保留五圆/可见性/二力，无机体NN。整体35px与完整RAW持续30FPS尚未达；弱轮廓参考且6段均暴露开发。默认`models/native_response_plane_v1.json`；[当前交接](docs/HANDOFF_20261110_CONDITIONAL_RESPONSE.md)、[运行说明](docs/NATIVE_RESPONSE_CANDIDATE.md)、[全部起点图文](../output/precision_update_20261110/index.html)。较402均值低0.46px，但35内63→59，不能称全面更稳。下方均为历史批次。

<!-- precision365-current:start -->
> **2026-11-09 当前334–365已收尾，30px／持续30FPS仍未达成。** 主线为原生ON/OFF桨叶、五圆／可见性、可观测＋未知二力，无NN。新增统一可运行精度候选 `run_rotor_force_candidate.py`，旧入口不改：六段102同起点0.4s均值43.764→43.317px，但15/11/4有退化，不能说稳定泛化成功。当前周期续接可选，4号共同10点62.736→55.938px但11退化，默认不叠加。已修出画支撑循环移位；原生推力变化率仍无有效FDE收益，积分幅值组合也未胜候选。先读[权威交接](docs/HANDOFF_20261109_PRECISION359.md)和[全起点图文](../output/precision359_20261109/index.html)。以下均为历史记录；无后台任务，没有额度耗尽的工具证据。
<!-- precision365-current:end -->

# Event Drone Forecast

<!-- inverse333-current:start -->
> **2026-11-09 精度更新：可选未知法向响应校准已接RAW。** 新参数 `--rotor-inverse-normal-response`，默认关，无整机NN，当前p/v/a不改。13/15/16/11固定0.4秒源时间误差37.52→36.22、48.48→44.83、44.85→41.10、39.63→37.18px；3/4无收益，不称泛化成功。四源/相位增益仍弱，30px/持续RAW30Hz未达。有效实际工厂核验为331；322/327对照因缓存空比较已标INVALID，不能引用。先读[最新交接](docs/HANDOFF_20261109_INVERSE333.md)及[完整图文](../output/precision333_20261109/index.html)。以下“当前”为历史批次。
<!-- inverse333-current:end -->

<!-- torque302-current:start -->
> **2026-11-09：不等逐桨推力已接力矩/未来法线，收益仍弱。** 新可选RAW参数 `--rotor-torque-innovation`，默认关；逐桨f²保持不同，缺桨交接未知项。实际原生15号48.476→48.470px，另两段不变；CPU联合相位15号约−0.110px，尚无显著普适收益。原生同源差动仅0/1/0包，角响应尚未可靠辨识。旧基线核验保持，30px/30Hz未达。先读[当前交接](docs/HANDOFF_20261109_TORQUE302.md)和[图文结果](../output/torque302_20261109/index.html)。下方“当前”为历史批次。
<!-- torque302-current:end -->

<!-- active293-current:start -->
> **2026-11-08 当前精度主线：可选切向未知响应学习已接RAW。** 参数 `--rotor-tangential-memory`，旧默认保留。13/15/16均值37.52→34.78、48.48→48.26、44.85→43.50px；11退约1.10px；冻结后新3经图像核验参考69.68→68.47px（仅5点）。仍未稳定30px/30Hz，不称真实阻力/推力标定。先读[当前交接](docs/HANDOFF_20261108_TANGENT292.md)与[图文结果](../output/precision293_20261108/index.html)。下面旧状态均为历史。
<!-- active293-current:end -->

<!-- active280-current:start -->
> **2026-11-08 当前：只攻移动目标的周期过滤入口。** 新录像暴露原/几何组合0个可评分预测；补偿后恢复4个、均27.75px，但另一录像退化，因此没有默认替换、没有宣称30px达标。正在将周期证据映回原像素，避免影响原几何和位移。先读[当前交接](docs/HANDOFF_20261108_ACTIVE280.md)。下面较早状态均为历史。
<!-- active280-current:end -->

<!-- active258-current:start -->
> **2026-11-08 最新：用户要求逐个解决精度根因，禁止按录像拟合。** 研究线仍为原生ON/OFF、五圆可见性和二力模型，无NN。GPU已释放。先读[当前交接](docs/HANDOFF_20261108_ACTIVE258.md)：位置/简单速度修正不是共同大头；单项未来合力记忆学习在三段都退化，未采用，预留experiment1尚未打开。联合相位在16局部改善1.52px、另两段略退且太慢，不是普适成功。真实读取空轮询已修复，候选cooperative短测22.06FPS/有效15.95Hz，30px/30Hz仍未达。以下全部旧状态均为历史。
<!-- active258-current:end -->

<!-- cpu235-current:start -->
> **2026-11-08 CPU续接：** 主预测误差没有新增改善，30px未稳定达成；GPU仍由用户使用，不测速。复用作者CPU联合几何，在41个既存候选中独立半窗确认2→4（新增2、保留原2），不是召回率、真实RPM或轨迹收益。新`joint_ring_phase`模块只输出观测，力反馈关闭。先看[当前交接](docs/HANDOFF_20261108_CPU235.md)、[本轮图文](../output/cpu_resume235_20261108/index.html)。旧主线/NN发布版保留，下方为历史。
<!-- cpu235-current:end -->

<!-- research230-current:start -->
> **历史资源说明（已被用户后续解除）：当时用户正在使用GPU。** 本轮GPU实验已结束；告知空闲前不再启动GPU任务。近期RAW帧率受共用负载干扰，之前19.9Hz等数值仅保留为当时观察，不能作为稳定/独占性能结论。新增可选累计研究入口`run_rotor_candidate.py`，旧入口/NN发布包保留。源时间精度改进仍是局部，未稳定30px；30FPS等待空闲验收。先读[当前交接](docs/HANDOFF_20261108_CURRENT230.md)。下方为历史。
<!-- research230-current:end -->

<!-- research224-current:start -->
> **2026-11-08 当前研究（进行中）：** 可选`plane-adaptive`把快速法向未知响应与平滑外扰变化率分开，并按到期观测分配因果信用。三段源时间均值改善1.6–3.9px，新11轻退，未稳定30px。原生完整弧/不确定度保原预测，真实RAW无丢帧吞吐15.48→24.73FPS；限制积压后有效发布19.94Hz、年龄中位68ms，实际发布时间0.4s仍60.90px（19/25弱参考起点）。不能称30FPS/30px。旧free/176默认与NN发布包保留。先看[进行中交接](docs/HANDOFF_20261108_PROGRESS224.md)和[精度/速度完整报告](../output/precision_delivery_20261108/index.html)，下方为历史。
<!-- research224-current:end -->

<!-- rootfix191-current:start -->
> **2026-11-07 当前研究191：未知力按机体法线组织，公共RAW入口新增 `--rotor-unknown-force-model plane`。** 二力模型不变，隐藏桨力/未标定法向响应随当前法线变化，其他残差不随平面旋转；幅值从0由过去位置学习，不称真实推力。新开发录像15同起点0.4s均值61.46→52.87px；代码冻结后另一录像16为51.67→48.01px（22同起点、弱事件轮廓参考，非独立人工GT）。旧两段收益小，黄昏17/18/19仍无有效预测，30px/持续RAW30FPS未达成。新研究对比推荐plane，默认free保留176控制，NN发布包不改。先读[当前交接](docs/HANDOFF_20261107_ROOTFIX191.md)和[完整结果/失败图文](../output/rootfix191_20261107/index.html)；下方为历史。
<!-- rootfix191-current:end -->

<!-- fixed-reference176-current:start -->
> **2026-11-07 当前176：固定图像参考，布局中心以未知偏置观测进入同一状态。** 首次几何不再移动输出点，模板锚点同步。纯桨叶relative路线默认fixed，可显式恢复173；NN发布包不变。侧视17/19点0.4s均值38.73→38.62px，另一段7点87.08不变；P95/最大仍略退，30px/持续RAW30Hz未达成。先读[最新交接](docs/HANDOFF_20261107_FIXED_REFERENCE176.md)、[图文报告](../output/layout176_20261107/index.html)。下方为历史。
<!-- fixed-reference176-current:end -->

<!-- gap-rim174-current:start -->
> **2026-11-07 当前：173丢测过冲修复已接入纯桨叶relative路线，174原弧恢复保留候选。** 同固定源起点0.4s：侧视17/19点38.73px不变，另一段7/7点102.04→87.08px；44.2s171.41→69.16。新原弧能扩大局部小核，但尚未改善预测，默认不接管位置。30px/持续RAW30Hz未达成。先读[最新交接](docs/HANDOFF_20261107_GAP_RIM174.md)、[图文报告](../output/gap173_20261107/index.html)。旧NN发布快照不变，下方为历史。
<!-- gap-rim174-current:end -->

<!-- geometry-motion172-current:start -->
> **2026-11-07 当前研究：用户已恢复桨叶主线，阶段170–172。** 凹曲率拆分恢复清晰C四桨证据，修复坏位移绕回绝对位置，批处理弧拟合保持数值相同。纯桨叶侧视17/19点0.4s均值38.83→38.73px；另一录像7点99.59→102.04px，当前定位改善尚未转成未来收益。稳定30px/30Hz未达成。先读[最新交接](docs/HANDOFF_20261107_GEOMETRY_MOTION172.md)与[图文报告](../output/motion172_20261107/index.html)。NN发布快照保持原样，下面为发布及历史说明。
<!-- geometry-motion172-current:end -->

> **当前推荐发布版：2026-11-05 NN主体版。** 按用户最新要求，以较强NN检测定位/跟踪/状态外推为主体，五圆/可见桨数/二力增量默认关闭，保留为未获得稳定收益的创新尝试。发布目录为工作区外层`output/github_nn_main_20261105/event-drone-forecast`，不是下面的旧几何主体包。辅助当前仅固定输入因果回放，未实时串联；不宣称30px/30FPS。详见[最新发布交接](docs/HANDOFF_20261105_NN_RELEASE.md)。该发布快照保留不改；当前研究状态以上方11月7日交接为准。

> **历史2026-11-05收尾：** 用户选择停止本轮主动优化，项目整理为冻结研究快照。当时整理的旧几何发布目录为`output/github_research_closeout_20261105/event-drone-forecast`（在工作区外层output中），包含旧RAW基线、独立阶段169几何后端及SDK/GPU-free数值示例；不包含整个本机实验工作区。新快照尚未上传GitHub，30px/30FPS和推力预测增益仍未证实。先读[收尾交接](docs/HANDOFF_20261105_RESEARCH_CLOSEOUT.md)，下文研究记录不应当作公开README直接使用。

<!-- decision169-latest:start -->
> **阶段169（2026-11-05，有限收口）：** 新增可独立调用的桨叶证据／可见性几何后端 `rotor_evidence.py`：四桨无序周期和、三桨近期布局补点、两桨／侧线保留方向和时效，区分当前实测与推断。64个受控单中心缺测案例中，相对完整检测布局的中心一致性均值：可见三点平均21.258px、过去布局速度预测0.837px、共享平移0.325px、旧仿射0.348px；默认采用简单共享平移。**这不是轨迹FDE、真实COM误差或实际遮挡检测精度。** 上游清晰候选用NN辅助ROI，几何恢复不读当前NN机体位置；另附纯桨叶侧视导出。一次固定标定／后段对齐、置零、错时序试验未证实稳定推力预测收益，保留负例，不在这些旧片段上继续扫参数。旧默认／发布ZIP保留，30px／持续30Hz仍未达成。见[图文报告](../output/decision169_20261105/index.html)、[模块调用与贡献边界](docs/ROTOR_MODEL_V1.md)、[交接](docs/HANDOFF_20261105_DECISION169.md)。以下为历史。
<!-- decision169-latest:end -->

<!-- source168-latest:start -->
> **阶段168（2026-11-05）：** 连续局部周期已通过`--rotor-track-local-source`接入正式纯桨叶入口（还需`--rotor-local-phase`，默认关闭），复用原单观测器，无NN。正确解析下当前输入纯侧视81/97→94/97、另一段21/67→36/67；这是资格覆盖，不是真RPM/AP。单观测器原9点35.635→35.528px，后续6点35.530→35.666px，尚未稳定改善。原生缓存正式运行35.527px；稳态29.93FPS但P95 54ms，含首包15.79FPS，不当RAW30Hz。保留额外未知项滤波负例及旧默认/ZIP。见[报告](../output/source_force168_20261105/index.html)、[交接](docs/HANDOFF_20261105_SOURCE_FORCE168.md)与[协议](docs/SOURCE_FORCE168_PROTOCOL_20261105.md)。以下为历史。
<!-- source168-latest:end -->

<!-- force167-latest:start -->
> **阶段167（2026-11-05）：** 研究重点按用户要求回到桨叶检测／五圆可见性分工／二力模型；NN仅作冻结观测。新条件动力学修复参考幅值、速度时间覆盖和依赖输入单位的k上限；辅助下侧视14次／清晰2次有效图像增益更新，纯桨叶仍0次。当前未稳定提精度：清晰同模型27.965→27.873px；侧视43.098→43.850px；较强纯桨叶侧视35.635→35.643px（仅法向35.606）。保留全部负例和旧默认，未发布、无30FPS新证据。见[报告](../output/visible_force167_20261105/source_gain/index.html)、[交接](docs/HANDOFF_20261105_FORCE167.md)与[机制协议](docs/FORCE167_PROTOCOL_20261105.md)。以下为历史。
<!-- force167-latest:end -->

<!-- nn166-latest:start -->
> **阶段166（2026-11-05）：** 按用户授权新增“桨叶结构/可见性/二力模型辅助NN”的固定输入原型，纯桨叶入口与旧默认保留。保持旧NN位置/速度，采用因果增量门控：清晰6终点27.555→26.889px，侧视17点37.110→37.653px，另一录像7点68.193px不变。同资格但已知输入置零得到清晰26.854px，故小收益尚不能归功于周期推力数值；无完整实时30Hz证明。见[图文与坏例](../output/nn_rotor166_20261105/index.html)、[交接](docs/HANDOFF_20261105_NN_ROTOR166.md)与[实验协议](docs/NN_ROTOR166_PROTOCOL_20261105.md)。未训练轨迹NN，未更新发布ZIP。以下为历史记录。
<!-- nn166-latest:end -->

<!-- native165-latest:start -->
> **最新阶段165（2026-11-05）：** 纯RAW ON/OFF桨叶主线保留。加入相对位移相关观测、有限延迟模式、碎片重定位保护与FOV内部匹配；侧视固定源查询均值38.65→35.63px，但P95有退化。另一录像同64包时序185.61→101.22px，全67包121.64→99.59px，仍远未30px；真实发布误差更高。RAW有效输出约20–24Hz，不把缓存30FPS当实时成功。目标未完成，发行ZIP未更新。见[报告](../output/native_plane165_20261104/index.html)与[最新交接](docs/HANDOFF_20261105_NATIVE165.md)。下方条目是历史。
<!-- native165-latest:end -->

> **当前阶段163：** 部分槽空间相位已接到周期选择，清晰段41次合格/30次选用，侧视仍0次；积分图匹配算子保留约11ms小试收益。固定九个侧视0.4s均值仍39.73px，完整链路核心本轮4.87/6.81FPS，**30px/30FPS未达标**。最新[交接](docs/HANDOFF_20261101_PHASE_SPEED163.md)与本地output/phase_speed163_20261101/index.html；--rotor-no-phase可作周期选择控制。旧条目为历史，发行ZIP未更新。

> **最新：2026-11-01 阶段162。** 已按用户定义接入当前可测桨叶贡献＋不可测合力，隐藏桨力归d；部分周期可独立启动、集合切换转移协方差，侧视混合谱仅作变化代理。固定9查询39.73px，未优于阶段161的39.54px；最终RAW短试核心7.15FPS，未达持续30Hz。接手先读[阶段162交接与文献方案](docs/HANDOFF_20261101_FORCE_PARTITION_IMPLEMENTED.md)。下方阶段161及更早条目为历史，ZIP未更新。

> 2026-11-01 阶段160：新增`run_fast_structure.py --rotor-only`完整RAW候选入口，绕过整机NN，用周期种子恢复原生ON/OFF桨盘轮廓、筛选后位移、五圆/部分可见/侧视轴与力学预测；原默认基线保留。滤波图掩膜外真正清零。清晰37窗中14次four、1次three。**尚无稳定预测收益**：侧视固定9个共同源查询0.4秒，旧基线34.84px，新入口周期开/关53.96/44.79px；另外11个按源起点抽样有均值改善但中位变差，收益对取样敏感。GPU核心约3.9FPS，远未30Hz；非实际发布精度。见[真实过滤图与视频](../output/rotor_pipeline_20261101/index.html)、[当前实现与交接](docs/ROTOR_NATIVE_PIPELINE_20261101.md)。旧试点从周期碎块求尺度失败，新方法改为回到原生轮廓，不将旧失败当所有桨盘尺度不可辨识的结论。

> 2026-11-01 桨叶入口试点（按 `docs/ROTOR_ONLY_HANDOFF_20261101.md`）：新增不依赖整机框的桨叶支持图前端 `rotor_support_front.py` 与四段三联对照（清晰/部分遮挡/侧视/空场）。周期支持像素 97.9% 落在冻结转子锥线内、空场 0；单窗连续性物体坐标 IoU 0.18，自估运动对齐后 0.83/0.65；清晰段 4 条区域轨迹命中 24/24 帧。**证伪**：固定间距并簇、圆拟合定桨盘、把前端当输入门控（侧视 34.60→35.79）。**缺口**：桨盘尺度不可辨识、验证门槛缺“周期性非桨”对照，4/3/2 与替代输入因此未做。**未改默认入口、权重与发布 ZIP。** 见[报告](../output/rotor_support_20261101/index.html)与[交接](docs/ROTOR_ONLY_HANDOFF_20261101.md)。

> 2026-10-31 阶段159：小目标候选新增`--recall90`、33ms模型、共享Singer状态与当前框身份门控。三段合格框306/332=92.17%，分段94.68%/92.50%/**89.83%**；FRED9仍差一个框。相同实际查询0.4秒均值FRED7 **101.11→92.55px**，FRED9 **62.73→57.04px**。有预测收益，仍有误报与时延，未全面达成25px／持续有效30Hz。原入口与模型保留。见[报告与视频](../output/recall90_motion_20261031/index.html)、[配置与交接](docs/RECALL90_MOTION_20261031.md)。

> 2026-10-31 阶段158：新增可选`--recall30`小目标检测入口。FRED7/9合格框比例44.2%/53.4%→73.3%/64.4%；同旧源起点0.4秒均值82.60→76.84px、54.18→49.69px。实际墙钟FRED9仍退化，自采侧视39.01px且有长尾，**不替换全部场景原默认**。EV-Tach轻量测速适配未通过，不接推力主线。见[完整报告与视频](../output/rpm_recall_pilot_20261031/index.html)、[交接说明](docs/RPM_RECALL_20261031.md)。旧`--realtime30`与发布ZIP保留。

> 2026-10-31 最新稳定性修复：30Hz入口补回预算内的当前NN失锁恢复，累计原数学编译与约0.33ms几何组件优化。FRED9同14源起点 **59.33→56.72px**、覆盖 **14/19→17/19**；相同实际输出查询为 **69.11→66.40px（13个）**。仍有大误差和计算长尾，25px／持续有效30Hz未全面达到。当前命令使用`--realtime30`，旧控制为`--legacy-realtime30`。净力补充保留为可选，未用开发段小收益掩盖完整流水线退化。

[跨片段根因、完整结果及对照视频](../output/stability_audit_20261031/index.html) · [交接说明](docs/STABILITY_AUDIT_20261031.md)。以下保留此前批次口径，旧发布ZIP未更新。

> 2026-10-31 新目标：持续30Hz，非侧视0.4秒误差优先低于25px，侧视适度放宽。新增工作配置`--realtime30`：严格中心平移、现有独立周期进程、实际时间对齐的同一力学模型输出。实测4／6秒发布120／180次，约30Hz、0漏发布时隙；有效预测108／103次，不能当成全部新检测。当前有标注片段约60.16px、FRED0约28.29px，25px未达成；持续有效新观测30Hz亦未全面做到。

[新目标、运行方式与限制](docs/TARGET_25PX_30HZ.md) · [本轮报告](../output/target25_continuity_20261031/index.html)

下面保留前一批记录与原控制。旧发布ZIP没有更新。

> 2026-10-29 最新：修复“部分机体弱框扩成整机强框时，真实速度被拒”的问题；跨参考的位移仅用于当前速度，不进入加速度／未知力求导。同批24个原起点 **45.88→30.19px**，单次异常 **397.63→20.85px**，后续原预测不变。这是特定失效修复，原26起点31.62px不变。最新完整RAW默认侧视 **31.74px（25/26）**、另一段 **58.93px（14/15）**；可选搜索组合侧视 **32.13px（25/26）**。固定侧视前4秒处理能力70.94FPS，当前观测28.44Hz／有效预测27.44Hz，30px与持续有效30Hz仍未全面达成。

[本批修复与运行参数](docs/REACQUISITION_MOTION.md) · [最新报告与预测图](../output/reacquisition_motion_20261029/index.html) · [事件包与搜索记录](../output/transport_stability_20261029/index.html)

`--target-candidate`已采用修复，`--no-contained-reanchor-motion`保留原门槛。空场搜索降频和3M共享槽仍为显式可选；相邻原调度同样低延迟，跨次测速差不能全归因于降频。下方为前一批记录；已发布ZIP未改。

> 2026-10-29 最新侧视响应候选：侧视以整机轴线和桨叶周期独立运行，不要求四桨模板。新增历史坐标修正，同26起点0.4秒均值 **34.60→31.62px**、P95 **72.54→60.86px**；另一段与FRED 7固定观测不变。新侧视RAW两次为 **36.05/31.57px**，均只保留原25/26起点，仍未稳定达30px。固定前4秒处理能力 **35.64FPS**；可选截止等待版 **33.09FPS**，有明显延迟长尾。修正已接入`--target-candidate`，`--no-position-history-rebase`保留控制；`--deadline-nn`仍为默认关闭的时序试验。

[独立侧视与变向响应说明](docs/SIDE_RESPONSE.md) · [最新报告和预测图](../output/maneuver_response_20261029/index.html)

以下为前一批FRED/物理候选记录，最新改动以上述报告为准。

> 2026-10-29 工作候选，尚未重新发布：沿五圆模型、可见性分工、相对周期推力和未知合力反馈继续优化。实际自采侧视0.4秒均值 **34.60px**（P95 **72.54px**），另一录像 **59.79px**。新增FRED运动资格分工 **48.49→43.18px**、可选小目标复核进一步到 **38.29px**；训练外FRED 7 **80.34→76.75px**。首版精简输出清晰2秒 **32.20处理FPS**，4秒滚动读取组合 **42.37处理FPS/29.03新观测Hz**。30px与稳定低延迟30FPS目标尚未全面完成。

当前推荐入口为run_fast_structure.py --target-candidate，包含当前加速度曲线、证据增益、端点未知合力平衡、已标定的缺周期持续性及严格弱目标运动资格，自动启用已有刚体位移/运动复用。SDK AVX2和结构控制NN需显式启用；原生周期为默认。适配模型复核、滚动读取、精简输出、精简历史均为显式选项，原权重与旧基线保留。

[FRED迁移与本轮选项](docs/FRED_TRANSFER.md) · [本轮完整实验报告](../output/fred_domain_adaptation_20261029/index.html) · [原物理方法与对照](docs/MOTION_RESPONSE_30PX.md) · [全部结果入口](../output/结果入口.md)

> 2026-10-29 归属审计、推力通道杠杆与覆盖诊断：整机空间方差的主峰属于某一个桨，不是整机。清晰段 26.100–26.600 秒的 14 个时刻，多峰谱把四个桨同时分辨出来（逐峰偏差中位 1.09%），同一截止时刻的单峰读数 231.875 Hz 与上一批逐位相同。消融：推力通道存在与否值约 **6.77px**（侧视 34.60→41.37），但 3% 的输入差只让预测移动 **0.38px** 以内，推力变化率按证据关闭。覆盖缺口也已定位：被打分的 4 个无输入起点全在出画区间，池化估计仍给 263–290Hz 但只有 2–23 个频率像素；补齐最多值 0.4–0.6px（“23px 差”是选择偏差）。源帧定位均值 6.92px，0.4 秒误差 34.60px——差的那约 28px 长在外推里，下一批回运动外推。见[归属审计](docs/PEAK_OWNERSHIP.md)与[报告](../output/peak_ownership_20261029/index.html)。

已发布alpha.1/alpha.2 ZIP保持冻结，本页工作候选指标不代表旧发布包的结果。

<details>
<summary>历史研究进展，非当前指标</summary>

> 2026-10-28 新阶段：目标改为0.4秒平均误差低于30px，保留30FPS要求。当前采用[无机头圆饼法线预测](docs/DISK_NORMAL_FORECAST.md)，不依赖桨盘编号；实际侧视同输入42.805→42.404px，固定前4秒34.09处理FPS。新方向已纳入target-candidate，可用--no-normal-turn-forecast对照。[最新报告](../output/disk_motion_30px_20261028/index.html)。周期时序支持恢复4次输入但未改善另一录像精度，保持显式实验。目标尚未达成，已发布alpha.2仍保持冻结。
>
> 2026-10-28 输出整理：最新结果入口见 [output索引](../output/结果入口.md)。历史散落结果已完整压缩到 [历史归档](../archive/output_history_20261028.zip)；下文旧output路径仅在按需恢复后可用，当前运行与发布依赖已保留。



> 最新实际结果：8秒/26目标的0.4s均值 **43.27px**、P95 **80.60px**、16/26在50px内；原有目标段处理能力 **35.41FPS**，清晰段 **32.94FPS**。可见范围扩张造成的可靠框中心跳变已与实际事件位移分开处理。train2仍 **62.65px**、周期输入为0，初期仍有积压，整体目标未完成。运行参数见[当前候选说明](docs/TARGET_50PX_30FPS.md)；发布ZIP保持原样。

> 2026-10-28本地候选：[50px/30FPS新进展与运行方式](docs/TARGET_50PX_30FPS.md)。侧视实际RAW 0.4s均值47.21px、处理能力36.88FPS；P95仍83.84px，只有9/18小于50px，初期有积压。新增可靠重定位时的速度纠偏与同窗二力反馈，位置和有效评分时刻保留。清晰段约28–30FPS、train2约60px，整体目标尚未完成。原发布ZIP不变。

> 当前目标：[0.4s平均误差50px与30FPS](docs/GOAL_50PX_30FPS.md)。候选入口加`--goal-profile`：近期未知力修正、当前侧轴/完整桨盘多速率、异步NN与解码降噪重叠。固定侧视4s实际52.01px、处理能力33.60FPS、当前观测28.13Hz；清晰段19.17FPS，train2周期输入不足，**两目标尚未同时满足**。同一新观测旧力学65.89→52.01px，P95仍95.83px；目标不能理解成每帧都在50px内。原模型和失败实验保留，公开ZIP未更新。

> 2026-10-27：新增[完整流水线等价计算优化](docs/FINAL_PIPELINE_SPEED.md)，默认启用，`--reference-compute`保留原计算路径。两段各60时刻的框、几何和预测逐项不变；本机同步对照侧视13.71→17.57FPS、清晰14.78→22.19FPS，短样本含调度抖动。可选降噪解码重叠+异步侧视4秒试验24.77FPS、当前观测23.02Hz，尚未稳定30FPS，也没有新的检测精度收益。已发布ZIP保持原文件。

> 最新结构方向：`structure-fast`默认使用[连续五圆共享状态](docs/CONTINUOUS_FIVE_CIRCLE.md)，保留`--legacy-structure`对照。局部两桨和轴线能更新同一模板，新增共享模板引导当前真实弧提取；同事件消融中三桨成功0→1次，不伪造隐藏桨心/转速。轨迹优势和稳定三桨恢复仍未证实，最终完整入口计算能力19.00FPS，未达稳定30FPS。报告：`../output/continuous_five_circle_20261026/index.html`。下方快入口测速为前一批历史结果。

> 当前五圆物理快入口：`forecast_mainline.py structure-fast --raw <event.raw> --output <新目录> --start 110 --duration 3 --async-nn --paced --verify-frames`。按用户要求，默认移除实时传统加速度/尺度预测，仅保留力学所需p/v初始状态；传统方法只作离线对照。五圆、可见性、周期平方代理与总扰动已接通，最新两小段计算能力约28FPS、实际新观测25–26.5Hz，**未达到稳定30FPS，物理精度仍未优于对照**。见[快物理主线](docs/FAST_PHYSICAL_ROUTE.md)与本机`../output/five_circle_fast30_20261026/index.html`。

> 项目研究主线是用户提出的五圆结构与可观测性切换、姿态/桨叶周期驱动的有效推力估计、总扰动反馈及变向响应。[机制3原型](docs/FIVE_CIRCLE_V3.md)入口为`forecast_mainline.py structure`。当前做过几何、周期与扰动实验，可靠的力学预测与参数学习闭环仍未完成；已有定位收益不能作为物理预测成功的证明。

> 最新结构累计版：[三桨续接与二力反馈](docs/THREE_CONTINUATION.md)。当前短弧法线约束修正五圆共享状态，未知合力从首个成熟位移开始学习，短缺测有120ms加速度保持。完整RAW三桨2次、清晰/侧视计算能力30.78/36.08FPS；持续低延迟30FPS和物理预测优势仍未证明。最新本机报告：`../output/three_continuation_20261026/index.html`。

> 独立周期实验：[五圆共享几何与批量空间相位](docs/SHARED_ANGULAR_PHASE.md)。用原生事件的角度—时间一致性验证旋转，60时刻取得47个当前逐桨周期，与匹配SDK周期平均差1.10%；没有独立RPM真值或位置预测收益证明，未替换默认周期输入。

> 工程对照与回退：使用 [forecast_mainline.py](forecast_mainline.py) 的`run`命令默认运行累计v7：整机框、原生ON/OFF光流、Kalman及运动学预测。v7默认位置状态不由五圆姿态或桨叶周期推力驱动。原版通过`--profile live-baseline`保留，历次基线改动见[累计优化说明](docs/ACCUMULATED_MAINLINE.md)。


> 2026-10-26 更新：v7默认无视频侧视RAW，同起点0.4s平均60.14→55.23px、P95111.70→92.48px，但中位53.36→58.62px，样本较少。train2信号不足保持原预测。最新本机报告/实际视频在 ../output/flow_acceleration_20261026/index.html；首次带视频17/20有效与默认无视频18/20结果均保留。此前检测器候选记录见 [模型候选与运行方式](docs/DETECTOR_CANDIDATES_20261026.md)。旧 wheel 和源码 ZIP 不包含本轮全部更新。

</details>

事件相机无人机跟踪与二维短时轨迹预测工具箱，含桨盘残弧、粘连、侧视退化与条件推力代理实验。

输入 Metavision RAW，使用纯ON/OFF事件特征，输出当前整机观测、未来0.1/0.2/0.4秒像素位置、预测视频和可检查的运行记录。当前为 **Windows + NVIDIA CUDA 的研究源码候选版**。

## 快速开始

1. 按 [环境与运行说明](docs/QUICKSTART.md) 安装SDK/GPU依赖并准备兼容权重。
2. 复制 `configs/runtime.example.json` 为 `configs/local.json`，填写Python与权重路径。
3. 自检后运行：

```powershell
python forecast.py doctor --config configs/local.json
python forecast.py run --config configs/local.json `
  --raw '<你的event.raw>' --output '<新的输出目录>' `
  --start 22 --duration 2 --profile research --preview
```

`forecast.py --help` 不需要SDK或CUDA。配置中的相对路径以配置所在目录为准。源码包不附RAW、SDK或检测权重；**模型获取/训练路径尚未完整整理，当前不是clone即用的正式发行**。

## 功能状态

| 层次 | 当前能力 |
|---|---|
| 核心流程 | RAW读取、ON/OFF处理、整机检测、位置/速度估计、短时外推、视频与因果记录 |
| 可选低延迟通道 | 独立RAW/SDK进程、最新观测槽、独立5Hz心跳发布、处理延迟补偿；短试验无漏发布槽，未保证连续有效5Hz |
| 可选研究组合 | 几何速度观测、LM拟合、安全双臂并行、未建模曲线回退 |
| 实验桨盘分支 | 阈值抑制、原弧拆粘连、缺桨推断、视觉身份续接、条件周期和相对推力代理 |
| 尚未完成 | 稳定真实RPM/牛顿推力、可靠侧视姿态、独立物理预测增益、连续有效5Hz及实时相机接入 |

`baseline`保持原算法默认；`research`使用当前156可选组合；`rotor-experimental`另加实验桨盘分支，含义见 [快速开始](docs/QUICKSTART.md)。5ms预览、20ms切片与视频FPS不等同于实际在线更新率。

新增 `--profile live` 用于低延迟位置预测，配置与实测见 [低延迟模式](docs/LIVE_MODE.md)。它的精度暂未胜过旧状态参考，不替代原研究流程。

另提供[精简实时源码候选包生成器](docs/LIVE_RUNTIME_SOURCE.md)，隔离历史测速依赖。用户关闭游戏恢复后，新检测进程、可选启动同步和移出目录运行已完成1.2秒RAW接线；复用本机环境，仍非干净机器完整部署验证，也没有新的精度增益。

## 已有结果

已用于开发的自采录像上，0.4秒预测平均误差：侧视46.430→45.594px，另一段59.301→58.930px。均为**起点确实发布的同一批预测**，分别64对和30对，缺失11对和6对。改进较小，侧视中位误差略变差；没有证明SOTA或未知录像泛化。详见 [结果与局限](docs/RESULTS.md)。

本地候选包的 `examples/` 提供少量已有预测演示；完整历史输出仍在原工作区外层output。视频中的50fps是录像播放时间轴。

## 文档导航

- [快速开始与评分](docs/QUICKSTART.md)
- [低延迟位置通道与实际发布结果](docs/LIVE_MODE.md)
- [流程、输出字段与代码导航](docs/ARCHITECTURE.md)
- [结果与当前缺陷](docs/RESULTS.md)
- [检测模型来源与获取缺口](docs/MODEL_CARD.md)
- [下一步工程路线](docs/ROADMAP.md)
- [本地发布准备清单](docs/RELEASE_CHECKLIST.md)
- [第三方代码与许可证](THIRD_PARTY_NOTICES.md)
- [项目贡献与创新定位](OPEN_SOURCE_POSITIONING.md)

研究历史保留在 `research/stage*`；旧README存入 `docs/README_HISTORY_THROUGH_STAGE156_20261025.md`。原控制台CLI和`run.py`仍是历史流程；0.3.1新增可安装的[位置运动Python核心](docs/POSITION_CORE.md)，当前完整RAW推荐入口仍为`forecast.py`。

本轮还加入独立5Hz发布时钟、本机HelixTrack旋转观测旁路和侧视周期投影轴。旋转支持更连续，尚未证明提高位置预测精度或估计真实推力；进展与负结果见[持续优化记录](docs/OPTIMIZATION_JOURNAL_20261025.md)。作者核心代码与DLL未进入源码候选包。

本项目是已有方法的工程整合与实验实现，不是所参考论文的官方代码。自有代码沿用MIT，第三方文件保留各自许可证。当前仅整理本地候选包，未发布GitHub远端仓库。

2026-11-01新增纯桨叶研究入口：`run_fast_structure.py --rotor-only`，不读取整机NN权重或G重建。五圆/部分桨/侧视观测接入联合位置、速度和合并未知作用估计；`--rotor-gpu-front --gpu-python <CUDA Python>`启用稀疏GPU前端，`--no-preview`关闭诊断视频导出。`--rotor-legacy-force`保留旧残差算法对照。这是离线研究入口，尚未达到持续30Hz或稳定25px，详见[本轮实际结果与限制](docs/ROTOR_OBSERVER_GPU_20261101.md)。

交接先读[用户最新二力模型要求与待办](docs/HANDOFF_20261101_OBSERVED_UNOBSERVED_FORCES.md)：可测力只计当前可靠贡献，隐藏桨推力归不可测项；现有代码的完整四桨初始化及隐藏先验计入总输入尚需按此修改，本次仅整理日志。
# 2026-11-01 当前纯桨叶入口更新

当前主线为原生RAW ON/OFF桨叶支撑→五圆/部分桨/侧视→当前可见周期贡献＋未知合力→二维预测。隐藏桨力归未知项，不补作已测RPM；部分周期不必先等待四路同时合格。侧视混合谱只作变化代理。

最新实施与交接：[阶段162分账、当前限制和下一步](docs/HANDOFF_20261101_FORCE_PARTITION_IMPLEMENTED.md)。原有阶段161数字及实施前交接是历史记录。新固定9查询0.4秒均值39.73px，未优于阶段161的39.54px；尚未达25px/持续30Hz。

`run_fast_structure.py --rotor-only --rotor-gpu-front --no-preview --gpu-python <CUDA Python> --raw <RAW> --output <新目录>` 使用当前桨叶路线；加`--rotor-legacy-decomposition`回放阶段161分账控制，`--rotor-legacy-force`回放更早残差控制。原NN入口保留。已发布ZIP未更新。


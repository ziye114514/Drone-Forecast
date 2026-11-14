# README 历史状态横幅存档（2026-11-14 整理前）

本文件保存重写 README 时从顶部移除的 19 条 `<!-- xxx-current -->` 历史状态横幅，原文未改。
按时间倒序，最新一条在 2026-11-13（collaboration523）。正文与完整实验记录见 `WORKLOG.md` 与 `docs/`。

---

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


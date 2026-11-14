# 阶段164交接：原生CUDA、局部周期、解码流水线与位置基准

2026-11-01。用户已撤销10%额度停止条件，继续实际优化，不用重置券或购买额度。目标仍为0.4秒二维预测均值≤30px、持续30FPS新观测；没有完成。不要重新回到NN/G前置，也不要把隐藏桨力放回可测项。

## 已完成，不要重复实现

1. `native/rotor_harmonics.cu`、`rotor_nvrtc.py`：使用本机NVRTC编译项目自己的CUDA内核。原生16字节事件计数、20ms计数复用、31×31密度、无排序CSR、两半窗NDFT及验证融合；保持全部当前原生时间戳。没有安装Numba/MSVC。原Torch路径保留。
2. 九个真实保存窗口的原生支撑及所选频率与旧参考相同；谱分数有约1e-4浮点差，不能宣称普遍逐位相同。第二次warp归约把共享统计区32KiB减到1KiB，九窗原生/频率字段相同，谱分数差≤2.9e-6；设备短试清晰10.85→9.31ms、侧视12.02→11.10ms，空场反而3.18→4.16ms。不是整线FPS。
3. CPU复用GPU的严格20ms ON/OFF计数；去掉完全未消费的通用连通分量描述，原生支撑重建仍保留。CPU三个工作区、有界一个未来GPU特征包；未来特征不进入当前模型。另复用现有`ProcessRAW`的两槽解码进程，不重新造解码器。
4. `local_pixel_phase.py`：保留每个像素自己的第一谐波相位，聚合局部的后半窗投影/能量，避免把不同像素的ON/OFF原始信号相加抵消。侧视无需椭圆/共享平面；区域不是电机身份，不能按区域数补总推力。最初每块4096事件的CPU原型真实窗全溢出，已被GPU充分统计量方案替代。
5. 局部与全局周期估计器使用不同的贡献键和零基线，切换来源转移到未知d，不能当成推力阶跃。真正同源频率/法线变化仍保留。混合周期不能校准成真实机械RPM/牛顿力，侧视k增益仍不学习。
6. 原生桨叶区域匹配增加`--rotor-subpixel-motion`，当前相关峰作二次插值，避免每帧整数位移量化。合成位移误差0.361→0.0245px，实录像有小收益，不能把合成数字当真实检测精度。
7. 首次进入画面被截断时，区域中心是暂定点；第一次完整可见再建立位置基准，暂定点不作为机体位置测量。否则暖启动会带着约22px横/65px纵偏移持续积分。66.43→45.30px修复的是这个偏差，仍有启动速度/加速度问题。
8. 官方EventsIterator要求起点是33333us的整数倍。现通过相邻原生包重新组装保持原始任意起点和全部时间戳，没有更改SDK。必须显式传入SDK16字节dtype，NumPy默认拼接会变为14字节，导致共享内存重解释错误。坏跑`raw_iterator_rechunk`已加INVALID_ABI.json，绝不能取其零检测/29FPS当成果；`fast_raw_reader`新增dtype拒绝守卫。
9. 来源快照在worker启动之前，结束检查源码哈希。导出白名单补NVRTC源码和local_pixel_phase，尚未重建或上传公开ZIP。

## 当前实测及口径

结果：`output/side_temporal164_20261101/index.html`、`summary164.json`（汇总生成后为准）。

- 固定九个侧视开发查询110.4–112.0s：阶段163 39.730px；局部周期39.578px；加亚像素及来源记账后直接RAW **38.647px**。小收益保留，没有达到30px。
- 同一录像带108s历史启动：旧截断初始化66.427px；完整可见重锚45.299px；亚像素/来源记账 **44.106px**。后续112.2–114.0s十个预定查询可评分8个：46.412→44.424px。它是时间延伸，仍是同一录像，不能叫独立新录像或把缺失两个查询丢掉不报。
- 最终侧视直接RAW：61包/60当前桨叶区域/57有效源预测/48代理输入，完整稳态 **24.596FPS**、间隔P95 **77.37ms**。CPU模型单独34.88FPS不代表完整FPS。
- 内存回放短清晰/侧视曾分别30.01/30.50FPS；6.4秒暖启动内存候选30.41FPS，含空场。**不是RAW持续30Hz，不是每33ms都按时输出**。完整RAW还没达标。
- 当前成绩是源时刻+0.4秒，以已有人工整机框中心为参考；未包含真正发布延迟。没有独立COM/姿态/RPM真值，也没有新FRED纯桨叶迁移成绩。清晰段没有充分同口径未来位置参考，不编造误差。

## 负结果保留，不继续扫九个标签

- 位移速度观测39.94px、旧力记忆42.68px、轴向未知力先验50.08px、放大初速先验41.30px、五帧启动拟合冷41.16px；均未接默认。
- 修正状态更新与未来Singer衰减的一致性：`RotorForceObserver(consistent_dynamics=True)`。沿用已有Singer精确离散化，净a=d+k*u整体用同一条件先验，分账切换保持整个未来不变，不能只让d衰减。机制成立，但冷42.70px、暖44.30px、后续54.47px，整体未优于控制，默认False。
- 透视尺度外推：侧视合格匹配椭圆0次，不增加观测。不要把未知深度当已测。

## 接手优先做什么

1. 精度：启动动力学与单点平移在旋转/尺度变化中的偏差。已有完整可见重锚和亚像素，不要再提为“首次实现”。下一步优先用实际当前桨盘/局部区域的共同变换约束同一机体中心；固定中心四桨波动要消掉，不能用包围框大小变化制造加速度。复用项目刚体/相似变换代码，输入仍只允许当前筛选后的桨叶。
2. 速度：完整RAW约24.6，CPU本身平均可超30但P95很差。下一步用少量分段计时找出相关匹配、几何刷新、预取解包的长尾，再做预算。不能把少算当前目标/丢事件/空发布当30Hz成功。保留10Hz完整几何刷新和30Hz当前轴证据的清晰含义。
3. 周期力：每像素局部模板解决了可用性，没解决叶片数/倍频、混合源切换、k与d可辨识性。真实转速变化率必须有当前相位证据，不能把算法来源切换当转速变化。没有新真值也可做可控合成干扰和独立时间段，但不要增加无意义验证量。
4. 泛化：下一次实际改动后，增加一小段另一自采录像/FRED当前桨叶输入，报告覆盖与归一化误差。FRED像素太少时应承认逐桨不可观测；不能借NN召回指标冒充该纯桨叶入口。

## 可运行入口

SDK Python `D:/drone forecast/.venv-event311/Scripts/python.exe`，CUDA Python `C:/Users/子夜/AppData/Local/Programs/Python/Python313/python.exe`。

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\run_rotor_only.py --raw 'E:\dataset\train\6\media\event.raw' --start 110 --duration 2 --output '<新目录>' --gpu-python '<CUDA Python>' --rotor-gpu-front --rotor-native-gpu --rotor-fast-geometry --rotor-prefetch --rotor-no-phase --rotor-local-phase --rotor-subpixel-motion --rotor-reader process-iterator --no-preview
```

`--rotor-no-phase`只关闭需要椭圆平面的空间相位，不关闭原生时间周期/力学输入。默认NN和参考Torch路径保留；遗留分账控制使用相同当前前端，历史完整控制见backup/20261101_pre_fused164。RAW、权重及两处yolov10目录没改。

研究参考：[DDHF](https://arxiv.org/abs/2603.08386)、[HelixTrack](https://arxiv.org/abs/2603.09235)、[NVRTC官方](https://docs.nvidia.com/cuda/nvrtc/index.html)、[Stone Soup Singer](https://stonesoup.readthedocs.io/en/v1.2/stonesoup.models.transition.html)。本批复用思想与本项目代码，没有复制新的论文作者实现。

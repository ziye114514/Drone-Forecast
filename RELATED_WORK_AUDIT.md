# 相关论文与代码核查（2026-10-16）

只把作者论文、作者项目页和实际 GitHub 源码作为技术依据。下面分别标明“读过/采用思想/实际执行”，不把下载当复现。此次搜索不是对所有论文的穷尽证明。

| 工作与原始来源 | 与本项目的关系 | 公开代码与本轮状态 |
|---|---|---|
| [Relative State Estimation using Event-Based Propeller Sensing](https://arxiv.org/abs/2604.18289v1)，2026 | 核心参考：逐桨频率→总推力，椭圆→推力轴，两 KF 耦合 | 未确认完整官方仓库；根据用户 PDF 独立实现，差异见 PAPER_REPRODUCTION.md |
| [HelixTrack](https://arxiv.org/html/2603.09235v1)，2026 | 事件映射至转子平面，逐事件相位 EKF 与批次 GN 同时更新几何/转速 | [实际 C++ 源码](https://github.com/radimspetlik/helixtrack)。已读本机源文件；当前机器未有可用 C++/OpenEB 构建链，未执行原程序。本轮只采用平移补偿和相位一致性思想，不称移植了其优化器 |
| [Efficient Real-time Quadcopter Propeller Detection and Attribute Estimation with High-resolution Event Camera](https://github.com/radimspetlik/effqpdet)，SCIA 2025 | 前端桨盘检测、椭圆属性与测速 | 作者 C++ 源码存在；未完成原程序构建。不包含通用未来无人机位置预测 |
| [EEPPR](https://arxiv.org/abs/2408.06899)，ICMV 2024 / proceedings 2025 | 三维时空相关估计周期；自带 front/side 等有频率真值的公共 RAW | [实际 Python 源码与数据](https://github.com/JackPieCZ/EEPPR)。本轮实际调用未改动的 eeppr.py/utils.py 核心，测试 front/side；SDK3.11 与 CUDA Python3.13 间用原事件解码文件桥接，只替代 IO/矩形过滤，参数和算法保留。详细结果、桥接及指纹均保存 |
| [frequency_cam](https://github.com/ros-event-camera/frequency_cam) | 每像素 IIR 亮度状态和零交叉估周期，可用于缺少完整桨盘轮廓时的备选 | ROS/ROS2 真正源码；已读，本轮未跑原节点；只测速/频率图，不自动给出完整姿态/轨迹 |
| [EventPro](https://arxiv.org/html/2511.13100v2)，SenSys 2026 | 地面事件相机测速、飞行指令判别，结合历史与其他定位源改善追踪 | [作者项目页](https://eventpro25.github.io/EventPro/)、[仓库](https://github.com/Eventpro25/EventPro)。核查时仓库主要 README/展示资源，未确认完整系统源码。不能把“仓库存在”写成可直接复现系统 |
| [Event-Only Drone Trajectory Forecasting with RPM-Modulated Kalman Filtering](https://arxiv.org/html/2603.01997v1)，ICUAS 2026 | 最直接的二维预测对照：RPM 调制 KF 过程噪声，0.4s/0.8s 预测 | 检索未确认官方实现。论文实验用过去真实标注框/差分速度隔离检测误差；本项目自动检测输入条件不同，不能直接比表格数字。这里没有冒称复现其全部超参数 |
| [EVPropNet](https://prg.cs.umd.edu/EVPropNet.html)，RSS 2021 | 合成事件训练桨叶检测器；遮挡下检测有实验支持 | [实际代码](https://github.com/prgumd/EVPropNet)。本轮未训练/跑其网络。部分桨叶仍能被检测，不等于椭圆姿态或推力已经准确恢复 |
| [EVPeriscope](https://arxiv.org/html/2609.11920v1)，2026 | 近期事件桨盘定位系统：持续衰减表征，已知非对称机体几何 PnP，200Hz 协同控制 | [实际代码](https://github.com/grasp-lyrl/evperiscope)。已读方法，未跑机器人 ROS 栈。已知几何、上视相机与受控协同平台，不是任意未知飞行目标的 RAW-only 未来预测保证 |
| [EV-Tach](https://wrap.warwick.ac.uk/id/eprint/181045/1/WRAP-EV-Tach-handheld-rotational-speed-estimation-system-event-camera-2023.pdf)，2023 | 手持事件相机测速，ICP、背景抑制、粗细估计；适合考察观察者小幅运动 | 未确认相应完整官方仓库；无本轮复现成绩。一般测速实验不能直接证明严重侧视下完整飞行状态可观 |
| [EventRadar demo](https://zhouzt21.github.io/EventRadar-web/)，SenSys 2026 demo | 远距离周期性证据、推力关系辅助载荷推断与云台追踪 | 作者展示页，未确认完整官方代码。另有 [EventRadar arXiv 2606.11285](https://arxiv.org/abs/2606.11285) 长距离发现论文；两者标题、方法和评测不同，不能混合指标 |

## 侧飞、残桨与极端侧视已经到哪一步

1. **残弧补全是已有几何问题。** [Arc-support Line Segments Revisited](https://arxiv.org/abs/1810.03243)、[ELSDc](https://github.com/viorik/ELSDc) 等已经处理椭圆弧检测/组合。本项目的曲率筛选与拟合是独立简化实现。较长弧和历史约束可帮助恢复中心/椭圆；短弧在噪声下有很多相似解，必须传播不确定性，不能强填一个高置信椭圆。
2. **几何和事件时间联合估计已经有人做。** HelixTrack 就是强相关工作；其 §4.8 明确写到极端侧视、严重缩短、非平面或重遮挡会破坏模型并降低性能，并且实验不做失败后的自动重初始化。不能声称它已经普遍解决了侧视，也不能认为简单补椭圆会自然超过它。
3. **侧视仍可能测到周期。** EEPPR 提供侧面旋转目标 RAW，可直接核查频率；frequency_cam、EV-Tach 属于可参考的时间信号路线。但固定目标周期正确，不代表飞行中能同时找对每个桨、测出总推力和方向。
4. **“像木板”只约束几何。** 桨盘扁成线仍可约束图像内轴方向；推力近似垂直于桨盘。速度方向受之前的速度、重力、阻力和控制决定，不等于木板斜率。向右倾斜时无人机仍可能正在向左飞，只是在减速。
5. **未知参数适应有大量先例。** RLS、残差动力学、扰动观测等不是新的单项概念。较可研究的是：在桨盘退化、谐波歧义、可见桨数改变时，如何给出可信的观测门控/不确定性，并证明与历史运动、纯周期、纯几何和强公开跟踪器相比带来独立增益。当前只能称待验证方向。

## 本轮证据的强弱

- 实际执行：自己的完整 RAW 管线、测试、受控观测实验、EEPPR 公共正/侧视周期实验。
- 已看源码/方法，尚未执行原系统：HelixTrack、frequency_cam、effqpdet、EVPropNet、EVPeriscope。
- 没确认完整源码：核心参考论文、RPM-Modulated KF、EventPro、EV-Tach、EventRadar。
- 尚无证据支持“超过参考论文”：缺相同数据、相同真值、同一协议和完整标定。自己的像素预测误差不能和论文的度/米/RPM 误差直接比。

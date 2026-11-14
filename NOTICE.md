# 方法来源与实现范围

**2026-10-25更正**：以下主体记录早期v0.1—v0.3与历史对照，不代表当前优化入口的依赖范围。当前运行链实际执行随附FilterPy原函数，并导入circular-center-calibration的Python源码；第三方许可证、模型和数据边界以 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 为当前说明。下文“运行时不导入研究仓库”仅描述其所列早期入口，不能推广到 `forecast.py` / `run_optimized_forecast.py`。

本项目是独立研究原型，不是下列论文的官方实现，也没有宣称复现其论文成绩。
`run.py` / `run_paper.py` 运行时不导入或执行工作区下载的研究仓库；主预测源代码独立编写。可选公共基准工具 `run_eeppr_benchmark.py` 单独导入原作者 EEPPR，见下方说明。

- [Relative State Estimation using Event-Based Propeller Sensing](https://arxiv.org/abs/2604.18289)：桨盘几何与转速共同提供动力学线索。
- [effqpdet / SCIA 2025](https://github.com/radimspetlik/effqpdet)：重复事件空间分布和椭圆属性作为前端思路。
- [HelixTrack](https://github.com/radimspetlik/helixtrack)：几何、周期信息耦合及因果更新的参考；本项目未移植其相位优化器。
- [EventPro](https://arxiv.org/abs/2511.13100)：分桨测速、历史运动与转速辅助预测。
- [Online Learning and Control for Data-Augmented Quadrotor Model](https://arxiv.org/abs/2304.00503)：迟到的预测残差用于后续模型适应。
- [T2S-MPC](https://arxiv.org/abs/2605.24852)：时变动力学适应的对照思路；本项目使用递归最小二乘，不实现其神经 MPC。
- [Residual Kalman Dynamics](https://arxiv.org/abs/2609.00839)：纯运动残差基线及事件信息增益的对照。
- [Arc-support Line Segments Revisited](https://arxiv.org/abs/1810.03243)：残弧分组与椭圆恢复的已有工作对照；未复制其 MATLAB/C++ 实现。
- [ELSDc](https://github.com/viorik/ELSDc)：椭圆弧与线段联合检测的已有工作对照；未复制或链接其 AGPL 源码。
- [Guaranteed Ellipse Fitting with a Confidence Region and an Uncertainty Measure for Centre, Axes, and Orientation](https://cs.adelaide.edu.au/~wojtek/papers/ellipsefitJournal.pdf)：椭圆不确定性方法的参考。本项目只用局部线性化协方差作诊断，没有实现该文的保证。

v0.2 的残弧模块采用独立编写的曲率筛选、有界确定性拟合假设和 OpenCV 拟合器；并未移植上述论文的完整方法。半窗相干与谐波检查也不能替代独立测速真值。

外部依赖由各自许可证约束；Metavision SDK 需独立安装，本仓库不分发该 SDK。
本项目不包含上述研究项目的第三方数据或模型权重。

v0.3 对照主论文的推力/推力轴/位置 KF 结构，逐项差异见 `PAPER_REPRODUCTION.md`。转子平面相位一致性只是独立的简化周期判别，不是完整 HelixTrack。

公共周期基准实际执行 [EEPPR](https://github.com/JackPieCZ/EEPPR) 原作者未修改的核心和工具代码。其代码及数据为 GPL-3.0，保留在用户已经下载的独立 `EEPPR-main` 目录；本项目打包不拷贝它们，也不将其重新标为 MIT。桥接脚本仅替代 SDK IO 并记录源码指纹；公开结果需引用 Kolář, Špetlík, Matas, EEPPR, ICMV 2024 / SPIE 2025, doi:10.1117/12.3055033。基准不进入默认预测管线，不作为无人机飞行或姿态真值。

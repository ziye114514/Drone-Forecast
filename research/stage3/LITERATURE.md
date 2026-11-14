# 第三阶段相关工作核查（2026-10-16）

这轮实现现有思路作为强对照，不把“多桨投票”“椭圆不确定性”“时间先验”本身包装成创新。

- Mariyanayagam等，*Pose estimation of a single circle using default intrinsic calibration*（2018）：已标定单圆存在双解；讨论缺标定及近似内参。支持明确区分“候选里有正确解”和“实际选对解”。<https://arxiv.org/abs/1804.04922>
- Cai和Wu，*Camera calibration with coplanar conics: a unified explanation and ambiguity analysis*（2018）：讨论多共面二次曲线、单应与歧义，并依赖额外约束；不能推断任意多个圆在所有退化条件下都唯一可辨。<https://link.springer.com/article/10.1186/s41074-018-0050-y>
- *Level Set-Based Camera Pose Estimation From Multiple 2D/3D Ellipse-Ellipsoid Correspondences*（2022）：将预测椭圆不确定性用于多对应姿态估计加权；任务与未知四轴不同，但“不确定性加权姿态估计”已有先例。<https://arxiv.org/abs/2207.07953>
- *Arc-support Line Segments Revisited*：支持从部分弧构造椭圆假设，残弧检测本身不是新问题。<https://arxiv.org/abs/1810.03243>
- *Matching 2-D Ellipses to 3-D Circles with Application to Vehicle Pose Estimation*：圆特征反推车辆姿态也有既有研究；需要对象表示/对应等前提。<https://arxiv.org/abs/0912.3589>

可查验代码：

- `affanadly/Ellipse-Fit-Uncertainty`已有椭圆参数误差与置信区间工具，面向其他观测任务；本轮未运行或复制该实现。<https://github.com/affanadly/Ellipse-Fit-Uncertainty>
- `walkfish8/Riemannian_Circle`已有圆锥初始化与双目联合圆姿态优化；本项目只有单事件相机，不能照搬其双目证据。<https://github.com/walkfish8/Riemannian_Circle>
- `tobycollins/IPPE`讨论共面点位姿双解与退化；不是本轮圆锥算法，但可作歧义处理参考。<https://github.com/tobycollins/IPPE>
- `radimspetlik/helixtrack`已有旋桨相位与位姿联合跟踪。本轮不将简化候选共识称为完整HelixTrack复现。<https://github.com/radimspetlik/helixtrack>

本轮代码独立实现：解析反投影候选沿用已冻结的本项目函数；最多16种候选组合的共同法向、基础离群抑制、连续块删除敏感性与时间先验是标准构件。需先证明在冻结实验和真实RAW上有效，才讨论后续贡献。

真实RAW尚无内参或独立姿态真值。已知标定的仿真结论不能挪用为真实姿态精度；没有内参时只验证像素轴向融合/置信度门控对二维预测的影响。

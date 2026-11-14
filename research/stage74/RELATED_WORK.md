# 双分支互补：先查资料，再实现（2026-10-21）

- **Blom / Bar-Shalom, 1988, IMM**：在多个运动假设之间传播模式概率并交互混合状态；不是两个同源观测当作独立传感器相乘。[原论文 DOI](https://doi.org/10.1109/9.1299)。
- **FilterPy IMMEstimator**：[作者源码与说明](https://filterpy.readthedocs.io/en/latest/_modules/filterpy/kalman/IMM.html)、[GitHub](https://github.com/rlabbe/filterpy)。实际复用固定提交 `3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33` 的完整IMM状态混合/协方差/模式似然算法。MIT许可原样保留在 external/filterpy_imm_upstream/。环境未安装FilterPy；运行时只替换显示辅助 pretty_str 的导入，不改IMM数学。滤波器接口与物理控制适配器由本项目提供。
- **Event-Only Drone Trajectory Forecasting with RPM-Modulated Kalman Filtering, 2026**：[作者论文](https://arxiv.org/html/2603.01997v1)。III-B把转速线索用于调整CV滤波过程噪声；实验的当前位置输入来自真实框。与本项目自动检测框输入的完整RAW预测不能直接比较论文分数。本轮未在作者论文页/检索结果找到可确认的作者代码仓库，不宣称绝对没有代码。
- **本地 Target_KF / 旧14 motion.py** 已有CV、CTRV与模式概率参考，但旧14 IMMFilter没有完整不同状态空间的交互混合。本轮用作者FilterPy完整IMM，所有模式统一 `[x,y,vx,vy]`，以外部加速度控制表示物理/历史机动假设，不重新写IMM。
- [MathWorks trackingIMM](https://www.mathworks.com/help/fusion/ref/trackingimm.html)也明确提供CV/CA/CT模式组合；只作官方参考，未调用MATLAB。

标准框架不是新创新点。待验证点是桨盘方向/活动是否提供早于框运动的可用机动信息，以及可靠性门控能否在实采数据上保留这部分收益。急转时物理分支不一定更好，必须与CV、强历史预测、相同高过程噪声但无事件控制、恒定输入对照区分。

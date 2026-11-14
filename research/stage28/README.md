# 阶段28：轴的时间符号与传播

冻结axis28/checks28/study28，不改旧paper_physics.py。四轴方法original/no_rate/canonical_static/instantaneous，共1699时刻；original完整回归27，各分支六历史模型和有效性不变。

结果`D:\drone forecast\output\stage28_axis_propagation_20261018\第二十八阶段轴向符号与传播报告.html`。侧视向下先验冲突110→0，no_rate仍110；0.4s FDE129.13→126.55px，CV127.62，仍很小收益。新方法沿用图像向上先验，不能用于任意相机滚转/倒飞，不是姿态真值已解决。查了libDirectional/S²滤波，未执行其算法，实际只是复用原每帧测量与协方差的消融。

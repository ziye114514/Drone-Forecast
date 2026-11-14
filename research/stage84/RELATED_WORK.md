# 84：相同未来投影的批量矩阵运算

80独立profile显示，每个滤波银行都逐个构建26组F/B并投影均值/协方差，造成明显Python循环开销。复用[NumPy matmul矩阵堆叠规则](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html)，一次构建固定26个时距矩阵，沿时间批量计算；双模式加权顺序保持。

FilterPy作者IMM/MMAE的状态预测、交互、概率和更新完全保留；84只更换本项目74/77自己编写的只读未来投影函数。不是重新实现作者滤波器，也不是新预测算法。80/82优化保留、不减49路、不改时间窗或推力学习。

预先限定绝对误差1e-8、相对容差0，只允许未来rollout_positions、rollout_position_covariance和发布的预测坐标存在微小浮点差异；当前位置/速度/协方差、模式概率、控制输入、学习状态和准入必须精确。所有变动路径和最大差异保存，不以全树allclose掩盖状态变化。若数值能精确一致则如实报告。

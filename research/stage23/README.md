# 阶段23：稳健加速度预测

same-input模型消融，无检测更改。native20名义0.4s侧视，原历史加速度均值293.03/最大2720.76px，Huber64.66/189.85；native20清晰收缩+限幅18.87px对CV24.66。native10清晰仍失败，不能统一升级。

冻结robust23/checks23/study23，执行SciPy Huber和原阶段22预测器，复用阶段22端点评分。7000条原模型记录逐项相同，所有模型有效时刻相同。协方差收缩是未校准启发式，3000px/s²非物理真值。

[报告](</D:/drone forecast/output/stage23_robust_acceleration_20261018/第二十三阶段稳健加速度报告.html>)；[阶段结论](</D:/drone forecast/output/stage23_robust_acceleration_20261018/阶段结论.md>)。开发数据非盲测，保留CV/hold/全部失败，无G/yolov10-main访问，旧默认保持。

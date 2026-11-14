# 阶段22：未来端点与存档框对照

将阶段21定位流接原CV/历史加速度和hold。全部预测系数先保存，随后才读标签；在T−H前最近锚点，精确计算T处端点，实际H至H+25ms。稀疏框可以如此评FDE，不能评完整连续ADE；更正旧文档的过度绝对说法。

native20清晰0.4s CV24.66px、hold53.03；侧视CV131、原历史加速度293.03（最大2720.76）。覆盖与最差例均保留。预测有效性5次测量、160ms停止/500ms重置。旧模型“强基线”结论此前只针对检测器未来输出，不能外推为真实框精度最优。

冻结predict22/checks22/study22，report22只作事后展示。[报告](</D:/drone forecast/output/stage22_sparse_endpoint_20261018/第二十二阶段未来端点评测报告.html>)；[结论](</D:/drone forecast/output/stage22_sparse_endpoint_20261018/阶段结论.md>)。框/时间近似，非质心物理真值；开发数据非盲测。旧默认保持，无G或yolov10-main访问。

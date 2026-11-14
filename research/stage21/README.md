# 阶段21：原生时间窗与过去速度补偿

35秒四分支：native20/10/5和warp20。10ms在train4联合通过5/5，而20ms为2/5；清晰覆盖却8/8→7/8。5ms出现明显错位，补偿也损失覆盖，无统一赢家。

冻结`windows21.py`、`checks21.py`、`study21.py`不改写；`report21.py`仅后处理。1750个native20观测与阶段19邻域基线逐项相同；四分支前缀一致。只是过去速度平移，不是完整CMax优化；不使用未来速度。

[报告](</D:/drone forecast/output/stage21_native_windows_20261018/第二十一阶段原生时间窗报告.html>)；[阶段结论](</D:/drone forecast/output/stage21_native_windows_20261018/阶段结论.md>)。后续已接存档框端点评分（阶段22）。旧默认保留，无G/无yolov10-main访问，无RPM/姿态/未来精度提升声明。

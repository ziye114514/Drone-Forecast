# 109—111：提取链路、合并保护与求解稳定性

这是研究模块实验，尚不是新的完整 RAW-only 部署入口。完整入口仍为 `run_center_bias_forecast.py`（107）；默认54、API93不改。

109重新读取自采RAW，记录原生ON/OFF→双极活动→阈值→开/闭运算→轮廓→分组。在555个20ms时刻，记录操作没有改变原geometry20。C112.02秒的两个左侧分量被3×3闭运算合并，是漏提取的一个可复现原因。

110对原闭运算（先判线控制）、全局取消闭运算、条件保留合并前合格分量进行固定对照。实际复用OpenCV形态学、现有PaperPerception/Recovery/先判线、104共面圆拟合和107中心滤波。无G、无强制四点、无隐藏桨心。AAMED仅核查作者仓库，本轮未编译运行；距离变换只作诊断图。

新几何来自RAW；下游使用既有完整RAW结果中同一时刻的机体框与侧影轴，以保持三路输入一致。四段共1155时刻，两个独立截断前缀各60时刻，共1275次20ms读取；每个时刻比较三路。侧视与连续段重叠，不是独立录像。真实RAW前缀严格一致（排除耗时字段）；每帧body-only KF三路严格一致；负例无初始化。

连续段四候选85→128，合格拟合10→17，但仍0初始化。C恢复四个实际轮廓后卡在80次求解预算。清晰段条件保护与闭运算控制预测相同（0.4s固定偏移KF16.843px），全局取消闭运算18.004px，原完整融合13.708px。此处是7/8个稀疏未来框中心，不是COM真值；不能将16.843相对107的变化归功于条件保护，因为本轮对照相同，几何输入改为了20ms。

111只对连续段merge_safe的44个“优化未收敛”时刻，将预算80→320作诊断，其余门槛不动，原源码未改。结果见独立`output/stage111_conic_budget_diagnostic_20261022`。数值收敛不等于姿态/物理中心正确，不因单帧变合格而降低五帧建模门槛。

复现命令（对应输出目录必须不存在；保留旧结果，重做需在脚本中另选输出路径）：

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast/research/stage109/audit109.py freeze
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast/research/stage109/audit109.py run --case continuous6
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast/research/stage110/study110.py freeze
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast/research/stage110/study110.py run --case continuous6
```

完整110固定方案还要求clear6、side6、negative5和clear6/continuous6实际prefix。全部完成后运行`delivery110.py`；`finalize110.py`加入111诊断、更新交接并存档。请先读冻结协议、状态和工作日志，不能直接覆盖已完成目录。

本轮是观测改进与失败定位，尚未形成已证实创新。下一步应检查真实支持弧段、模型参数边界/不确定性，再验证姿态变化与真实平移分离；不能仅增加候选或求解次数宣称预测成功。

# 四点透视模型实验

主输出：`D:\drone forecast\output\stage5_four_point_projective_20261016`。

本轮要求四个实际椭圆观测才初始化、更新骨架。缺点时只短时外推，不用补出的点当观测。真实 RAW 无内参，输出二维单应性及归一化形状；OpenCV IPPE 只在已知真实 K 的模拟中运行。未升级主程序默认行为。

在项目根目录，用已有 Python 3.11 / Metavision 环境执行：

```powershell
& '..\.venv-event311\Scripts\python.exe' research\stage5\study.py tests
# freeze/raw拒绝覆盖旧结果。另开实验须使用新输出目录并重新冻结。
& '..\.venv-event311\Scripts\python.exe' research\stage5\study.py freeze
& '..\.venv-event311\Scripts\python.exe' research\stage5\study.py synthetic
& '..\.venv-event311\Scripts\python.exe' research\stage5\study.py raw --case clear
& '..\.venv-event311\Scripts\python.exe' research\stage5\study.py raw --case side
& '..\.venv-event311\Scripts\python.exe' tools\stage5_official_detector.py
& '..\.venv-event311\Scripts\python.exe' tools\stage5_temporal_shape.py
& '..\.venv-event311\Scripts\python.exe' tools\stage5_report.py
```

`external/evperiscope` 下载自官方 MIT 仓库，固定提交 `a2e6f2208bf12ffbe8663d2ea01d3f32f93c2bbc`。只导入 `scripts/detect.py`，核心源码未修改；没有运行任何飞行控制或ROS消息发送。归档保留源码与许可证。适配ROI输入的对照与官方全图模式分别报告。

`tools/stage5_temporal_shape.py` 是发现首次形状冻结过早后设计的独立因果跟进，不是第一轮冻结协议的一部分。只重放 RAW 派生观测，人工框仅由报告程序事后读取。

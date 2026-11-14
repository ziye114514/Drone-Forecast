# 整机侧影与二维推力轴实验项目

输出：`D:\drone forecast\output\stage6_body_axis_v2_20261017`。

初次四点建立等效机架，此后机体事件支持和真实线段可以独立更新侧影方向；不要求每帧有四桨。仅在近侧视、弱/近似正交投影条件下解释为二维推力轴。没有真实内参、箭头符号或推力大小测量，不输出校准三维姿态。

独立项目入口，在 `D:\drone forecast\event_drone_forecast` 运行：

```powershell
& '..\.venv-event311\Scripts\python.exe' run_body_axis.py --raw 'E:\dataset\train\6\media\event.raw' --output '..\output\my_body_axis_run' --duration 30
```

只需 RAW。未遇到清晰四点前输出未建模；命令行默认从0秒启动，便于自动获取模型。输出 `observations_forecasts.jsonl` 含位置、方向观测来源、是否已有模型、各预测器0.1/0.2/0.4秒图像位置与不确定性说明。

实验包括未出现0–2秒、清晰20–30秒、侧视与离场108–116秒。保留同一RAW单无人机模型，跨片段重置动态状态；这是模型保持/再捕获对照，未声称连续追踪未测试时段。

```powershell
& '..\.venv-event311\Scripts\python.exe' research\stage6\study.py tests
# 冻结和RAW命令拒绝覆盖既有结果；新实验应另设输出并重新冻结。
& '..\.venv-event311\Scripts\python.exe' research\stage6\study.py freeze
& '..\.venv-event311\Scripts\python.exe' research\stage6\study.py synthetic
& '..\.venv-event311\Scripts\python.exe' research\stage6\study.py raw
& '..\.venv-event311\Scripts\python.exe' tools\stage6_report.py
```

相关论文/开源的完整核查在报告中。此次为独立二维缩减实现，不是DronePose、M3T或事件6DoF跟踪器的完整复现。所用OpenCV LSD实际执行。原src与此前协议未修改，本入口为研究用途，未替换默认程序。

初次开发检查遇到OpenCV 5的LSD输出形状兼容错误，在RAW实验前修复为统一 `reshape(-1,4)`，旧协议目录保留；v2协议重新冻结，所有有效RAW结果均来自v2。

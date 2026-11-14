# 整机骨架可行性小试验

输出：`D:\drone forecast\output\stage4_skeleton_pilot_20261016`。

本轮保留旧 src 与之前阶段协议不变。参考 EVPeriscope / DroneKey / LOPET / HelixTrack，独立实现二维仿射骨架与机身事件 LK 辅助；未宣称完整复现这些算法。输入仅原 RAW；自动结果不使用人工标签。人工四点初始化诊断在独立目录中，不能作为 RAW 自动预测成绩。

在项目根目录使用 Python 3.11 与已安装的 Metavision SDK：

```powershell
& '..\.venv-event311\Scripts\python.exe' research\stage4\study.py tests
# 以下 freeze / raw 命令拒绝覆盖已冻结的本次结果。
& '..\.venv-event311\Scripts\python.exe' research\stage4\study.py freeze
& '..\.venv-event311\Scripts\python.exe' research\stage4\study.py synthetic
& '..\.venv-event311\Scripts\python.exe' research\stage4\study.py raw --case clear
& '..\.venv-event311\Scripts\python.exe' research\stage4\study.py raw --case side
& '..\.venv-event311\Scripts\python.exe' tools\stage4_mechanism_probe.py
& '..\.venv-event311\Scripts\python.exe' tools\stage4_seed_diagnostic.py
& '..\.venv-event311\Scripts\python.exe' tools\stage4_report.py
```

再次试验请复制本阶段到新目录并更改输出目标，重新冻结协议；不要修改已记录的旧协议。该原型仅用于定位初始化、关联、形状与运动预测的瓶颈，不是主程序默认升级。

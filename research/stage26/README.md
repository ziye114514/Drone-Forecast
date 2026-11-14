# 阶段26：无周期增强的自然布局消融

冻结`structured26.py`、`study26.py`及根`run_structured_forecast.py`，不要编辑后覆盖结果。四段35秒/1750时刻，neighbor20当前和所有六预测器严格回归阶段21/23。layout20只有阶段19布局变化。

结果：`D:\drone forecast\output\stage26_natural_layout_20261018\第二十六阶段真实布局消融报告.html`。自然侧视覆盖13/14保持，中心误差27.97→28.81px，0.4s收缩预测71.32→74.77px，不推广默认。受控删减收益和自然泛化分开陈述。

运行新入口示例：

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_structured_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --start 22 --duration 2 --output 'D:\drone forecast\output\my_layout_prediction' --preview
```

输出目录须不存在；仅RAW是传感器输入。预测仍用位置历史，`--rotor-features`只增加诊断。`resume26.py`仅负责检查已完成片段并补空负例；未覆盖已完成数据。

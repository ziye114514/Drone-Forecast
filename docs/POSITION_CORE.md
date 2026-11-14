# 可安装的位置运动核心（0.3.1）

当前live通道的状态更新和外推已迁入`src/event_drone_forecast/position`。它直接使用当前实现的107 KF、23稳健加速度、144阻尼及154曲线外推，算术没有重写或调参。根目录`live_motion.py`、`live_measurement_kf.py`和`live_covariance.py`仅保留兼容导入。

```python
from event_drone_forecast.position import LiveMotion, predictions

motion = LiveMotion(tracker_hz=25.0)
body = {
    "detected": True,
    "center": [320.0, 240.0],
    "bbox": [280.0, 220.0, 360.0, 260.0],
    "confidence": 0.9,
}
coefficient, valid, diagnostic = motion.step(0.02, body)
preferred, constant_velocity = predictions(coefficient, age=0.03)
```

连续输入的`t`严格递增，单位为秒，坐标为像素。`preferred["0.4"]`对应观测截止时刻加`age+0.4`秒。初始化数次后才可能有效，第一条观测不等于有效预测。未检测时传`{"detected": False, "center": None}`，不能把惯性外推框当新检测。

位置API不导入Metavision、CUDA、Ultralytics或历史research脚本；需要已有NumPy和SciPy。完整工程的依赖声明仍兼容旧CLI，因此包含OpenCV/Pillow。`pip install .`或本地wheel现在安装0.3.1包；原控制台命令仍是历史流程，**完整当前RAW运行入口依然是源码根目录`forecast.py`**，不要混作全链已打包。

## 本机验证范围

在迁移前冻结四段实际观测，legacy/continuous两种噪声模型共632次更新。源码包、移出原目录的wheel以及实际pip离线安装到新目录后的包：全部系数、预测坐标、有效性和诊断逐项完全相同。迁移检查会禁止读取原research/external源文件，并确认没有导入SDK。

本地wheel约70KB、33个文件，随附未改动的固定FilterPy作者源码、快照和MIT LICENSE。核心从包内资源加载原作者函数；不依赖用户另装FilterPy或原目录布局。历史外部快照仍保留，用于旧研究复跑。

之后真实侧视RAW1.2秒，25次跟踪、6次发布、其中3次有效；原生65片含0.1秒预热，源指纹保持、共享帧校验通过。该次观测年龄P95约231ms，过旧条目无效，不能称连续有效5Hz或比上一轮更快。它是接线检查，不是新的精度或性能基准。

完整RAW入口仍需要SDK、CUDA工作进程、源码布局和兼容模型权重。这里复用了本机已有依赖，并非另一台干净机器上的完整安装。结果目录为`output/portable_position_core_20261025`，构建/安装日志、wheel SHA、冻结观测和迁移检查均保留。

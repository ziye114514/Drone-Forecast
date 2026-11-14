# 阶段30：真正RAW姿态/转速/在线混合入口

新根入口`D:\drone forecast\event_drone_forecast\run_adaptive_physics.py`冻结，勿修改后覆盖研究结果。只有RAW输入，原生10ms冷启动，实际接入频率/轴/投影推力与到期误差混合；与旧24/26仅频率诊断的入口区分。

结果`D:\drone forecast\output\stage30_raw_adaptive_physics_20261018\第三十阶段RAW自适应物理预测报告.html`，两个MP4/CSV/完整15模型JSON。重解码清晰200/侧视500时刻与阶段12/29完整观测、频率、预测精确相同。处理仍不实时，700视频帧全部核验。

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_adaptive_physics.py' --raw 'E:\dataset\train\6\media\event.raw' --start 22 --duration 2 --output 'D:\drone forecast\output\my_adaptive_physics' --preview
```

输出须不存在；start/duration10ms整数倍。所有15模型始终保存，`--primary adaptive_history_mix`可显示纯历史在线混合。默认全模型研究混合，不保证更好，物理真实参数不可从像素增益直接识别。

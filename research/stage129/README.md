# 129：禁止视野外拟合参与完整桨盘初始化

2026-10-24视频QA发现：128新增8次当前占据模型更新中，22.68—22.78秒6次的拟合椭圆越出传感器。原代码却标记完整可见，不能把提前22.68秒初始化当可靠收益。

`boundary129.py`计算旋转椭圆解析包围边界，超出像素中心FOV的拟合保留为当前部分观测（arc、inferred_shape），可见弧比例未知。原104/119拒绝其完整四/三桨资格，五帧门槛不改。规则不代表物理桨盘真值，也没有识别实际遮挡。

124同期几何/NN回放1505时刻，短前缀严格。清晰原计数236、未守门独立模型244、守门235（计数233/占据2），首次22.86、占据学好22.90。原更早建模收益撤回，其余正例仍0首次模型。逐个未来端点见scores129.json，属于评分后的开发诊断，不是新独立评测。

新RAW入口：`D:/drone forecast/event_drone_forecast/run_FOV_guard_forecast.py`。129在实际128 worker相同进程内加启动与响应前守门，不再多加一层SciPy进程；原71路保留，新增support_FOV_guard_kf129。追加实际RAW仅60正例＋30负例，原71精确128、守门精确129缓存。没有声称重新完成8秒端到端测试，完整RAW790仍是128。

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_FOV_guard_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --start 22 --duration 2 --output 'D:\drone forecast\output\my_guarded_demo' --preview --verify-frames
```

输出必须不存在；实验明显非实时，50fps仅预览回放速度。默认稳定54/API93不改。只用RAW与已有权重，无G、无新依赖、yolov10-main未访问/修改、E盘只读。

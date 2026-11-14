# 阶段24：RAW研究流水线

新增根入口`D:\drone forecast\event_drone_forecast\run_research_forecast.py`，一份RAW即可产生当前桨盘/整机观测、六种因果预测模型和可选原逐桨频率诊断；新入口默认native20+trusted_acc显示，旧run.py默认保持。

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_research_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --start 22 --duration 2 --output 'D:\drone forecast\output\my_research_prediction' --preview --rotor-features
```

输出目录须不存在，start/duration为20ms整数倍。频率是诊断，不输入本入口预测器。去掉--rotor-features可省相应计算。--primary cv保留基线，其他模型/窗长可显式选择。所有预测均保存，视频按源50fps不是实际吞吐。无质量/尺寸/风速/推力参数输入，也无真实物理参数恢复声明。

冻结study24及根入口/依赖后，五段20秒/1000时刻完成；清晰100观测/预测与阶段21/23逐项一致。额外train3仅3/12有机标签覆盖，未标注日夜/黄昏只能定性看。性能仍未实时。

[报告与视频](</D:/drone forecast/output/stage24_raw_pipeline_20261018/第二十四阶段RAW研究流水线报告.html>)；[阶段结论](</D:/drone forecast/output/stage24_raw_pipeline_20261018/阶段结论.md>)。代码已由stage25指纹引用，勿改写后冒充同次结果。原src保持，未用G/未访问执行yolov10-main。

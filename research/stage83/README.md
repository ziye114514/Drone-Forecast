# 83：保留49路的流式取样RAW入口

`run_stream_forecast.py`继承81完整RAW、FP32 NN、10ms桨叶worker和原78数值模型；82只将等价的全局索引取样提前到大窗口拼接前，80缓存/预测映射优化保留。不启用79 SDK新频率、不调阈值、不少算模型。输出目录须不存在。

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_stream_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --output 'D:\drone forecast\output\my_stream_trial' --start 108 --duration 5 --preview --verify-frames
```

实际完成以checks83.json为准。视频50fps为源时间；数学/预测精确78，所以不是新准确率成果。默认54和旧78/81入口保留。83复用81的完整检查函数，在独立namespace只改路径/入口/阶段元数据与输出文件名；不修改旧源。完整RAW推理阶段不读缓存，推理结束后才与78保存结果作回归比较。

首版完整清晰100帧推理完成、数值精确78，但包装在审计钩子命名空间被清理后写元数据导致NameError，整批停止。失败源/原协议/事后检查保留在stage83_RAW_streamed_window_20261021；修正版结果在stage83_RAW_streamed_window_v2_20261021，必须重跑全部检查，不复用失败批次冒充完成。

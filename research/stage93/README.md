# 93：RAW互补历史实验入口

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_complementary_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --output 'D:\drone forecast\output\my_complement_trial' --start 108 --duration 8 --preview --verify-frames
```

输出目录必须不存在。仅原生ON/OFF；固定检测权重和模型常量与既有85相同，不用G、不读标注/推理缓存。原49路保持，新增训练FRED选择的sigma6400运动KF和五个互补路由/固定对照，共55路。

`complement_fallback`在原物理输入可靠时保留原主线，否则用CV/可信历史/训练KF的延迟专家权重。0.2s后只用新检测位置反馈，未使用独立未来真值。`complement_full`始终融合原主线与三条历史候选。所有旧数值、坏结果、默认54保留。新回退侧视仍有退步，未推广为默认。

`forecasts.mp4`保留旧可视化；实验工具study93另生成`complement_forecasts.mp4`，青线新回退、黄线旧主线。按源50fps播放不代表运行50Hz。端到端仍慢。完整检查见output/stage93_RAW_complementary_20261022/checks93.json：1240父/2480原生、1100视频帧，三完整段和两个独立实际RAW前缀，旧49/当前精确78、新6及路由学习状态精确92。

原`raw_forecast_api.py`仍隔离调用85；本轮新增93 CLI，没有改旧API的默认行为。94—97为当前官方HDF5事件＋过去标注框的预测层/几何诊断，不能混称93的RAW-only部署精度。只输入RAW不等于已识别机械RPM、质量、风或完整姿态。

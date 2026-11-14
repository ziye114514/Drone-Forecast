# 状态隔离与未来输入刷新（76–78）

最新结果以 `output/stage78_RAW_separated_forecast_20261021/checks78.json` 和评分为准；仅有源文件/协议不代表RAW批次已完成。

新入口 `run_separated_forecast.py` 保留原39路，再加77十路结构对照：未来历史输入刷新、物理缺失回强历史，以及IMM/MMAE、区间控制/仅未来修正、事件/匹配恒定输入。最新当前轴和周期同时支持才接入物理输出；其余仍预测，按规则回强历史，不通过少输出美化误差。

“当前支持”指本次过去160ms窗口产生新周期测量，不代表零延迟真实转速。状态与未来控制分离不保证最终加权预测总更好。

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_separated_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --output 'D:\drone forecast\output\my_separated_trial' --start 108 --duration 8 --preview --verify-frames
```

输出目录须不存在。只输入RAW与固定检测权重；视频蓝色原CV、紫色旧IMM、黄色mmae_state_fresh。无当前物理支持时黄色按规则回历史，不等于机械RPM测量或事件信息已成功。

复用FilterPy同一固定提交的作者IMM/MMAE，见77 RELATED_WORK.md。原始MMAE向量混合协方差问题由明确的外部适配器修正，原源/许可未改。独立子滤波CV不会被另一子模式错误控制改写；仍不保证最终融合总更好。

77开发片段结构改进明显，但恒定输入对照几乎同样好。默认仍54，未证明持续高精度实时；现阶段优先分辨结构收益与事件信号收益，再看另外三段固定录像的结果。

# 81：精确计算优化的完整RAW入口

`run_exact_forecast.py`的预测数学/49路与78一致；80减少重复背景频谱计算及旧预测字典深复制。只输入RAW与原固定检测权重，不使用缓存检测/评测标签，FP32。原78入口/默认54保持。79的SDK候选未启用。

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_exact_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --output 'D:\drone forecast\output\my_exact_trial' --start 108 --duration 5 --preview --verify-frames
```

输出目录须不存在。视频蓝CV、紫旧IMM、黄mmae_state_fresh，黄色缺少当前物理输入时回纯历史。视频50fps是源时间，不是运行吞吐量。数学不变所以不应声称新准确率改善；完整验证以checks81.json和实际耗时为准。只存在代码/协议时不能说复现已结束。

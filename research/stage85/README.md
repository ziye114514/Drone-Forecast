# 85：流式取样及批量投影的完整RAW预测

`run_batched_forecast.py`在83修正后的完整入口上接84批量未来投影，保留80/82与全部49路。预测数学、滤波更新、学习、阈值不变，79新SDK周期未启用；默认54及旧入口保留。

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_batched_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --output 'D:\drone forecast\output\my_batched_trial' --start 108 --duration 5 --preview --verify-frames
```

目录须不存在，输出JSONL和源时间50fps视频。是否已完成以checks85.json为准；全49/当前与78精确时才能称保持预测。完整耗时看实际表，模块加速不是端到端实时。蓝CV、紫旧IMM、黄mmae_state_fresh；无当前物理支持黄回历史。不能把速度工程改进当作RPM信息增益或创新性成果。

当前入口按独立CLI进程运行。内部继承的推理审计钩子属于该进程，收尾会清理动态命名空间；不支持把run当成同一常驻Python进程可反复调用的普通库API。83首版外层收尾失败正是此生命周期问题。接入常驻服务/异步调度时应先隔离子进程或改造审计生命周期，再做对应实际测试。

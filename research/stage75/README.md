# 框CV与事件物理分支并行交互

用户要求保留原始框位置/速度分支，在平稳飞行与突然变向时和物理分支互补。74先查论文/作者库并固定复用FilterPy完整IMM；75将其接到真正RAW入口。最新结果须看实际 `checks75.json` 与74评分；不能仅凭本文件存在宣称正式RAW批次已结束。

两个分支统一状态 `[x,y,vx,vy]`：CV模式假设匀速；机动模式使用先前学到的有效推力系数和当前可用的事件特征，或历史加速度加事件推力增量。每个假设接受同一个当前位置一次；作者IMM负责状态/协方差交互及新观测似然更新。标准框架不是新的创新点。

`run_hybrid_forecast.py` 保存全部39路预测，包括原33路、CV+高噪声但无事件输入、历史加速度、恒定输入和事件物理控制的对照。视频蓝色CV、紫色原事件增量预测、黄色双分支IMM增量融合，并显示物理模式概率。它是实验入口；默认 `run_deploy_forecast.py` 保持。

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_hybrid_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --output 'D:\drone forecast\output\my_hybrid_trial' --start 22 --duration 5 --preview --verify-frames
```

输出目录须不存在。推理只接受RAW和固定检测权重；不能读旧观测/特征/标签。`observations_forecasts.jsonl` 包含全部预测、IMM概率、分支加速度及协方差；视频按源时间播放，不代表处理实时。

实际复用：[FilterPy作者固定提交](https://github.com/rlabbe/filterpy/tree/3b51149ebcff0401ff1e10bf08ffca7b6bbc4a33)，MIT许可与原始源在 `external/filterpy_imm_upstream/`。没有安装FilterPy，仅适配显示用pretty_str导入，IMM核心原样执行。相关论文和边界见 `research/stage74/RELATED_WORK.md`。

当前缓存小试验仅清晰片段明显受益，侧视仍弱于强历史分支，3号变差；平稳/机动分组来自过去CV变化，不是独立姿态/机动标签。不存在已证明的持续高精度实时预测保证。

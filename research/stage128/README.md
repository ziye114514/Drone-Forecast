# 128：独立事件支持模型的实际RAW验证

2026-10-23开发/冻结，2026-10-24继续核验。实验数据均为历史开发录像，不是盲测。

入口：`D:/drone forecast/event_drone_forecast/run_independent_support_forecast.py`。
输入仅RAW与项目原检测权重；没有读取评估框、旧检测缓存、G重建或人工转速。旧122/126在独立子进程保留，新占据结构与count结构分别学习，当前几何进入共享位置滤波。新增`support_independent_kf128`，保留原70路作可核验控制；默认54/API93不改。

计划并核验：6号22—30秒400时刻、2号40—46秒300时刻、5号0—0.6秒30时刻、6号独立22—23.2秒60时刻。所有标签只在完整预测后评分。最终完成状态以每个raw目录checks128.json与delivery_checks128.json为准。

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_independent_support_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --start 22 --duration 2 --output 'D:\drone forecast\output\my_support_demo' --preview --verify-frames
```

输出路径必须不存在。运行较慢，50fps预览只是回放速度。该CLI用于实验，默认稳定接口没有升级为本版本。旧运行入口审计钩子是进程级的，始终用CLI子进程，不在要继续I/O的父进程直接调用run。

中断保留：2026-10-24检查发现第一次128进程已退出，最终文件核对91条完整记录，无completion（先前末次进度打印79时刻）；原目录与父日志保存为interrupted_natural6_20261024和interrupted_natural6_parent_20261024.log。重新按冻结方案独立运行，不拼接残缺滤波状态，不修改推理源码或阈值。

资源失败：扩展2首次运行已保存142条完整输出后，原73在线支持分配3.52MiB float32数组失败（之前末次进度打印75时刻）。输出与日志保存在failed_memory_extension2_20261024。系统提交空间偏紧，另有重型核验父进程及同期分析增加占用；改用仅标准库serial_runner128.py，预测及其worker退出后才启动独立核验，绘图放在全部预测后。没有改变线程参数、权重、推理源码、系统分页文件或其他应用；不能把执行资源错误当检测失败。是否完整成功以checks128.json为准。

研究依据：

- [Relative State Estimation using Event-Based Propeller Sensing](https://arxiv.org/html/2604.18289v1)：CC二值事件密度/形态学与聚类、桨盘椭圆及周期信号；这类基本步骤已有。
- [EVPeriscope](https://arxiv.org/html/2609.11920v1)：泄漏事件累积、非零像素mean+std阈值、形态学与k=4簇。已阅读本地作者detect.py；本轮没有直接运行其完整系统，也没有搬入固定四簇或已知CAD假设。
- [ELSDc](https://github.com/viorik/ELSDc)：有弧/线联合检测实现；本轮未编译或移植，现有先判线/椭圆路径已能作为对照。
- [Prophesee frequency map](https://docs.prophesee.ai/stable/samples/modules/analytics/vibration_py.html)：可复用的像素周期工具，本轮未接入，不声称新增RPM/推力收益。

下一步应验证像素时间周期如何区分转动桨盘与机体双极边缘；不能把占据面积、事件数直接当作转速/推力。仅当有可靠当前轮廓时再估计姿态，侧视/重叠不足四盘时继续保留缺失，不降低五帧门槛制造首次建模成功。

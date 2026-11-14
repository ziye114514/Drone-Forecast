# 阶段31：跨录像完整物理预测检查

冻结study31和阶段30根入口，直接跑train3 aligned25–31秒。此段在阶段24被探索过，称跨录像检查，不称盲测。预测文件全部保存后再执行原阶段22评分器，目标参考train3存档近似整机框中心。包含全部15模型，覆盖相同，不按本段结果调参。

结果`D:\drone forecast\output\stage31_cross_recording_physics_20261018`。完成标志为analysis.json和runtime_summary.json；HTML`第三十一阶段跨录像物理预测报告.html`，生成命令report31.py。若仅有片段输出/无completion，不要覆盖重跑，应先查原工具会话66994。

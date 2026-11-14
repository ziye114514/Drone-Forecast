# 第八阶段：纯 ON/OFF 桨盘提取

当前试验已完成。输出：`D:\drone forecast\output\stage8_onoff_rotors_20261017`，先读其中`阶段结论.md`或HTML报告。

用户最新要求：不使用重建G；不得修改、执行或在`D:\yolov10-main`内写日志。本阶段没有访问该目录，所有实现和结果留在本项目。旧预测器、阶段6/7源码及结果保留。

`polarity.py`复用原检测器的几何、分组、关联，只替换事件计数/双极特征输入。四种固定方法为原10ms、邻域10ms、过去30ms/3、邻域过去30ms/3。`PolarityFrontend(width,height,method).detect(events,t)`可用于新研究分支，输入须为连续、10ms的原生结构化事件批次，`t`为批次结束时间；变步长输入不在本轮验证范围。30ms启动不读取未来数据。

这不是原预测入口的默认替换。邻域10ms对清晰帧有小幅收益，侧视退化；30ms增加部分覆盖却产生定位偏移，FRED小目标没有改善。

在项目根目录复查已完成结果（不重新解码RAW）：

```powershell
& '.\.venv-event311\Scripts\python.exe' 'event_drone_forecast\research\stage8\summarize.py'
& '.\.venv-event311\Scripts\python.exe' 'event_drone_forecast\research\stage8\report.py'
```

`study.py all`不应对现有输出重跑：已有完成标记会拒绝覆盖。新推理改动应新建版本与协议，保存这次负结果。

原汇总函数因NumPy int64不能直接写JSON退出，六段推理均已先成功落盘；`summarize.py`仅在序列化时转为Python标量，未改冻结的`study.py`/`polarity.py`或重算推理。

三个清晰帧的参考中心由AI在无检测叠加图上近似放置，含约5px不确定度，只是辅助核对。整机框、候选数、选中四点都不能当作四电机真值。未评估真实RPM、姿态、推力或未来轨迹提升。

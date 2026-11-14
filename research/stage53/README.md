# 真实RAW联合入口

入口：`D:/drone forecast/event_drone_forecast/run_joint_forecast.py`。只输入一份RAW；程序使用固定的本地整机检测权重，内部同时运行20ms整机定位、10ms桨叶几何/逐桨频率、轴向/有效推力预测及在线参数与模型权重更新。

```powershell
& 'D:/drone forecast/.venv-event311/Scripts/python.exe' 'D:/drone forecast/event_drone_forecast/run_joint_forecast.py' --raw 'E:/dataset/train/6/media/event.raw' --start 108 --duration 5 --output 'D:/drone forecast/output/my_joint_prediction' --preview
```

输出目录必须未存在。`--verify-frames`增加父进程/GPU进程完整像素SHA核验；默认FP32，也可指定`--precision 16`，本阶段正式整合回归只验证FP32。16位模型此前有独立验证，但此入口的16位组合尚未做完整同样本回归。

输出`observations_forecasts.jsonl`中，先检查`frontends.delayed_feedback.valid`，再读取`frontends.delayed_feedback.predictions.adaptive_all_mix["0.4"]`等字段。全部0.1/0.2/0.4秒、六历史基线、四物理模型及五个组合模型都保存；另有原重置和固定参考两路完整对照。`rotor_ticks.jsonl`是内部每10ms的实际观测/频率输出，不是输入缓存。

视频绿色为历史收缩、紫色为轴向+频率模型；当`active=False`时，紫色分支回退至CV，不能把该曲线解释为仍有可靠桨叶/推力观测。当前框中心是可见外观中心，不是质心。输出为原生1280×720像素，未来点可以在画幅外；不输入或恢复真实质量、尺寸、风速、机械RPM与米制三维轨迹。沿用原图像向上/非倒飞先验，未验证任意相机滚转。

为保持旧几何的数值状态，20ms和10ms分支分别读取同一RAW，两读器都只处理到当前请求截止。独立SDK桨叶进程与GPU共享内存进程隐藏运行，结束回收；程序禁止读旧检测/频率/预测缓存和人工标签。GPU编译缓存及本地模型权重不属于检测结果缓存。

正式14秒700整机/1400桨叶时刻、另外短试跑和独立前缀已完成。全部975整机像素/观测及三路预测精确回归旧控制，正例1830个原生10ms几何/频率时刻精确旧34，独立RAW前缀225整机/450桨叶精确；540行真实未来评分精确旧51/52。没有新增精度提升，侧视仍只有10次学习，3号没有合格特征学习。

处理单步中位清晰225.5ms、侧视182.5ms、3号135.7ms、空场85.4ms，全部超过20ms；这里含三个完整物理实验对照。计时排除父进程下一RAW块/JSON/视频，wall含冷启动、双解码和视频。视频50fps为源时钟，不代表实时吞吐。旧默认和47/30入口均保留。

结果：`D:/drone forecast/output/stage53_raw_joint_physics_20261020/联合RAW预测结果总览.html`。`study53.py`支持freeze、run/audit --case、prefix、score、report；所有已完成输出拒绝覆盖，不能直接重新freeze/run到旧目录。

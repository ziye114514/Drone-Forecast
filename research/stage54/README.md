# 完整物理部署与单次解码

`D:/drone forecast/event_drone_forecast/run_deploy_forecast.py`保留最终完整15模型（六历史、四物理、五组合）和独立历史回归检查，去掉53中重复的两组研究对照。默认shared：只解码一次RAW，将完整原生事件字节、类型布局及截止时钟送给独立10ms桨叶worker。20ms整机前端/GPU和所有原物理算法不变。

```powershell
& 'D:/drone forecast/.venv-event311/Scripts/python.exe' 'D:/drone forecast/event_drone_forecast/run_deploy_forecast.py' --raw 'E:/dataset/train/6/media/event.raw' --start 108 --duration 5 --output 'D:/drone forecast/output/my_deploy_prediction' --preview
```

输出目录必须未存在；`--transport dual`保留同完整分支两解码器对照；`--verify-frames`增加图像和原始事件双端SHA；`--event-capacity-mib`配置有界事件缓冲（默认32MiB）。超容量失败，不降采样。正式54只验证FP32，16位组合需另测。独立几何进程、GPU进程和自有共享内存均在结束后回收。

输出字段沿用53，读取`frontends.delayed_feedback`。紫色轴向频率模型在active=False时回退CV。输出是像素动力学，没有真实质量/风速/机械RPM识别或一般相机滚转/倒飞保证。

实际RAW1265整机/2530桨叶时刻（含独立对照、试跑与前缀）全部当前观测、频率、15模型预测和生命周期精确53；全部225评分行保持。侧视单分支dual中位142.2ms/shared153.7ms，shared没有稳定单步优势；wall82.7→76.0s。仍超20ms，不能由50fps视频称实时。研究shared额外事件SHA和机器跨次负载影响计时。传输发布中位0.765ms，不是当前主要瓶颈。

原`study54.py prefix`要求跨独立解码全结构体字节SHA相同，已发现该要求包含无语义的填充字节（偏移6/7）。原失败保留；最终检查使用冻结补充脚本`prefix_padding54.py prefix`，先前已执行其freeze/diagnose。它核对全部科学数据和模型，传输内字节SHA不放宽。不要把原失败校验重新包装为通过，也不要修改原预测器来消除填充差异。

结果：`D:/drone forecast/output/stage54_single_decode_deploy_20261021/部署结果图与前缀校验说明.html`。下一步55尚未启动：复用17的事件ROI裁剪，以过去已到达的body-only框约束当前10ms几何；不能使用当前20ms图像给更早子帧，频率历史仍保持原始事件。先ROI关闭回归54，再ROI开启测同位置未来误差、空场与前缀。下一轮跨运行比较忽略非语义填充SHA，或另存规范化原生字段哈希；传输内完整字节校验保留。

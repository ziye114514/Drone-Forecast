# 101：93互补预测的独立进程Python接口

```python
from complementary_raw_api import forecast_raw

result = forecast_raw(
    r"E:\dataset\train\6\media\event.raw",
    r"D:\drone forecast\output\my_api93_trial",
    start=108.0, duration=8.0, preview=True,
)
print(result.observations, result.completion)
```

把event_drone_forecast加入Python导入路径；默认子进程使用项目.venv-event311，调用方只需标准库。SDK、推理审计和GPU只在子进程；返回后可继续父进程文件I/O。已有输出/日志不覆盖，非RAW拒绝，失败保留日志/部分输出。保护用户指定的yolov10-main路径。

一个干净父进程连续两次真实RAW，各12父/24原生时刻，当前观测与全部55、学习状态精确93前缀；每次返回后父I/O正常，导入不加载NumPy/OpenCV/Torch/SDK，非RAW与已有输出保护通过，见checks101.json。旧raw_forecast_api仍85，未修改。

这是同步进程隔离接口，每次含冷启动，不是实时/异步管线，也不提高原预测精度。

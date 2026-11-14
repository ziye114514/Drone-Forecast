# 87：调用方安全的RAW接口

`raw_forecast_api.forecast_raw`用独立子进程运行原85完整RAW。父进程只导入标准库，不加载SDK/ML，也不继承子进程的推理审计钩子；返回可用路径和完成元数据。解决83实际暴露的动态审计命名空间生命周期问题，不是异步/实时调度，也不改变预测模型。每次有原冷启动成本。

```python
from raw_forecast_api import forecast_raw

result = forecast_raw(
    r"E:\dataset\train\6\media\event.raw",
    r"D:\drone forecast\output\my_api_trial",
    start=108, duration=5, preview=True,
)
print(result.observations, result.completion["frames"])
```

输出目录和同名进程日志须不存在；失败保留日志和部分结果，抛出ForecastProcessError，不覆盖或删除文件。本版同步调用，无取消/后台接口。继续做有界队列/异步服务时还需实际时钟和延迟观测状态处理。

本机测试结果以checks87.json为准。一个父进程连续两次真实RAW，随后访问项目JSON；每次当前/49结果必须精确85对应前12时刻。只存在源码/协议时不能说测试完成。

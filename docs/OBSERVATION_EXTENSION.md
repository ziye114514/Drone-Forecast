# 可选观测扩展接口

`run_optimized_forecast.run(..., event_observer=observer)`允许独立传感特征作为旁路输出，默认`None`不运行扩展，也不改变已有预测。

对象实现三个方法：

- `protocol()`返回可JSON序列化的配置、依赖和代码指纹。
- `step(events, t, body, current_rotors, diagnostics, coefficients)`返回当前观测字典。events为当前原生20ms事件的只读视图；其余参数均为深拷贝。只能使用当前及过去信息。
- `close()`关闭线程或原生句柄，需能重复调用。

扩展输出写入`auxiliary_observation`，发生在运动状态和各预测坐标已经计算完之后。接口不自动把新特征送入动力学，扩展处理耗时计入`processing_ms`。异常传播并触发主入口资源清理。

本机HelixTrack适配器使用该接口，已从当下RAW+NN+四个当前椭圆自动初始化，独立输出方向假设、相位/当前像素周期一致性、当前投影与有条件的相对旋转活动量。后者不等于真实机械RPM或推力。作者代码与DLL不进入公开源码候选包，未满足可分发依赖前不能声称该可选适配器clone即用。

要把任意旁路特征变为预测输入，仍须单独实现过去到期误差反馈、同起点对照和缺失回退；观测的一致性增加不能替代轨迹收益证据。

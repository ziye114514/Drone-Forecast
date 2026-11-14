# 79：已有逐像素频率算法复用

- [Prophesee官方FrequencyAlgorithm接口](https://docs.prophesee.ai/stable/api/cpp/cv/algorithms.html)：逐像素闪烁频率；本轮实际调用已安装SDK，非重新实现。原安装和许可未改、不复制SDK二进制。Windows Python接口实查与所调用模块SHA保存在protocol。
- [Frequency Cam论文](https://arxiv.org/abs/2211.00198)、[作者项目](https://github.com/ros-event-camera/frequency_cam)：异步像素周期，已阅读本地C++与官方文档；本轮没有运行此ROS工程，也没有冒称其性能。
- [Relative State Estimation using Event-Based Propeller Sensing](https://arxiv.org/abs/2604.18289)：周期结合几何的研究动机；逐像素SDK结果不等于其真实转速或推力复现。

默认7次/1500us阈值来自实际安装接口，频率60–1200Hz沿用原工程；严格7次/200us和短3次/200us是事前固定的精度/时延对照，200us约为1200Hz最短周期的四分之一。比较原生流与严格7次的160ms过去速度平移回放。ROI只选当前结果，不用真值或未来位置。

全局照明也能形成完美周期。每像素一票、ROI/环带相同频带的空间密度比用于拒绝共同闪烁；这只是启发式，遮挡背景或相机移动时仍可能失败。保留Poisson、有色随机、突发、切频及整体闪烁控制的所有结果。没有机械RPM真值，不能用覆盖率代替准确率。

本机SDK `Event2dFrequencyBuffer.numpy()`存在dtype24B/stride16B不一致，copy=True也复制了错位视图。外部适配只依据_buffer_info给出的原生shape/itemsize/strides重建只读视图后立刻复制，数值与分块控制验证；不过滤掉坏行假装输出有效、不修改SDK。

首轮仅人工负控制中default7输出过1212–2553Hz等超出请求范围的值，严格范围断言停止，尚未跑真实RAW。失败协议/明细保留；第二版在池化前显式选60–1200Hz并记录排除数，重新冻结。此处过滤的是已正确解码的算法越界输出，和上面的错位内存问题分开。

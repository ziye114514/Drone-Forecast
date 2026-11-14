# 发布时效之后：已有异步/延迟观测方法

- [Stone Soup官方OOSM示例](https://stonesoup.readthedocs.io/en/latest/auto_examples/oosm/KalmanFilterOOSMExample.html)明确区分源时间和到达时间，并比较错误地当当前更新、固定延迟处理、丢弃延迟观测等方案。该示例假定已知固定延迟；不能直接声称它解决了本工程的可变NN/桨叶时延。
- [Stone Soup官方延迟观测例子目录](https://stonesoup.readthedocs.io/en/latest/auto_examples/oosm/index.html)包含重排、逆时间等路线，可作为下一步公开基线。这里只查阅，未安装/运行Stone Soup，未声称移植其实现。
- [Python Queue](https://docs.python.org/3.11/library/queue.html)的有界队列可限制积压，不能自动解决观测时间戳、丢包后的频率窗口、密度衰减和滤波学习问题。

86只对已保存实际processing_ms做乐观FIFO截止审计，不是实时摄像头评测。源码50fps可视化不能解释为运行50Hz；原0.1/0.2/0.4s误差是源时间精度，不能自动当实际发布后仍有同样提前量。

下一轮最低可审查方案：分别保存source_t、arrival_t、publish_t；有界队列和现有全FIFO同条件对照；在观测源时间更新并正确传播到发布时间；显式区分延迟的几何/周期和当前控制。必须重新因果运行RAW/截断前缀，验证跳帧/大dt下密度、频率缓冲、RLS队列与重置，保留输出覆盖/缺席失败，不能把既有预测按新时钟改名就宣称实时成功。

当前RAW入口以独立CLI进程为边界，推理审计钩子和动态命名空间是同进程全局生命周期；83首版已实际暴露返回后外层文件I/O故障。常驻异步服务应先做子进程隔离/受控生命周期处理，不能直接在一个Python进程反复调用这些旧run函数。

TQE的公开作者README称数据需向作者申请，不能在用户未授权发信时自行发送申请。本机HelixTrack仓库没有附TQE数据；源码可读不等于已得到独立RPM真值。现有EEPPR公共周期资料可作模块控制，但不能替代自采飞行片段的姿态/推力真值。

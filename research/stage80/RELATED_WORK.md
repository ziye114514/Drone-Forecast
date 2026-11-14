# 80：先定位再优化，保持输出精确

- [Python3.11 cProfile](https://docs.python.org/3.11/library/profile.html)：使用内置C性能剖析器，区分profile内的相对开销和无profile的墙钟基准。
- [Python3.11 functools](https://docs.python.org/3.11/library/functools.html#functools.lru_cache)：73已有相位载波缓存继续复用；本轮新增同一当前窗口内相同背景/频率的纯函数结果复用，不缓存跨窗口背景或未来事件。
- [Python3.11 copy](https://docs.python.org/3.11/library/copy.html)：完整预测字典中已生成的旧系数只读，新模型只添加键。新根/系数/预测映射各自复制，小系数、状态和延迟队列保留原深复制。

这是工程优化，不是新预测算法或创新性声明。保留49路，不通过关闭物理/少输出获得速度；原冻结源、阈值、周期窗口、学习参数保持。实际RAW当前观测/全部预测、模式概率/学习状态必须逐项精确78，历史行和输入不能被后续步回写。基准测试须等79批次结束，避免并行CPU争用；性能剖析结果只用于归因。

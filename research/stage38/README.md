# 阶段38：噪声/尺度的固定权重对照

冻结common38/render38/checks38/study38/predict38/score38；SDK活动过滤10ms、坐标整数重分箱d1/4/8、过去10/50ms，同40时刻480图。权重/conf不变，原尺度无过滤图像SHA与输出严格回归37。所有框还原原图，padding中心不收。

输出`D:\drone forecast\output\stage38_noise_scale_detector_20261018\第三十八阶段事件噪声与尺度迁移报告.html`。融合clear3/3、side3/14、train3 0/12，11无机0；但29有机最高框仅1个IoU≥.5，多为局部框，不升级默认。不报未来轨迹或全RAW实时。

执行：study38 freeze/prepare(SDK311) → fusion38 freeze(预测前) → predict38(Py313) → fusion38 run(真值拦截) → score38 → report38。40预存独立融合与评分器重算逐项一致。v1仅运行时真值文件哈希列表在准备/推理前已分离，原协议源码保留。所有算法参数始终未按结果更改。

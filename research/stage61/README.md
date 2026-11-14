# 61：先判线，宽轮廓才拟合椭圆

61是当前测量修复的实验分支，**完整未来评测尚未开始，默认仍54**。60旧完整预测已在严格回归停止、未评分；不能用其草稿交付脚本称已改善预测。

真实原因证据在stage60：同一11px共线点集，当前OpenCV5的fitEllipse依赖RNG流，重复输出约9.86–13.23px，能改变尺度门槛/三四点分组。当前5.0源码未核定，结论来自实际固定数组/重置RNG/消耗RNG干预，不冒用4.x源码。正式新推理不设置全局seed。

preline61仅向原PaperPerception._candidates AST唯一fitEllipse try前加入薄线分支，实际调用原geometry._line。太短/不支持的线拒绝，不回退病态椭圆。MRO实际复用原57/9/8候选包装、宽椭圆、残弧、分组与频率代码；已有LineFrontend60通过组合变为StablePairedFrontend，StableFrontend提供无配对基线。线段仍不是单桨身份或物理端点真值。

模块检查：真实共线轮廓不调用fitEllipse；11px线拒绝；20px长线中心正确；五帧宽椭圆完整前端精确旧路径；MRO确实拦截。实际RAW clear22–22.8、side112–112.8、train325–26.2、empty0–0.2，共300不同原生10ms时刻。两个独立随机历史normal/每tick额外消耗RNG，共600窗口；普通/配对分支的全部当前观测和审计精确。3号薄线38路由/18拒绝/20实测，side1/0/1，clear/empty0。75737退出0，图已实际查看。这只是小范围可重复性，不是实飞真值或未来精度。

输出：`D:/drone forecast/output/stage61_preellipse_lines_20261021/第六十一阶段先判线再拟合椭圆报告.html`。源与派生AST已冻结，SDK Python执行study61.verify()复核，run防覆盖。delivery61有独立SHA。

下一步完整61：新增独立worker/入口，将原60动态worker namespace中的LineFrontend注入StablePairedFrontend，仍用shared单解码和原15模型。line_mode=off应关闭配对但 **保留preline**，paired启用融合；不能退回CandidateFrontend作为所谓稳定基线。两路用同清晰2s/侧视5s/3号6s/空场1s，独立实际RAW前缀检查所有状态/命名事件字段SHA。整机NN/六历史/覆盖精确，旧桨盘几何因为短线修复允许变化。全部预测/回归通过后用原22稀疏未来评分，结果优劣都保留，才决定升级。需要锁定实际OpenCV二进制与构建指纹，不能仅锁Python源。

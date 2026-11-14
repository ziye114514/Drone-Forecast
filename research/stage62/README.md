# 62：方向与转速独立使用

目的：在缺桨/缺频率时保留可信方向信息，分别检验预测准入和在线回归准入。现有桨心仍由椭圆/残弧拟合计算。本阶段不新增检测器，不声称提高四桨提取准确率。

复用原27预测器、28方向观测、51固定频率参考/生命周期、52延迟RLS、29选择器和22稀疏评分。先查原论文 https://arxiv.org/html/2604.18289v1 与标准在线RLS https://www.mathworks.com/help/ident/ug/online-recursive-least-squares-estimation.html 。没有新依赖、外部完整系统执行或新颖性主张。

三路分别为`shared`原共同门槛、`axis_gate`只独立预测、`axis_learn`独立预测及回归样本准入。轴向使用axis.quality，不要求整机geometry_quality或四频率同时成立。新回归量必须有当前方向观测支持与整机位置；保留原0.15预测/0.45学习门槛、0.16秒过期、0.6秒预热、0.2秒延迟与全部RLS数值。旧转速分支及六历史逐项不变，回退仍CV，不能把改用历史回退当物理增益。

700当前缓存输入/三路2100步、原完整15及生命周期精确61，独立缓存前缀通过。540行评分：清晰0.4秒轴向15.736→14.004px，侧视122.894→113.917px，train3 136.693→138.262px（变差）；侧视历史81.913px、3号历史110.319px仍更好。侧视方向active84→175/250、实际更新10→142；3号active28→84/300、更新0→51。开发片段与稀疏框不足以声称泛化、真实姿态/RPM/风速精度。

实际RAW独立入口：

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' 'D:\drone forecast\event_drone_forecast\run_axis_forecast.py' --raw 'E:\dataset\train\6\media\event.raw' --output 'D:\drone forecast\output\my_axis_run' --start 108 --duration 5 --axis-mode axis_learn --preview --verify-frames
```

每个输出目录必须全新。紫色预览为轴向分支，绿色为历史；原轴向频率分支仍保存于完整15模型JSONL。默认部署入口run_deploy_forecast.py保持，试验入口并未成为精度升级。

正式RAW检查由`raw62.py`/`tools/run_raw62.py`执行：原14秒、共享门槛0.8秒清晰控制、侧视4.5秒和3号1.2秒独立RAW前缀。最终是否完成只看`output/stage62_independent_axis_20261021/raw/raw_prefix_checks.json`。缓存评分只有在实际完整15/生命周期精确回归后才可解释为该RAW入口同范围的预测结果，不额外当成第二批独立精度样本。

冻结源保持。`study62.py score`原入口存在协议cases列表与原22所需映射不兼容的错误，执行在写analysis之前失败；使用单独的`score62.py`只读映射原61冻结时间区间，评分公式不变，`score_adapter.json`保留说明。不要重跑`tools/run_stage62.py`覆盖既有目录。需要全新复核时应另建独立协议目录；已有结果核查用verify，不重新冻结。

下一步：纯线几何上限0.4低于学习0.45，现阶段纯线只能引导预测，不能独自启动学习。先独立验证线方向的多时刻一致性/失效行为，再考虑如何用角度不确定性构造准入，不提高虚假质量。若原转子轴覆盖仍不足，复用6的`body_measurement`（整机PCA/LSD）借整机检测约束身份、只输出方向，不能重复原四点初始化要求，也不能伪造桨心。缺中心的局部周期与多桨重叠解混另做实验，不能用一条最强频谱峰替代四桨推力和。

输出目录：`D:/drone forecast/output/stage62_independent_axis_20261021`。RAW/dataset只读；不使用G，不读取/执行/修改yolov10-main。

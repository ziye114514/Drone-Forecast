# 137：几何执行加速，保留135预测逻辑

2026-10-24。当前入口仍为 `D:/drone forecast/event_drone_forecast/run_optimized_forecast.py`，默认几何后端为 `fast`；`--geometry-backend legacy` 在同一CLI中选择原136后端。每次CLI独立进程，不用导入旧研究入口作为同进程API。

[结果、耗时图和视频](<D:/drone forecast/output/stage137_fast_geometry_20261024/几何计算加速与预测结果.html>)。

## 改动

1. `fast137.fast_fixed_jac` 批量计算四桨/八参数的原Sampson目标解析导数。固定结构只需八列，原117先算十列再丢弃两列；相同点、初始化、优化器与资格不变。
2. 计数前端直接复用124 `SupportFrontend(...,'count_global')` 的当前原始阈值掩膜，防合并阶段免重复Gaussian/分位数。
3. 占据分支的阈值始终为0.5，去掉原通用函数中未用的全图正值提取/分位数。ROI仍只做原中心资格，不裁断轮廓，不改全图计数阈值。
4. 入口增加前端/模型耗时统计，修复cProfile相对`__file__`与绝对根路径的指纹兼容问题。136完整入口源码保存在`research/stage136/run_optimized_forecast_v3.py`，历史快照不要在research目录直接运行。

## 已完成结果

- 同入口22—24秒共同100步：处理233.279→189.984ms（18.56%）；几何171.640→117.788ms（31.37%）；结构拟合111.413→58.098ms（47.85%）；前端72.399→63.758ms（11.94%）。顺序运行、系统负载非受控，不是跨硬件基准。
- 新清晰完整8秒400步：处理180.659ms/步，另有序列化/视频12.983ms；包含冷启动/读取/视频总97.906秒。50fps是离线回放，不是实时速度。
- 清晰400步前端输出、机体检测、当前建模资格完全相同；未来坐标最大差1.751e-11px。原共同发布未来框误差0.1/0.2/0.4秒仍8.578/11.312/30.834px。原未发布/未配对分母保持，不筛误差。
- 26组实际拟合输入，旧新导数最大差1.706e-12，并用独立中心数值差分方向导数检查；交替同输入单次1.814→0.629ms。
- 实际RAW成功：旧后端100、新清晰400、侧视50、空场30，共580；另有cProfile100（不用于耗时对照），总680。侧视只核对同一机体/预测，不称完整几何逐项核验或侧视收益。空场无有效预测；事件cutoff/无缓存标签推理/源码指纹/视频解码通过。
- 初次cProfile相对路径在启动NN前指纹失败，`profile_raw`空目录与`profile.log`保留；绝对路径剖析完成，旧入口相对文件名兼容问题已修复。没有调检测权重、OpenCV/BLAS线程、OS、依赖或保护目录。

## 运行

```powershell
& 'D:\drone forecast\.venv-event311\Scripts\python.exe' `
  'D:\drone forecast\event_drone_forecast\run_optimized_forecast.py' `
  --raw 'E:/dataset/train/6/media/event.raw' `
  --output 'D:/drone forecast/output/my_fast_run' `
  --start 22 --duration 8 --preview
```

输出目录须不存在；固定1280×720/20ms，现有SDK/两套Python环境/固定ONNX权重。默认54/API93未变。

## 下一步

当前全图前端与几何拟合仍慢，20ms实时尚未达到；GPU检测本身约34ms。下一项实现保留全图阈值统计的ROI共享支持计算，并对跨ROI轮廓回退全图，不能直接裁图后声称完整桨盘。侧视法线与真实时间周期进入预测的链路还未完成；此轮未改善侧视，也未新增转速/推力标定或论文创新。继续用小固定段决定是否保留代码变更，不扩张历史控制树。

# 第九阶段：整机状态保护与重新确认

输出：`D:\drone forecast\output\stage9_state_recovery_20261017`，先读`阶段结论.md`或`第九阶段整机状态保护报告.html`。

保持旧src、旧入口和阶段8结果不变；不使用G，不访问或修改`D:\yolov10-main`。仅原生RAW作为传感输入，不需要质量、尺度、推力系数或同步飞控真值。

`recovery.py`继承已冻结的阶段8前端，复用原椭圆拟合、分组及局部布局传播。新增实际四候选的连续尺度参考、异常状态回滚，以及丢失后的三帧新组合确认。部分/推断布局不刷新可靠四点年龄。所有输出仍是图像像素观测，不代表真实姿态、机械转速或身份保证。

四主方法：`strict10`、`neighbor10`旧对照，`guard_strict10`、`guard_neighbor10`新保护分支。侧视额外运行`protect_neighbor10`，关闭新假设重新确认以分离贡献。参数在RAW试验前冻结，无结果后参数搜索。

已完成：19.5秒六段开发RAW、975发布时刻；本轮重新计算的neighbor10与旧缓存975时刻逐项相同；侧视53处理帧前缀检查通过。6个模块检查通过。`audit_side.py`额外记录10ms状态事件，确认109.95s一次转移；每20ms结果会漏掉该事件，因此不能从主表“转移0”推断没有重新捕获。

复查结果（不会重算RAW）：

```powershell
& '.\.venv-event311\Scripts\python.exe' 'event_drone_forecast\research\stage9\study.py' analyze
& '.\.venv-event311\Scripts\python.exe' 'event_drone_forecast\research\stage9\report.py'
```

不要对现有目录再次执行`freeze`/`all`：完成标记禁止覆盖。新算法应新建版本/协议。

复用接口：将`src`与`research/stage9`加入Python模块路径后，`RecoveryFrontend(width,height,'neighbor10').detect(events,t)`；事件须为连续10ms批次，t为截止时刻。不要以其他批次宽度直接运行。独立RAW观测命令入口为项目根目录`run_state_recovery.py`，帮助及限制由该入口说明；它不是未来轨迹预测入口。

例如在项目工作目录观察同一侧视段，选择未使用的新输出目录：

```powershell
& '.\.venv-event311\Scripts\python.exe' 'event_drone_forecast\run_state_recovery.py' --raw 'E:\dataset\train\6\media\event.raw' --start 108.97 --duration 4.03 --method neighbor10 --output 'output\my_state_observation' --preview
```

CLI从指定start冷启动，逐10ms写出观测与全部audit。上述起点包含与固定研究相同的30ms前置片段；没有自动隐藏预热。CLI已实际烟测108.97–109.50秒、53处理时刻，输出视频为事件计数叠加，不是G或未来预测。

外部查阅：DeepSORT的确认/失配生命周期及ByteTrack的lost/reactivate；没有引入其网络或运行其整套作者程序。DSH仅做过一次受限文本审查，证据和监督驳回在输出`dsh_review`，不是实验真值。

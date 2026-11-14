# 118：已学固定结构的完整 RAW 预测实验

入口：`event_drone_forecast/run_learned_structure_forecast.py`。仅输入事件RAW，复用现有SDK、Python环境及固定NN检测权重。原107的64路保持，追加3路匹配实验；默认54、API93不替换。

```powershell
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast/run_learned_structure_forecast.py --raw 'E:/dataset/train/6/media/event.raw' --output 'D:/drone forecast/output/my_learned_structure_trial' --start 22 --duration 5 --preview --verify-frames
```

输出目录必须不存在。运行时不接收既有框、轮廓、标注或录像缓存。新增几何worker独立读取同一个RAW、只推进到请求时刻，并禁止读取output中的JSON/轮廓/视频缓存；NN机体框和当前轴由同一完整RAW流程提供。进程隔离避免新增椭圆运算改变旧OpenCV随机数状态。此版是同步研究原型，新增读取和求解开销，尚未实时。

结构假设仍是四个等半径共面桨盘、近似矩形机架。原五帧合格初始化后，规范90度模板重排，学习无量纲半径和机架长宽比，随后固定结构、只求当前投影H8。它没有恢复真实米制尺寸或物理重心。当前姿态过期、五帧重新确认和机体身份重置规则保持；结构记忆不产生缺失桨盘观测。每段独立启动，不将22秒模型注入110秒片段。

追加系数：

- `model_free20_kf118`：相同当前20ms几何、每帧自由结构的匹配对照。
- `model_fixed_kf118`：首次建模后固定已学结构。
- `model_fixed_guarded_kf118`：同上，增加以固定结构为条件的位姿中心不确定性门槛。该局部协方差尚未包含结构学习误差。

清晰同7/8个既有稀疏0.4s未来框端点，16.8426→16.7800px，只是很小的开发集差异；原完整融合13.7081px仍更好。侧视未首次建模，不能声称侧视恢复收益。没有新机械RPM或真实COM/姿态真值。

112/113保存原生阈值轮廓、对比完整椭圆周长/实际轮廓/弧段/稳健损失，弧段分支清晰退至21.64px；114/115弱轮廓分水岭单C帧合格但整段退至22.86px，均保留为负结果、未接入本入口。116同目标解析导数独立交错基准约1.094倍配对比，不等于全链路提速。117已学固定结构还测试静止姿态摆动、真实平移、变向及缺桨恢复；为轮廓层合成，缺桨期仍有大误差。

完整本轮结果、协议、报告与视频位于 `output/stage118_RAW_learned_structure_20261022`。`study118.py`执行真实RAW后，在独立父进程核查原64精确107、新3及内部状态精确117、两实际RAW截断前缀与负例。`delivery118.py`生成额外绘图和叠加视频，`finalize118.py`更新工作日志和可恢复交接。

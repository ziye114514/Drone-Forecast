# 第七阶段：FRED定位与预测误差对照

输出目录：`D:\drone forecast\output\stage7_localization_budget_v3_20261017`。

实际复用 OpenCV 5.0.0 `KalmanFilter.predict/correct`、原项目RAW解码/桨叶前端，以及第二阶段已修复的 `activity_fallback_fixed.py`。没有改写原 `src` 或前六阶段。该项目是误差诊断工具，不是完整RPMKF/RKD复现，也未更换部署默认入口。

FRED本地0–10号各连续前20秒。官方canonical划分固定在FRED提交 `2bf89c5376eda528431b62d6c60f2c13d8f95ab4`；8、9号属于官方测试，但本轮只是这两条的短前缀，不能称完整官方成绩。1–3号此前看过；0、4、5、6、7、10号是新增训练序列迁移检查。预测器参数在全部评分前冻结，无测试集参数搜索。

四种输入：100Hz自动流；采样到标注时刻的自动位置；相同测量时刻/相同缺失模式下替换为过去标注位置；全部过去标注位置。后二者是诊断对照，不能称RAW-only。三个预测器：CV卡尔曼、CA卡尔曼、CV状态加最近0.35秒历史加速度。标注不插值，实际预测时长/缺失输出逐项保留。

运行环境：项目根目录 `D:\drone forecast`，Python `.venv-event311\Scripts\python.exe`。源码位于 `event_drone_forecast\research\stage7`。

```powershell
# 本轮固定结果已经存在；run/all复用已完成RAW，不覆盖原生数据。
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage7\study.py tests
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage7\study.py all
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage7\diagnose.py
& '.\.venv-event311\Scripts\python.exe' event_drone_forecast\research\stage7\report.py
```

新实验应另设OUT并重新freeze，保留既有协议。`prepare.py`为初始上游审计/本地数据准备脚本，其原始OUT保留在v1；不必重复下载或解压。v3通过inventory引用这些已准备数据。

失败记录保留：v1在检查测试区间外的零面积标注时中断；v2误引用第二阶段原失败活动器，造成无法重新捕获，已在测试8/9前停止。v3改为既有修复适配器，并实际测试目标消失后重新捕获。v3所有活动观测从RAW重算；只有源码/RAW指纹相同的已完成桨叶观测缓存可复用。计时明确标记缓存行。

无无人机语义分类、独立实飞姿态/RPM真值、真实质量/推力系数/风速恢复或实时性保证。活动前端的目标关联错误也属于本次位置替换干预的组成部分。公开仓库DroneStalker权重仅核查存在和适用条件，未执行或将其自报指标当作公平基线。

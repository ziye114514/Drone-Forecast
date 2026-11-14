# 60复用核查与后续方向，2026-10-21

本轮实际复用OpenCV LSD、SciPy指派、旧57/9/8几何与54/53/52完整预测。双边条带配对是标准几何的小实现，不声称复现下列完整作者系统或算法创新。

- OpenCV LSD：https://docs.opencv.org/4.x/db/d73/classcv_1_1LineSegmentDetector.html 。现有依赖可直接运行，输出线段/宽度等；本轮将相反梯度方向的近似平行边配对，之后还需端点和身份证据。
- RT-EvLDT（ICCVW NeVi2025）：https://github.com/event-driven-robotics/RT-EvLDT ，https://arxiv.org/html/2510.06829v1 。事件线段跟踪已有完整作者实现；本轮没有编译执行，不能将本地LSD当作该论文复现。
- PowerLine Tracking with Event Cameras（IROS2021）：https://github.com/uzh-rpg/line_tracking_with_event_cameras 。作者ROS/C++事件线跟踪开源，Ubuntu18.04/ROS Melodic环境且有标定要求，GPL3；并非现成无人机桨心或完整直径身份识别。本轮只查阅。
- ELSED：https://github.com/iago-suarez/ELSED 。已有条线延伸/断裂处理检测器，Apache2，需要C++构建/Python绑定；当前已有OpenCV，未增加安装或冒称实际运行。

## 避免重复旧运动补偿实验

项目21 windows21.warp已经按过去速度把事件平移到当前时刻，比较过20/10/5ms和warp20，整体没有统一赢家，不能再把平移运动补偿本身称新点。原6也试过平移旧密度图。两者应复用/引用，新的对照须明确改进的是速度来源、局部边缘记忆或双边/身份验证。

- 运动补偿经典主来源：https://rpg.ifi.uzh.ch/docs/CVPR18_Gallego.pdf 。固定时间事件累积会把运动边缘叠宽；按事件时间对齐是已有方法。其是否是本次侧视失败的主因仍须对照，不能由截图宽就断定。
- SCARF新论文（2026-09预印本）：https://arxiv.org/html/2609.19973 ，作者代码总库https://github.com/robotology/event-driven 。局部重叠patch保存有界原始事件，保留时戳，对多速度和线特征有对照。本轮未核定/下载其具体源码提交或编译执行。它可能比继续调高计数阈值更值得下轮核查，但保留过时形状、桨叶转动和机体平移不同速度等问题还需测试。
- Motion-aware Event Suppression（RSS2026）：https://github.com/uzh-rpg/event_suppression ，作者项目https://rpg.ifi.uzh.ch/event_suppression/ 。公开代码面向当前/未来动态掩码，在DSEC/EVIMO训练验证，含预训练权重链接；EED loader当前缺失。未来掩码预测不等于本项目桨叶姿态/推力/中心FDE。抑制动态事件的原任务可能会删掉目标无人机，若复用需明确保留目标侧掩码并测跨域；当前未下载安装或运行。

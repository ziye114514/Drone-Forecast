"""Save Unicode-path PNG posters via Python file I/O from verified videos."""
from pathlib import Path
import cv2
from study import OUT
from build_report import write_image

for sequence in ('1','2','3'):
    folder=OUT/f'activity_fallback_fixed/sequence{sequence}'
    video=cv2.VideoCapture(str(folder/'真实未来对照.mp4'));video.set(cv2.CAP_PROP_POS_FRAMES,100)
    ok,frame=video.read();video.release();assert ok
    write_image(folder/'未来对照_15秒.png',frame)
for experiment in ('integrated','guarded'):
    folder=OUT/experiment/'own_train6';video=cv2.VideoCapture(str(folder/'自有录像_真实未来对照.mp4'))
    video.set(cv2.CAP_PROP_POS_FRAMES,39);ok,frame=video.read();video.release();assert ok
    write_image(folder/'自有录像_24秒.png',frame)
path=OUT/'第二阶段实验报告.html';document=path.read_text(encoding='utf-8')
marker='这是当前推荐用于继续四桨论文复现的实验入口；'
replacement='本片段中受约束的活动接续触发0次，15个端点的预测与原v0.3完全一致。这轮没有证明该接续能提高真实预测精度，已确认的是无约束回退不可直接用于自有录像，以及限制身份后不再增加本片段的背景误报。这是用于后续实验的入口；'
assert marker in document;document=document.replace(marker,replacement)
path.write_text(document,encoding='utf-8')
print('Five posters saved; no-real-fallback-activation result made explicit',flush=True)

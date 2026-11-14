"""Domain-adaptation result and visible coarse-label error examples."""
import html,json
import cv2,numpy as np
from common40 import *
def table(rows,cols):return '<table><tr>'+''.join('<th>'+c+'</th>' for c in cols)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(c)))+'</td>' for c in cols)+'</tr>' for r in rows)+'</table>'
def main():
    verify();a=load(OUT/'analysis.json');truth={r['id']:r for r in load(OUT/'targets.json')['targets']};inputs={r['id']:r for r in load(OUT/'inputs.json')['images']};rr=[r for r in a['details'] if r['method']=='finetuned'];pos=[r for r in rr if r['recording']==3 and r['state']=='present'];chosen=[pos[0],pos[35],pos[70],pos[-1]]+[r for r in rr if r['recording']==3 and r['state']=='absent' and r['detected']]+[r for r in rr if r['recording']==6 and r['state']=='present' and not r['detected']][:2]
    canvas=np.full((2*258,4*448,3),245,np.uint8)
    for k,r in enumerate(chosen[:8]):
        im=cv2.imdecode(np.fromfile(OUT/inputs[r['id']]['image'],np.uint8),1)
        for b,color in [(truth[r['id']].get('bbox'),(220,55,210)),(r.get('bbox'),(0,190,255))]:
            if b is not None:b=np.rint(b).astype(int);cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),color,3)
        im=cv2.resize(im,(448,252));cv2.putText(im,f'{r["id"]} {r["state"]}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);canvas[k//4*258:k//4*258+252,k%4*448:(k%4+1)*448]=im
    cv2.imencode('.png',canvas)[1].tofile(str(OUT/'heldout_examples.png'))
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>第四十阶段整机检测微调</title><style>body{font:16px/1.8 Arial;margin:30px auto;max-width:1250px;color:#233d4b}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border:1px solid #ccd9e1}th{background:#edf3f7}</style><h1>第四十阶段：用自采粗框适配公开整机检测器</h1><p><b>权重留出的3号录像107个有机标注时刻，检出106、105个最高置信框IoU≥0.5，中心误差7.14px。</b>同输入原公开权重零检出。11个无机时刻有2次输出，误报仍存在；另7张已知空场5号零输出。3号此前已开发查看，不是全球盲测，也不能代表不同机型/环境普遍泛化。</p><h2>固定划分与训练</h2><p>1/2/4号355张训练，6号180张验证、只用于选择best，3号118张测试及5号7张负例。全部由RAW生成原生过去20ms ON/OFF，SDK10ms降噪，无空间缩小/G/GT裁剪。原标注是可见框，不是质心/姿态/RPM。固定一次20轮/640/batch8/AdamW/seed40/FP32，约412秒；实际启动352组全部预训练参数逐项精确。训练不能访问测试图/真值，推理不能访问所有真值。</p><p>复用<a href="https://github.com/takara-ai/DroneStalker">DroneStalker</a>固定公开YOLO结构/权重与<a href="https://docs.ultralytics.com/modes/train/">Ultralytics官方训练接口</a>，属于标准域适配，不称新网络或算法创新。训练数据与推理表示在同一自采相机域，仍需要独立新场景检验。</p><h2>固定conf0.25最高置信框评测</h2>'''+table(a['table'],['recording','method','present','detected_present','center_success','iou_success','mean_error_px','absent','false_positive_frames'])
    text+='''<p>验证集6参与best选择，其成绩不能当独立测试。训练器的验证mAP使用裁到FOV的YOLO验证框；此表用存档原始框，阈值固定，二者定义不同。测试全保留，没有剔除小目标/无事件帧。</p><h2>画面与失败</h2><img src="heldout_examples.png"><p>紫框=已有可见框；橙框=最高置信网络输出。前四图固定取3号有机序列位置0/35/70/末，后两图为3号无机输出，再两图为6号漏检；没有只展示最佳例。两张无机标注时刻在画幅左缘可见局部目标，存在标注规则或时间对齐冲突；仍按存档状态保留2次输出计分，不自行改标签。近距离四盘清晰的大目标也有漏检，说明尺度/外观泛化仍有限。</p><p>准备时6个验证框超720px画幅2.62–15.21px，保留失败目录，仅YOLO标签裁剪，原targets与评分不变。评分首次JSON拒绝np.int64，以仅类型序列化桥修复，未改预测/阈值/算术。所有冻结协议/失败/补充文件保留。</p><p>本阶段只证明当前整机定位改善，未证明预测改善。连续真实未来评测见<a href="../stage41_continuous_body_forecast_20261018/第四十一阶段连续整机预测报告.html">阶段41</a>。<a href="analysis.json">逐目标完整结果</a> · <a href="protocol.json">冻结协议</a> · <a href="initialization_check.json">预训练参数核验</a></p></html>'''
    (OUT/'第四十阶段整机检测微调报告.html').write_text(text,encoding='utf-8');(OUT/'阶段结论.md').write_text('3号权重留出测试106/107检出、105/107 IoU>=.5，11无机2输出。当前检测明显改善，非全球盲测/异机型泛化或姿态RPM证明。固定标准域适配，详见HTML。',encoding='utf-8');print('Stage40 report complete',flush=True)
if __name__=='__main__':main()

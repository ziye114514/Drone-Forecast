"""Join saved forecast results and the separately verified threshold probe."""
import argparse,html
import study57 as current
import delivery57 as delivery
import density_probe57 as density

def render():
    delivery.verify();density.verify();folder=current.OUT
    data=current.load(folder/'three_way_results.json')['table'];probe=current.load(folder/'density_probe_analysis.json')['summary'];gates=current.load(folder/'initialization_diagnostic.json')['table']
    assert probe['changed_threshold_ticks']==probe['active_roi_ticks']==192
    tab=delivery.old.table
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>57：修复桨盘退化，继续定位侧视瓶颈</title><style>body{max-width:1450px;margin:28px auto;padding:0 16px;font:17px/1.8 "Microsoft YaHei",Arial;color:#203747}img{width:100%}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ccd8df;padding:6px}</style><h1>修复硬ROI引起的清晰几何退化，侧视预测仍未突破</h1><p>改为完整轮廓之后按过去整机框筛选候选，保留全部当前事件与原全图密度阈值。清晰桨盘几何/频率恢复原54，旧机架合格时刻从硬裁的3恢复60、最长连续从1恢复13；它是回归修复，不能写成超过原方案。</p><p>同目标0.4秒物理误差：清晰原14.22/硬裁17.20/候选14.22px；侧视122.46/120.94/122.45px；3号136.69px不变。各段联合成功数不变，侧视纯历史81.91px仍更好。所有当前位置和六历史固定，未升级默认。</p><img src="three_way_comparison.png">'''+tab(data,['case','model','targets','valid','global_fde_px','hard55_fde_px','candidate57_fde_px','global_success','hard55_success','candidate57_success'])+'''<h2>实际阈值记录确认硬裁剪改变了统计量</h2><p>对原8检测函数AST仅增加一条记录赋值，两次独立实际RAW读取分别逐时刻精确复现55全图/硬裁的200个完整几何观测。192个ROI启用时刻的阈值全部改变，每帧阈值比值中位约1.315、最大约1.503；全段阈值中位约3.025→3.977，非零密度格中位18412→11631。被删事件虽少，空间分布足以改变分位数。</p><img src="density_threshold_probe.png"><p>这确认了阈值变化，但尚未将轮廓切边与阈值变化对预测的影响单独消融；不宣称阈值是唯一原因。新方案同时保留两者，恢复了清晰几何。探针不用于在线推理，不改变原检测数值。</p><h2>侧视仍卡可靠共同机架</h2>'''+tab(gates,['case','native10','four_current_candidates','eligible_raw_current_ticks','maximum_raw_current_streak','required_streak','reasons'])+'''<p>侧视71个四候选中只有2个时刻满足旧机架条件，最长连续2不足原5；3号仍无合格机架。旧平行四边形假设受透视与椭圆中心偏差影响，不能直接降低门槛、凑四点或以任意单应性零残差证明身份。</p><h2>证据与使用</h2><p>主体1715父/3430子时刻、1480视频帧、独立225/450真实RAW前缀；关闭完整15模型精确54，两路六历史/覆盖固定，全部360行评分中252基线/历史行精确54，空场零输出。本轮无通信解析失败。额外阈值探针400原生10ms观测单独计数。</p><p><a href="三路预测对照与固定时刻画面.html">完整视频与固定时刻画面</a> · <a href="第五十七阶段完整轮廓候选ROI报告.html">全部统计</a> · <a href="density_probe_analysis.json">完整阈值记录</a> · <a href="density_probe_supplement.json">探针协议</a> · <a href="raw_prefix_checks.json">真实RAW前缀</a></p><h2>下一步已有研究基础</h2><p>下一阶段优先核查共同桨盘圆锥约束和真实轴心投影，而不继续重复框余量试验。<a href="https://link.springer.com/article/10.1186/s41074-018-0050-y">共面圆锥的单应性与歧义分析</a>已有理论；<a href="https://github.com/xiahaa/circular-center-calibration">TIM2026圆心校正作者代码</a>已有2D求解与多圆消歧，但其已发布合成试验给定相机内参，不能直接视为本机无标定问题已解决。当前仅完成资料核查，尚未执行这些外部代码。</p></html>'''
    (folder/'实验结论与密度阈值诊断.html').write_text(page,encoding='utf-8')
    print('Stage57 conclusion and threshold diagnosis published',flush=True)

if __name__=='__main__':render()

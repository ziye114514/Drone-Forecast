"""State what the completed controls prove and which bottleneck they expose."""
from pathlib import Path
import datetime
import forecast61 as current
OUT=current.current.OUT

def run():
    current.verify();delivery=current.load(OUT/'full_future_delivery61.json');usage=delivery['physical_feature_usage_at_scored_anchors'];gate=delivery['unchanged_old_layout_initialization_gate']
    train=next(r for r in usage if r['case']=='off_train3' and r['horizon']==.4)
    side=next(r for r in usage if r['case']=='off_side6' and r['horizon']==.4)
    assert train['valid_sparse_targets']==11 and train['axis_freq_prediction_equal_CV']==10 and train['anchors_with_prior_RLS_updates']==0
    assert next(r for r in gate if r['case']=='off_side6')['longest_streak']==2
    assert next(r for r in gate if r['case']=='off_train3')['eligible_current_ticks']==0
    success_changes=[r for r in delivery['all_changed_score_rows'] if len({r['legacy_success'],r['stable_success'],r['paired_success']})>1]
    gate_audit=current.load(OUT/'feature_gate_audit61.json')
    pagepath=OUT/'稳定线段与配对完整预测报告.html';page=pagepath.read_text(encoding='utf-8')
    block='''<h2>这轮结论与实际卡点</h2><p><b>稳定性修复已完成完整链验证，未来位置没有实质新改善，配对融合暂不作为默认。</b>清晰0.4秒轴向频率仍14.224px；侧视122.448→稳定122.404px，加配对仍122.404px；3号136.693px基本相同。配对额外变化远小于像素级，不能宣传超过论文或新的稳定精度收益。配对在这批数据增加处理开销，当前不值得默认开启。</p><p>3号0.4秒11个有效目标中，物理分支只有1个锚点active，10个目标的预测精确等于CV，所有锚点之前的实际RLS更新均0；侧视5个有效目标中只有3个物理active，2个等于CV。这解释了为什么当前轮廓变化未必进入最终未来预测，不能只继续数四桨候选。</p><p>旧机架门槛侧视只有2个合格时刻、最长连续2（要求5），3号0；尚不能靠配对提供稳定四桨布局。仿射门槛可能受透视/遮挡影响，不能通过放宽门槛凑四点证明身份。</p><p>后续实验优先用run_stable_paired_forecast.py、line_mode=off作为稳定测量基线，旧54/57保留历史对照；这不是新的精度升级。下一轮先复用原6整机PCA/LSD无向轴观测，利用当前可靠的NN整机身份/位置作约束，把“完整桨心”和“轴方向”分开准入。再复用原28轴向模型验证独立于四个转子ID的轴状态与在线有效参数，不伪造缺失桨心/转速。必须同时保留纯历史和相同回退策略的对照，避免把CV改成历史回退带来的收益误称桨叶物理增益。</p>'''
    old='下一轮先复用原6整机PCA/LSD无向轴观测，利用当前可靠的NN整机身份/位置作约束，把“完整桨心”和“轴方向”分开准入。再复用原28轴向模型验证独立于四个转子ID的轴状态与在线有效参数，不伪造缺失桨心/转速。'
    new='源码复查发现，原27为公平消融故意共用频率门槛，现集成的axis_only虽设ratio=1，仍要求新鲜四频率、frequency_quality×geometry_quality，学习启动还要求four_visible。原51已保留轴状态/轴模型跨转子epoch，不能再重复该修复。下一轮优先复用原28/52轴模型，单独对照“仅轴分支独立几何准入/独立学习”，其余转速分支保持逐项相同，不伪造缺失桨心或RPM。既有轴观测不足时，再评估原6整机PCA/LSD观测器，避免先重复实现。'
    assert old in block;block=block.replace(old,new)
    block+='<p>新鲜轴状态不等于已验证姿态质量。侧视175个新鲜轴状态时刻中91个同时没有新鲜频率、157个没有四桨同时可见；3号84个中分别47和81个。独立准入仍需几何质量、当前支持、时间期限和原延迟反馈/预热，不是删除所有质量约束。另估算过去CV速度对应10ms平移中位clear1.50/side3.07/train35.70px，3号沿轴平移约当前major的17.7%；只是补偿可行性线索，未测真实模糊或收益，原21已实现warp且旧warp20无统一收益。</p><p><a href="feature_gate_audit61.json">已有准入代码与当前状态诊断</a> · <a href="motion_feasibility61.json">仅过去速度的平移尺度诊断</a></p>'
    marker='<h2>0.4秒同目标未来误差</h2>';assert page.count(marker)==1;page=page.replace(marker,block+marker,1);pagepath.write_text(page,encoding='utf-8')
    current.save(OUT/'conclusion_full61.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),helper_sha256=current.sha(Path(__file__)),input_delivery_sha256=current.sha(OUT/'full_future_delivery61.json'),
        measured_forecast_gain_substantial=False,pairing_promoted=False,preferred_research_baseline_entry=str(current.ROOT/'core/run_stable_paired_forecast.py'),preferred_line_mode='off',
        legacy_default_entry_retained=str(current.ROOT/'core/run_deploy_forecast.py'),production_accuracy_upgrade_claim=False,
        physical_usage04=dict(train3=train,side6=side),joint_success_changed_rows=success_changes,
        feature_gate_audit_sha256=current.sha(OUT/'feature_gate_audit61.json'),motion_feasibility_sha256=current.sha(OUT/'motion_feasibility61.json'),
        next_step='First isolate original27 shared frequency/axis gate on adaptive_axis_only with original28/52 predictor, geometry-only axis quality/current support, unchanged quality values/warmup/delayed feedback. Keep RPM branches and six histories exact; include pure-history/equal-fallback controls. Existing51 already preserves axis/model across rotor epochs; do not repeat it. Only if pose support remains insufficient reuse6 whole-body PCA/LSD with NN identity. No fabricated centers/RPM or lower layout gates.'))
    print(dict(conclusion='No substantial future gain; pairing not promoted',train3_CV_fallback_targets=train['axis_freq_prediction_equal_CV'],joint_success_changed_rows=len(success_changes)),flush=True)

if __name__=='__main__':run()

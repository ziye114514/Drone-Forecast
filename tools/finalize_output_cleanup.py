"""Write current output entrypoints and handoff after native cleanup finishes."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,html,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output';RECORD=OUT/'cleanup_20261028'
def size(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
def main():
    plan=json.loads((RECORD/'plan.json').read_text('utf-8'))
    deletion=json.loads((RECORD/'deletion.json').read_text('utf-8'))
    assert deletion['completed'] and len(deletion['deleted'])==len(plan['delete_after_verified'])
    old=OUT/'github_release_final_20261027/event-drone-forecast-v0.4.0-alpha.1.zip'
    new=OUT/'github_release_update_20261028/event-drone-forecast-v0.4.0-alpha.2.zip'
    assert hashlib.sha256(old.read_bytes()).hexdigest()=='ff1172b3f2d2e1752d8ebc6d299c6d972adfaaa1776f8773240e44aa3eb58abb'
    assert hashlib.sha256(new.read_bytes()).hexdigest()=='6260e0919a8dea574e8c502a0f982b3ed5791d5fe8d159c6b2e740f889b97e45'
    assert (ROOT/'archive/output_history_20261028.zip').is_file()
    before=json.loads((RECORD/'archive_verified.json').read_text('utf-8'))
    result=dict(original_output_GiB=15.989912928082049,current_output_GiB=size(OUT)/1024**3,
        history_archive=str(ROOT/'archive/output_history_20261028.zip'),archive_GiB=before['archive_bytes']/1024**3,
        disk_space_reclaimed_GiB=(shutil.disk_usage(ROOT).free-plan['disk_free_before'])/1024**3,
        historical_files_verified=before['verified_file_SHA256_count'],removed_scattered_entries=len(deletion['deleted']),
        remaining_top_entries=len(list(OUT.iterdir())),
        kept_current_results_labels_checkpoints_releases=True,published_ZIPs_unchanged=True,
        original_dataset_and_working_source_untouched=True,history_preserved=True,remote_action='none')
    (RECORD/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf-8')
    guide='''# 当前结果入口

- [GitHub新版交付](github_release_update_20261028/拖拽上传说明.txt)
- [当前完整流程](github_release_update_20261028/event-drone-forecast/docs/CURRENT_PIPELINE.md)
- [最新预测与性能报告](shared_template_budget_20261027/index.html)
- [最新交接](checkpoints/latest_handoff.json)
- [清理记录](cleanup_20261028/summary.json)
- [历史完整压缩归档](../archive/output_history_20261028.zip)

历史散落目录已归档，不属于运行当前项目的必需内容。
人工标注、checkpoint、当前实验及发布所依赖的演示素材保留。
历史脚本中的旧output路径需要先恢复对应归档条目；不用为运行当前版本解压全部历史。
'''
    (OUT/'结果入口.md').write_text(guide,'utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>当前无人机预测项目</title>
<style>body{font:17px/1.8 system-ui;max-width:900px;margin:50px auto;padding:0 24px;background:#f4f6fb;color:#18304d}a{color:#1260a1}li{margin:12px 0}</style>
<h1>当前无人机预测项目</h1><p>五圆可见性模型、条件周期输入与二力反馈；v0.4.0-alpha.2源码已准备，尚未远程发布。</p>
<ul><li><a href="github_release_update_20261028/拖拽上传说明.txt">GitHub：两批拖拽上传路径</a></li>
<li><a href="github_release_update_20261028/ROOT_README_FOR_NESTED.md">外部README源代码</a></li>
<li><a href="shared_template_budget_20261027/index.html">最新实际预测与性能报告</a></li>
<li><a href="cleanup_20261028/summary.json">清理结果与空间统计</a></li>
<li><a href="../archive/output_history_20261028.zip">历史完整归档（按需恢复）</a></li></ul>
<p>原始数据、人工标注、交接和两版正式发布包保留。历史已压缩归档，旧路径脚本需先恢复对应目录。</p></html>'''
    (OUT/'查看实验结果.html').write_text(page,'utf-8')
    note='# 历史output归档\n\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n\nZIP内部路径相对于原output根目录；可仅恢复所需子目录。历史源码/统计/失败结果/数据副本均保留，逐文件SHA256已核验。文件清单见ZIP内__ARCHIVE_MANIFEST__.json。\n'
    (ROOT/'archive/README.md').write_text(note,'utf-8')
    source_readme=ROOT/'event_drone_forecast/README.md';text=source_readme.read_text('utf-8')
    marker='> 2026-10-28 输出整理：最新结果入口见 [output索引](../output/结果入口.md)。历史散落结果已完整压缩到 [历史归档](../archive/output_history_20261028.zip)；下文旧output路径仅在按需恢复后可用，当前运行与发布依赖已保留。\n\n'
    if marker not in text:source_readme.write_text(marker+text,'utf-8')
    checkpoint=OUT/'checkpoints/latest_handoff.json';handoff=json.loads(checkpoint.read_text('utf-8'))
    stamp=datetime.now(timezone.utc).isoformat()
    handoff.update(updated_utc=stamp,output_cleanup=result,current_running_jobs=[])
    handoff.pop('output_cleanup_in_progress',None)
    handoff['github_update']['upload_batches']=json.loads((OUT/'github_release_update_20261028/upload_batches.json').read_text('utf-8'))
    handoff['github_update']['outer_readme']='output/github_release_update_20261028/ROOT_README_FOR_NESTED.md'
    handoff['output_cleanup']['restore_note']='Historical output paths remain recorded in old scripts; restore selected ZIP entries into output before those historical reruns. Current runtime and release generator dependencies retained.'
    checkpoint.write_text(json.dumps(handoff,ensure_ascii=False,indent=2),'utf-8')
    with (ROOT/'WORKLOG.md').open('a',encoding='utf-8') as f:
        f.write('\n\n## 2026-10-28 用户要求精确上传路径、外部README及output清理\n'
            '- 已准备两批上传第1批/第2批下同名event-drone-forecast，80/39文件；第三步根README/许可文件，不拖外层批次目录。外部README改为可复制简版；正式ZIP未变。\n'
            '- output原约15.99GiB、321顶层条目；历史'+str(result['historical_files_verified'])+'文件逐一SHA256验证后压缩归档archive/output_history_20261028.zip，再由原生PowerShell安全删除322个散落/临时条目。\n'
            '- 保留当前shared_template实验、人工标注、checkpoints、两版正式包、发布依赖three_continuation/shared_angular_phase。代码/模型/E盘数据/受保护项目未动。\n'
            '- 实际当前output约'+str(round(result['current_output_GiB'],2))+'GiB；历史压缩约'+str(round(result['archive_GiB'],2))+'GiB；净释放'+str(round(result['disk_space_reclaimed_GiB'],2))+'GiB。\n'
            '- 历史代码内旧路径需要按需恢复归档，不能把这些路径当仍存在；最新入口output/查看实验结果.html已更新。原始独有结果没有永久丢弃，旧/新发布ZIP哈希核对保持。\n')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()

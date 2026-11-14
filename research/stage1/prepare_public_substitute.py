"""Inventory/extract only existing public RAW and annotations; no inference yet."""
from pathlib import Path
import csv,hashlib,json,os,shutil,zipfile
import numpy as np
from run_study import OUT,save
dest=OUT/'public_truth_substitutes';dest.mkdir(exist_ok=True,parents=True)
zip_path=Path(r'E:\FRED\1.zip')
with zipfile.ZipFile(zip_path) as archive:
    entries=archive.infolist();wanted={'Event/events.raw':'events.raw','coordinates.txt':'coordinates.txt','tracks.txt':'tracks.txt'}
    records=[]
    for member,filename in wanted.items():
        info=archive.getinfo(member);target=dest/'fred_sequence1'/filename;target.parent.mkdir(exist_ok=True,parents=True)
        if not target.exists():
            with archive.open(info) as source,target.open('wb') as f:shutil.copyfileobj(source,f)
        records.append(dict(member=member,destination=str(target),bytes=target.stat().st_size,zip_crc32=info.CRC,
                            sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    # Show provided event image near the first annotation, using its actual
    # filename timestamp, not a prediction or a reconstructed video frame.
    frames=[i for i in entries if i.filename.startswith('Event/Frames/') and i.filename.endswith('.png')]
    chosen=min(frames,key=lambda i:abs(int(Path(i.filename).stem.split('_')[-1])/1e6-5.099949))
    target=dest/'fred_sequence1/provided_event_frame.png';target.write_bytes(archive.read(chosen))
    save(dest/'fred_inventory.json',dict(zip=str(zip_path),zip_bytes=zip_path.stat().st_size,sequence='1',
        official_split='train',role='Public development data-format/transfer validation; NOT official test performance.',
        source='https://github.com/miccunifi/FRED',raw_and_annotations=records,
        provided_frame_member=chosen.filename,annotation_sample=(dest/'fred_sequence1/coordinates.txt').read_text().splitlines()[:5],
        annotation_scope='Public image-plane boxes; not propeller RPM, thrust, mass or attitude truth.',
        coordinate_policy='Use provided Event-coordinate boxes; verify native RAW/image dimensions and timestamp alignment before scoring.'))
known=OUT.parent/'paper_reproduction_v030/eeppr_benchmark'
for name in ('velcro_front_result.json','velcro_side_result.json','combined_results.json','sdk_bridge_validation.json'):
    if (known/name).exists():shutil.copy2(known/name,dest/name)
save(dest/'replacement_plan.json',dict(user_has_no_synchronized_hardware_truth=True,
    priorities=[dict(task='2D future position',source='Existing FRED RAW plus provided annotations',available_now=True),
                dict(task='Generic periodicity',source='Existing EEPPR public RAW and GT frequencies, original-code results already validated',available_now=True),
                dict(task='Real four-rotor RPM with ego-motion',source='TQE',available_now=False,
                     access='Official HelixTrack README requires author request; no email has been sent.'),
                dict(task='Identifiability and error-injection mechanism',source='Frozen controlled simulation',available_now=True),
                dict(task='Real unknown-target thrust/attitude',source='Not covered by existing public material inspected here',available_now=False)],
    inference_policy='Independent labels/sensor ground truth used only for evaluation. Do not call estimates made by this same model independent truth.',
    stage_order='Complete controlled and RAW-deletion Stage1 report before Stage2 public-data inference; this script only prepares existing files.'))
print(json.dumps(dict(prepared=str(dest),fred_members=len(records),has_raw=True,has_position_annotations=True)),flush=True)

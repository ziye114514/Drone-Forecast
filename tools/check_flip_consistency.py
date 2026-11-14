"""Geometry/gradient checks and one real training-image forward before the run."""
from pathlib import Path
import json,sys,hashlib,os
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT.parent;sys.path.insert(0,str(ROOT/'core'));sys.path.insert(0,str(ROOT))
OUT=WORK/'output/flip_consistency_20261026'
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(OUT/'local_config'))
import torch,cv2,numpy as np
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.cfg import get_cfg
from consistency_detector import paired_batch,matched_flip_loss,ConsistencyDetectionModel
torch.set_num_threads(2)
batch=dict(img=torch.zeros((1,3,64,96)),batch_idx=torch.tensor([0.]),bboxes=torch.tensor([[.25,.5,.2,.3]]),cls=torch.zeros(1,1))
paired=paired_batch(batch)
assert torch.equal(paired['img'][1],batch['img'][0].flip(-1))
assert torch.allclose(paired['bboxes'][1],torch.tensor([.75,.5,.2,.3]))
class Head:
    stride=torch.tensor([8.])
    def _inference(self,x):return x
a=torch.tensor([[[24.],[32.],[19.2],[19.2],[.9]]],requires_grad=True)
z=torch.tensor([[[72.],[32.],[19.2],[19.2],[.9]]],requires_grad=True)
value,count=matched_flip_loss(Head(),a,z,batch);assert count==1 and float(value.detach())==0
offset=z.detach().clone();offset[:,0]-=2;offset.requires_grad_(True)
loss,count=matched_flip_loss(Head(),a,offset,batch);assert count==1 and float(loss.detach())>0
loss.backward();assert float(a.grad.abs().sum())>0 and float(offset.grad.abs().sum())>0
missing=z.detach().clone();missing[:,4]=.1
value,count=matched_flip_loss(Head(),a,missing,batch);assert count==0 and float(value.detach())==0
design=json.loads((OUT/'design.json').read_text('utf-8'))
parent=YOLO(design['parent_weight'],task='detect').model.float().cuda()
model=ConsistencyDetectionModel(parent.yaml,nc=1,ch=3,verbose=False).float().cuda();model.load_state_dict(parent.state_dict(),strict=True)
model.args=get_cfg();model.train();model.consistency_weight=1.
images=sorted((WORK/'output/roi_augmented_detector_20261026/dataset/train/images').glob('*'))
images=[p for p in images if (p.parents[1]/'labels'/(p.stem+'.txt')).read_text().strip()]
im=cv2.imread(str(images[0]));x=LetterBox(new_shape=(384,640),auto=False)(image=im)
height,width=im.shape[:2];gain=min(640/width,384/height);padx=(640-width*gain)/2;pady=(384-height*gain)/2
label=images[0].parents[1]/'labels'/(images[0].stem+'.txt')
labels=np.loadtxt(label,ndmin=2);boxes=labels[:,1:].copy()
boxes[:,0]=(boxes[:,0]*width*gain+padx)/640;boxes[:,1]=(boxes[:,1]*height*gain+pady)/384
boxes[:,2]*=width*gain/640;boxes[:,3]*=height*gain/384
batch=dict(img=torch.from_numpy(np.ascontiguousarray(x[:,:,::-1].transpose(2,0,1)[None])).float().cuda()/255,
    batch_idx=torch.zeros(len(boxes),device='cuda'),cls=torch.tensor(labels[:,:1],dtype=torch.float32,device='cuda'),
    bboxes=torch.tensor(boxes,dtype=torch.float32,device='cuda'))
loss,items=model.loss(batch);assert torch.isfinite(loss).all();loss.sum().backward()
assert sum(p.grad is not None for p in model.parameters())>0
report=dict(reflection_exact=True,matched_zero_loss=True,disagreement_nonzero_gradient=True,background_omitted=True,
    real_training_image=str(images[0]),real_forward_and_backward=True,auxiliary=model.consistency_audit,
    parent_sha256=hashlib.sha256(Path(design['parent_weight']).read_bytes()).hexdigest())
(OUT/'mechanism_checks.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

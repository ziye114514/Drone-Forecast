"""Training-only flip consistency for the existing Ultralytics detector.

Related method: https://github.com/soo89/CSD-SSD (NeurIPS 2019).
This is a small adaptation for the installed YOLO head, not copied author code
or a reproduction of CSD/TEDNet. Exported deployment is an ordinary detector.
"""
import torch
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import RANK


def split_predictions(preds, start, end):
    return {k: [x[start:end] for x in v] if isinstance(v, list) else v[start:end]
            for k, v in preds.items()}


def paired_batch(batch):
    """Same cutoff, exact horizontal reflection; no temporal target smoothing."""
    b = len(batch['img'])
    result = dict(batch)
    result['img'] = torch.cat((batch['img'], batch['img'].flip(-1)), 0)
    result['batch_idx'] = torch.cat((batch['batch_idx'], batch['batch_idx'] + b))
    flipped_boxes = batch['bboxes'].clone()
    flipped_boxes[:, 0] = 1 - flipped_boxes[:, 0]
    result['bboxes'] = torch.cat((batch['bboxes'], flipped_boxes), 0)
    result['cls'] = torch.cat((batch['cls'], batch['cls']), 0)
    return result


def box_iou_to_one(boxes, target):
    xy1 = boxes[:, :2] - boxes[:, 2:] / 2
    xy2 = boxes[:, :2] + boxes[:, 2:] / 2
    t1, t2 = target[:2] - target[2:] / 2, target[:2] + target[2:] / 2
    overlap = (torch.minimum(xy2, t2) - torch.maximum(xy1, t1)).clamp(min=0).prod(-1)
    return overlap / (boxes[:, 2:].clamp(min=0).prod(-1) + target[2:].prod() - overlap + 1e-7)


def matched_flip_loss(head, original, reflected, batch):
    """Compare the strongest same-object boxes; omit background/unmatched pairs.

    Selection is detached; the selected decoded boxes retain their gradients.
    Class score and overlap thresholds are fixed before the first experiment.
    Pixel residuals use the finest detector stride, independent of target size.
    """
    a, z = head._inference(original), head._inference(reflected)
    h, w = batch['img'].shape[-2:]
    z = torch.cat((w - z[:, :1], z[:, 1:]), 1)
    scale = a.new_tensor([w, h, w, h])
    stride = float(head.stride.min())
    terms = []
    for j, target in zip(batch['batch_idx'].long(), batch['bboxes'] * scale):
        aa, zz = a[j, :4].T, z[j, :4].T
        with torch.no_grad():
            ka, kz = a[j, 4].argmax(), z[j, 4].argmax()
            eligible = (a[j, 4, ka] >= .25) & (z[j, 4, kz] >= .25)
            eligible &= box_iou_to_one(aa[ka:ka+1], target)[0] >= .5
            eligible &= box_iou_to_one(zz[kz:kz+1], target)[0] >= .5
        if bool(eligible):
            terms.append(F.smooth_l1_loss(aa[ka] / stride, zz[kz] / stride, beta=1.))
    value = torch.stack(terms).mean() if terms else a.sum() * 0
    return value, len(terms)


class ConsistencyDetectionModel(DetectionModel):
    consistency_weight = 1.

    def loss(self, batch, preds=None):
        if not self.training or preds is not None:
            return super().loss(batch, preds)
        if getattr(self, 'criterion', None) is None:
            self.criterion = self.init_criterion()
        assert not getattr(self, 'end2end', False), 'This pilot targets the existing one-to-many YOLO head.'
        bs = len(batch['img'])
        paired = paired_batch(batch)
        predictions = self.forward(paired['img'])
        total, items = self.criterion(predictions, paired)
        total = total * .5
        items = {name: value.clone() for name, value in items.items()}
        # The zero-weight control executes the same branch and same paired BN pass.
        auxiliary, count = matched_flip_loss(self.model[-1],
            split_predictions(predictions, 0, bs), split_predictions(predictions, bs, 2*bs), batch)
        total = total.clone()
        total[0] = total[0] + bs * self.consistency_weight * auxiliary
        first = next(iter(items))
        items[first] = items[first] + self.consistency_weight * auxiliary.detach()
        self.consistency_audit = dict(loss=float(auxiliary.detach()), pairs=count, targets=len(batch['bboxes']))
        return total, items


class ConsistencyTrainer(DetectionTrainer):
    consistency_weight = 1.

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = self.set_model_names_for_load(ConsistencyDetectionModel(
            cfg, nc=self.data['nc'], ch=self.data['channels'], verbose=verbose and RANK == -1))
        if weights:
            model.load(weights)
        model.consistency_weight = self.consistency_weight
        return model

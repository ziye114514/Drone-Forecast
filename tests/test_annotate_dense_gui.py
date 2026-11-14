"""Headless self-test of the dense labelling GUI's editing logic.

Builds a withdrawn Tk root, drives store / navigate / clear / undo / accept_model
the same way the mouse and keys do, and checks what lands in labels.json. No
window is shown and no RAW is decoded.

This is a manual harness, not a unittest case: it needs a prepared case directory
and a display. Imports happen inside main() so that `unittest discover` can import
this module on a machine without tkinter (CI runs it that way).
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    try:
        import tkinter as tk
    except ImportError:
        print(json.dumps(dict(ok=False, skipped='tkinter is not available')))
        return
    sys.path.insert(0, str(ROOT / 'tools'))
    from annotate_dense_gui import Labeller

    case = Path(sys.argv[1]).resolve()
    label_path = case / 'labels.json'
    if label_path.exists():
        label_path.unlink()
    root = tk.Tk()
    root.withdraw()
    app = Labeller(root, case, (640, 360))

    assert app.position == 0, app.position
    assert (case / 'labels.json').exists() is False

    # draw a box on frame 0 like finish_box does
    app.store(0, dict(index=0, t=app.frames[0]['t'], state='present', box=[10.0, 20.0, 110.0, 90.0],
                      decision_source='operator_draw', complete=True))
    assert json.loads(label_path.read_text(encoding='utf-8'))['records']['0']['box'] == [10.0, 20.0, 110.0, 90.0]

    # Enter advances, then mark frame 1 absent like the operator would
    app.navigate(1)
    assert app.position == 1
    app.set_state('absent')
    saved = json.loads(label_path.read_text(encoding='utf-8'))
    assert saved['records']['1']['state'] == 'absent' and saved['records']['1']['box'] is None
    assert saved['records']['0']['box'] == [10.0, 20.0, 110.0, 90.0], 'frame 0 must stay intact'

    # accept a model box where one exists
    model_frame = next((f for f in app.frames if f.get('model_bbox')), None)
    assert model_frame is not None
    app.position = model_frame['index']
    before = app.position
    app.accept_model()
    saved = json.loads(label_path.read_text(encoding='utf-8'))
    record = saved['records'][str(before)]
    assert record['decision_source'] == 'model_prefill_accepted'
    assert record['box'] == [round(float(v), 1) for v in model_frame['model_bbox']]
    assert app.position == before + 1, 'accept_model must advance'

    # resume: reopening lands on the first unlabelled frame, not on zero
    root.destroy()
    root2 = tk.Tk()
    root2.withdraw()
    app2 = Labeller(root2, case, (640, 360))
    labelled = set(app2.labels['records'])
    expected = next(index for index in range(len(app2.frames)) if str(index) not in labelled)
    assert app2.position == expected, (app2.position, expected)

    # clear then undo restores the record
    app2.position = int(next(iter(labelled)))
    app2.clear_box()
    assert str(app2.position) not in json.loads(label_path.read_text(encoding='utf-8'))['records']
    app2.undo()
    assert str(app2.position) in json.loads(label_path.read_text(encoding='utf-8'))['records']
    root2.destroy()
    print(json.dumps(dict(ok=True, frames=len(app.frames), records=len(labelled),
                          resumed_at=expected), ensure_ascii=False))


if __name__ == '__main__':
    main()

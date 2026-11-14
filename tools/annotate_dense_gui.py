"""Continuous dense labelling GUI for event frames (Tkinter).

Interaction follows the design of the existing UAV annotation GUI in the user's
other project: left-drag draws a box, right-drag pans, the wheel zooms, a state
selector sits next to the image, the detector's own boxes can be shown, and
Enter saves and advances. Additions that matter for a *continuous* pass:

  * model prefill: press A to accept the detector box for this frame, which keeps
    a 20 Hz pass to roughly one key per frame;
  * ghost of the previous labelled box, for continuity across frames;
  * playback (space) so the segment can be watched before labelling it;
  * atomic save after every change plus resume at the first unlabelled frame;
  * a frame is never silently turned into "absent": leaving it empty keeps it
    unlabelled so a second pass can pick it up.

Usage:
    python tools/annotate_dense_gui.py --case output/dense_labels_train6_20_35
"""
from pathlib import Path
import argparse
import json
import os
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk

STATES = {'present': '有目标', 'absent': '无目标', 'unjudgeable': '无法判断'}
STATE_KEYS = {'1': 'present', '2': 'absent', '3': 'unjudgeable'}
BOX_COLORS = {'model': '#2ecc71', 'ghost': '#f1c40f', 'current': '#e74c3c'}


def atomic_write(path, payload):
    path = Path(path)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix='.tmp')
    with os.fdopen(handle, 'w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


class Labeller:
    def __init__(self, root, case, canvas_size=(1180, 664)):
        self.root = root
        self.case = Path(case)
        self.manifest = json.loads((self.case / 'manifest.json').read_text(encoding='utf-8'))
        self.frames = self.manifest['frames']
        self.width = self.manifest['width']
        self.height = self.manifest['height']
        self.label_path = self.case / 'labels.json'
        if self.label_path.exists():
            self.labels = json.loads(self.label_path.read_text(encoding='utf-8'))
        else:
            self.labels = dict(schema_version=1, usage='dense continuous hand labels',
                               raw=self.manifest['raw'], start_s=self.manifest['start_s'],
                               fps=self.manifest['fps'],
                               coordinate_system=self.manifest['coordinate_system'],
                               size=[self.width, self.height], records={})
        self.position = int(self.labels.get('last_position', 0))
        self.canvas_size = canvas_size
        self.scale = min(canvas_size[0] / self.width, canvas_size[1] / self.height)
        self.center = [self.width / 2, self.height / 2]
        self.drag_start = None
        self.drag_current = None
        self.pan_start = None
        self.pan_center = None
        self.show_model = tk.BooleanVar(value=True)
        self.show_ghost = tk.BooleanVar(value=True)
        self.playing = False
        self.undo_stack = []
        self.cache = {}
        self.photo = None
        self.build()
        self.jump_to_first_unlabelled()

    # ---------- layout ----------
    def build(self):
        self.root.title(f'连续密集标注 — {self.case.name}')
        self.canvas = tk.Canvas(self.root, width=self.canvas_size[0], height=self.canvas_size[1],
                                background='#101418', highlightthickness=0)
        self.canvas.grid(row=0, column=0, columnspan=8, padx=6, pady=4)
        self.canvas.bind('<ButtonPress-1>', self.begin_box)
        self.canvas.bind('<B1-Motion>', self.drag_box)
        self.canvas.bind('<ButtonRelease-1>', self.finish_box)
        self.canvas.bind('<ButtonPress-3>', self.begin_pan)
        self.canvas.bind('<B3-Motion>', self.do_pan)
        self.canvas.bind('<ButtonRelease-3>', self.end_pan)
        self.canvas.bind('<MouseWheel>', self.zoom)

        row = ttk.Frame(self.root)
        row.grid(row=1, column=0, columnspan=8, sticky='we', padx=6)
        ttk.Button(row, text='◀ 上一帧 (←)', command=lambda: self.navigate(-1)).pack(side='left')
        ttk.Button(row, text='保存并下一帧 (Enter)', command=lambda: self.navigate(1)).pack(side='left', padx=4)
        ttk.Button(row, text='-10帧', command=lambda: self.navigate(-10)).pack(side='left')
        ttk.Button(row, text='+10帧', command=lambda: self.navigate(10)).pack(side='left', padx=4)
        ttk.Button(row, text='跳到未标注', command=self.jump_to_first_unlabelled).pack(side='left', padx=8)
        self.play_button = ttk.Button(row, text='播放 (空格)', command=self.toggle_play)
        self.play_button.pack(side='left')
        ttk.Label(row, text='  状态：').pack(side='left')
        self.state_var = tk.StringVar(value=STATES['present'])
        selector = ttk.Combobox(row, textvariable=self.state_var, width=10, state='readonly',
                                values=[STATES[key] for key in ('present', 'absent', 'unjudgeable')])
        selector.pack(side='left')
        selector.bind('<<ComboboxSelected>>', lambda _event: self.set_state_from_selector())
        ttk.Button(row, text='采纳模型框 (A)', command=self.accept_model).pack(side='left', padx=6)
        ttk.Button(row, text='清除框 (D)', command=self.clear_box).pack(side='left')
        ttk.Checkbutton(row, text='显示模型框 (M)', variable=self.show_model,
                        command=self.render).pack(side='left', padx=6)
        ttk.Checkbutton(row, text='显示上一帧框 (G)', variable=self.show_ghost,
                        command=self.render).pack(side='left')

        self.status = ttk.Label(self.root, anchor='w', justify='left', wraplength=self.canvas_size[0])
        self.status.grid(row=2, column=0, columnspan=8, sticky='we', padx=8, pady=(2, 2))
        self.hint = ttk.Label(self.root, anchor='w', justify='left', wraplength=self.canvas_size[0],
                              foreground='#39506b')
        self.hint.grid(row=3, column=0, columnspan=8, sticky='we', padx=8, pady=(0, 6))
        self.hint.config(text='左键拖拽画框 · 右键拖拽平移 · 滚轮缩放 · 1 有目标(自动采纳建议框) · '
                              '2 无目标并前进 · 3 无法判断 · A 采纳建议框(模型框或上一帧框) · D 清除 · '
                              'Enter 保存并下一帧 · ←/→ 前后一帧 · U 撤销 · 空格 播放 · 0 复位视图。'
                              '建议框是虚线（绿色=模型，黄色=上一帧）；留空直接下一帧＝未标注，不会被当成无目标。')

        self.root.bind('<Return>', lambda _event: self.navigate(1))
        self.root.bind('<Left>', lambda _event: self.navigate(-1))
        self.root.bind('<Right>', lambda _event: self.navigate(1))
        self.root.bind('<Escape>', lambda _event: self.cancel_drag())
        for key, state in STATE_KEYS.items():
            self.root.bind(key, lambda _event, s=state: self.set_state(s))
        self.root.bind('<a>', lambda _event: self.accept_model())
        self.root.bind('<d>', lambda _event: self.clear_box())
        self.root.bind('<u>', lambda _event: self.undo())
        self.root.bind('<g>', lambda _event: self.toggle(self.show_ghost))
        self.root.bind('<m>', lambda _event: self.toggle(self.show_model))
        self.root.bind('<0>', lambda _event: self.reset_view())
        self.root.bind('<space>', lambda _event: self.toggle_play())
        self.root.bind('<plus>', lambda _event: self.zoom_at(self.canvas_size[0] / 2,
                                                            self.canvas_size[1] / 2, 1.25))
        self.root.bind('<minus>', lambda _event: self.zoom_at(self.canvas_size[0] / 2,
                                                             self.canvas_size[1] / 2, 0.8))
        self.root.protocol('WM_DELETE_WINDOW', self.close)

    # ---------- image / view ----------
    def frame(self):
        return self.frames[self.position]

    def image(self):
        path = self.case / self.frame()['image']
        key = str(path)
        if key not in self.cache:
            if len(self.cache) > 6:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = Image.open(path).convert('RGB')
        return self.cache[key]

    def to_canvas(self, point):
        origin = self.view_origin()
        return ((point[0] - origin[0]) * self.scale, (point[1] - origin[1]) * self.scale)

    def to_image(self, point):
        origin = self.view_origin()
        return (point[0] / self.scale + origin[0], point[1] / self.scale + origin[1])

    def view_origin(self):
        width = self.canvas_size[0] / self.scale
        height = self.canvas_size[1] / self.scale
        x0 = min(max(self.center[0] - width / 2, 0), max(self.width - width, 0))
        y0 = min(max(self.center[1] - height / 2, 0), max(self.height - height, 0))
        return (x0, y0)

    def render(self):
        x0, y0 = self.view_origin()
        width = self.canvas_size[0] / self.scale
        height = self.canvas_size[1] / self.scale
        crop = self.image().crop((int(x0), int(y0), int(x0 + width), int(y0 + height)))
        self.photo = ImageTk.PhotoImage(crop.resize(self.canvas_size, Image.BILINEAR))
        self.canvas.delete('all')
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        record = self.record(self.position)
        previous = self.record(self.position - 1) if self.position > 0 else None
        if self.show_ghost.get() and previous and previous.get('box'):
            self.draw_box(previous['box'], BOX_COLORS['ghost'], dashed=True, label='上一帧')
        current = self.drag_current or (record or {}).get('box')
        if current:
            self.draw_box(current, BOX_COLORS['current'], label='本帧框')
        if self.show_model.get() and self.frame().get('model_bbox'):
            self.draw_box(self.frame()['model_bbox'], BOX_COLORS['model'], dashed=True, label='模型')
        self.update_status()

    def draw_box(self, box, color, dashed=False, label=''):
        x0, y0 = self.to_canvas((box[0], box[1]))
        x1, y1 = self.to_canvas((box[2], box[3]))
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2,
                                     dash=(4, 3) if dashed else None)
        if label:
            self.canvas.create_text(x0 + 2, max(y0 - 10, 10), anchor='w', text=label, fill=color)

    def update_status(self):
        record = self.record(self.position)
        done = len(self.labels['records'])
        present = sum(1 for item in self.labels['records'].values() if item['state'] == 'present')
        absent = sum(1 for item in self.labels['records'].values() if item['state'] == 'absent')
        state = '未标注' if not record else STATES[record['state']]
        source = '' if not record else {'operator_draw': '手绘', 'model_prefill_accepted': '采纳模型',
                                        'previous_box_accepted': '沿用上帧'
                                        }.get(record.get('decision_source'), record.get('decision_source', ''))
        model = self.frame().get('model_detected')
        remaining = len(self.frames) - done
        self.status.config(text=f'第 {self.position + 1}/{len(self.frames)} 帧 | t={self.frame()["t"]:.2f}s | '
                                f'本帧：{state} {source} | 已标 {done}（有目标 {present} / 无目标 {absent}）'
                                f' | 剩余 {remaining} | '
                                f'模型本帧：{"检出" if model else "无"}')

    # ---------- editing ----------
    def record(self, index):
        return self.labels['records'].get(str(index))

    def store(self, index, payload):
        self.undo_stack.append((index, self.labels['records'].get(str(index))))
        self.labels['records'][str(index)] = payload
        self.save()

    def save(self):
        self.labels['last_position'] = self.position
        atomic_write(self.label_path, self.labels)

    def begin_box(self, event):
        self.drag_start = self.to_image((event.x, event.y))
        self.drag_current = None

    def drag_box(self, event):
        if self.drag_start is None:
            return
        self.drag_current = [self.drag_start[0], self.drag_start[1],
                             self.to_image((event.x, event.y))[0], self.to_image((event.x, event.y))[1]]
        self.render()

    def finish_box(self, event):
        if self.drag_start is None or self.drag_current is None:
            return
        x0, x1 = sorted((self.drag_current[0], self.drag_current[2]))
        y0, y1 = sorted((self.drag_current[1], self.drag_current[3]))
        self.drag_start = None
        self.drag_current = None
        if x1 - x0 < 4 or y1 - y0 < 4:
            self.render()
            return
        x0 = max(0.0, x0)
        y0 = max(0.0, y0)
        x1 = min(float(self.width), x1)
        y1 = min(float(self.height), y1)
        self.store(self.position, dict(index=self.position, t=self.frame()['t'], state='present',
                                       box=[round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                                       decision_source='operator_draw', complete=True))
        self.state_var.set(STATES['present'])
        self.render()

    def cancel_drag(self):
        self.drag_start = None
        self.drag_current = None
        self.render()

    def proposal(self):
        """Model box for this frame, else the last accepted box (continuity proposal)."""
        box = self.frame().get('model_bbox')
        if box:
            return list(box), 'model_prefill_accepted'
        for index in range(self.position - 1, max(self.position - 4, -1), -1):
            record = self.record(index)
            if record and record.get('box'):
                return list(record['box']), 'previous_box_accepted'
        return None, None

    def accept_model(self):
        box, source = self.proposal()
        if not box:
            self.status.config(text=self.status.cget('text') + '  ← 本帧没有可用的建议框')
            return
        self.store(self.position, dict(index=self.position, t=self.frame()['t'], state='present',
                                       box=[round(float(v), 1) for v in box],
                                       decision_source=source, complete=True))
        self.state_var.set(STATES['present'])
        self.navigate(1)

    def clear_box(self):
        index = self.position
        if self.record(index):
            self.undo_stack.append((index, dict(self.labels['records'][str(index)])))
            self.labels['records'].pop(str(index))
            self.save()
        self.render()

    def set_state(self, state):
        self.state_var.set(STATES[state])
        record = self.record(self.position)
        if state == 'present':
            if record and record.get('box'):
                return
            box, source = self.proposal()
            if box:
                self.store(self.position, dict(index=self.position, t=self.frame()['t'],
                                               state='present',
                                               box=[round(float(v), 1) for v in box],
                                               decision_source=source, complete=True))
                self.navigate(1)
                return
            self.status.config(text=self.status.cget('text') + '  ← 有目标请画框或按 A 采纳建议框')
            return
        self.store(self.position, dict(index=self.position, t=self.frame()['t'], state=state, box=None,
                                       decision_source='operator_state', complete=True))
        # one key per frame: store and move on
        self.navigate(1)

    def set_state_from_selector(self):
        for key, text in STATES.items():
            if text == self.state_var.get():
                self.set_state(key)

    def undo(self):
        if not self.undo_stack:
            return
        index, payload = self.undo_stack.pop()
        if payload is None:
            self.labels['records'].pop(str(index), None)
        else:
            self.labels['records'][str(index)] = payload
        self.position = index
        self.save()
        self.render()

    # ---------- navigation ----------
    def navigate(self, delta):
        target = self.position + delta
        if target < 0 or target >= len(self.frames):
            return
        self.position = target
        self.save()
        self.render()

    def jump_to_first_unlabelled(self):
        for index in range(len(self.frames)):
            if str(index) not in self.labels['records']:
                self.position = index
                self.render()
                return
        self.render()

    def toggle(self, variable):
        variable.set(not variable.get())
        self.render()

    def reset_view(self):
        self.scale = min(self.canvas_size[0] / self.width, self.canvas_size[1] / self.height)
        self.center = [self.width / 2, self.height / 2]
        self.render()

    def zoom(self, event):
        self.zoom_at(event.x, event.y, 1.25 if event.delta > 0 else 0.8)

    def zoom_at(self, x, y, factor):
        before = self.to_image((x, y))
        self.scale = max(0.15, min(6.0, self.scale * factor))
        after = self.to_image((x, y))
        self.center[0] += before[0] - after[0]
        self.center[1] += before[1] - after[1]
        self.render()

    def begin_pan(self, event):
        self.pan_start = (event.x, event.y)
        self.pan_center = list(self.center)

    def do_pan(self, event):
        if self.pan_start is None:
            return
        dx = (self.pan_start[0] - event.x) / self.scale
        dy = (self.pan_start[1] - event.y) / self.scale
        self.center = [self.pan_center[0] + dx, self.pan_center[1] + dy]
        self.render()

    def end_pan(self, _event):
        self.pan_start = None

    def toggle_play(self):
        self.playing = not self.playing
        self.play_button.config(text='暂停 (空格)' if self.playing else '播放 (空格)')
        if self.playing:
            self.tick()

    def tick(self):
        if not self.playing:
            return
        if self.position + 1 >= len(self.frames):
            self.playing = False
            self.play_button.config(text='播放 (空格)')
            return
        self.position += 1
        self.render()
        self.root.after(120, self.tick)

    def close(self):
        if self.undo_stack:
            self.save()
        unlabelled = len(self.frames) - len(self.labels['records'])
        if unlabelled and not messagebox.askokcancel(
                '退出', f'还有 {unlabelled} 帧未标注，进度已保存。确定退出？'):
            return
        self.save()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    root = tk.Tk()
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    canvas_w = min(1500, int(screen_w * 0.74))
    canvas_h = int(canvas_w * 720 / 1280)
    if canvas_h > screen_h * 0.76:
        canvas_h = int(screen_h * 0.76)
        canvas_w = int(canvas_h * 1280 / 720)
    root.resizable(False, False)
    Labeller(root, args.case, (canvas_w, canvas_h))
    root.mainloop()


if __name__ == '__main__':
    main()

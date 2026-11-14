"""Postprocess completed frozen runs; no inference or protocol changes.

The original analysis serializer rejects numpy.int64 inside candidate sums.
Convert numpy scalars while saving, preserving the frozen analysis logic.
"""
import json
import numpy as np
import study


def save_scalars(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                              allow_nan=False,
                              default=lambda x: x.item() if isinstance(x, np.generic)
                              else x.tolist() if isinstance(x, np.ndarray)
                              else (_ for _ in ()).throw(TypeError(type(x).__name__))),
                    encoding='utf-8')


if __name__ == '__main__':
    study.verify()
    study.save = save_scalars
    study.analyze()

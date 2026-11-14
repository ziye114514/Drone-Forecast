"""Verify that every flat filename listed by the export/snapshot tools resolves.

Several tools keep module lists as flat names (`'live_decode.py'`, `'rotor_nvrtc.py'`)
because the same string doubles as the destination path inside the flat public
export. The sources themselves live in `core/`, `tools/`, or the repository root,
so the lists are only valid if a resolver can still find each one. That breaks
silently whenever a module moves, which is exactly what happened when 339 modules
were collected into `core/`.

This checks the lists statically (AST only, no imports of the target modules) so it
runs anywhere without CUDA, the Metavision SDK, or model weights.

    python tools/check_flat_names.py
"""
from pathlib import Path
import ast
import sys

ROOT = Path(__file__).resolve().parents[1]
SEARCH = (ROOT / 'core', ROOT / 'tools', ROOT)

# (file, module-level list/tuple constant) pairs that must stay resolvable.
TARGETS = (
    ('tools/prepare_github_release.py', 'COPY'),
    ('tools/prepare_github_release.py', 'TESTS'),
    ('core/run_rotor_only.py', 'SOURCE_FILES'),
    ('tools/replay_coupled_source168.py', 'SOURCES'),
    ('tools/replay_gated_nn166.py', 'SOURCES'),
    ('tools/run_force167.py', 'SOURCES'),
    ('tools/run_source_force168.py', 'SOURCES'),
)


def names_in(path, constant):
    """Every string element ever assigned to `constant`, including += extensions."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    out = []
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AugAssign) else []
        if not any(isinstance(t, ast.Name) and t.id == constant for t in targets):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue
        for element in node.value.elts:
            if isinstance(element, ast.Constant) and isinstance(element.value, str):
                out.append(element.value)
    return out


def resolve(name):
    for base in SEARCH:
        if (base / name).exists():
            return base / name
    return None


def main():
    missing, checked = [], 0
    for relative, constant in TARGETS:
        path = ROOT / relative
        if not path.exists():
            missing.append((relative, constant, '<file itself is missing>'))
            continue
        names = names_in(path, constant)
        if not names:
            missing.append((relative, constant, '<no literal entries found>'))
            continue
        for name in names:
            checked += 1
            if resolve(name) is None:
                missing.append((relative, constant, name))
    for relative, constant, name in missing:
        print(f'MISSING  {relative}  {constant}  {name}')
    print(f'checked {checked} flat name(s) across {len(TARGETS)} list(s); {len(missing)} unresolved')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())

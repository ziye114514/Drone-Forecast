"""Read the single selected Zotero attachment; do not modify the library."""
from pathlib import Path
import json,hashlib
import pdfplumber
ROOT=Path(__file__).resolve().parents[2]
out=ROOT/'output/ddhf_review_20261027'
selected=json.loads((out/'local_source.json').read_text('utf-8'))
source=Path(next(r['path'] for r in selected if r['content_type']=='application/pdf' and r['exists']))
with pdfplumber.open(source) as doc:
    pages=[p.extract_text(layout=False) or '' for p in doc.pages]
    (out/'paper_text.txt').write_text('\n\n'.join('PAGE '+str(i+1)+'\n'+s for i,s in enumerate(pages)),'utf-8')
    for i in [0,3,5]:
        if i<len(doc.pages):doc.pages[i].to_image(resolution=115).save(out/f'paper_page_{i+1}.png')
    print(json.dumps(dict(source=str(source),pages=len(pages),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),text_characters=sum(map(len,pages))),ensure_ascii=False))
# Dispose of the temporary full-library lookup copy, only this known local file.
lookup=(out/'zotero_title_lookup.sqlite').resolve()
assert lookup.parent==out.resolve()
if lookup.exists():lookup.unlink()

"""Review frozen external images without reading either forecast output."""
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=ROOT.parent/'output/frozen_external518_20261113/reference'
    files=sorted((f for f in p.glob('*.png') if f.stem.replace('.','',1).isdigit()),key=lambda f:float(f.stem))
    assert len(files)==27
    for page,start in enumerate(range(0,len(files),8)):
        panel=Image.new('RGB',(1280,1540),'white');draw=ImageDraw.Draw(panel)
        for i,f in enumerate(files[start:start+8]):
            x=i%2*640;y=i//2*385
            panel.paste(Image.open(f).convert('RGB').resize((640,360)),(x,y+25))
            draw.text((x+5,y+4),f'{f.stem}s native events; no predictions',fill='black')
        panel.save(p/f'review_page{page}.png')
    print('27 native frames in4 pages; no predictor outputs read.')

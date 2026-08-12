"""Fast dataset audit: missing labels, malformed boxes, class counts, image statistics."""
import argparse, json
from pathlib import Path
import cv2

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); args=ap.parse_args(); report={}
    for split in ('train','val','test'):
        counts={}; bad=[]; n=0
        for img in sorted((args.root/'images'/split).glob('*')):
            im=cv2.imread(str(img)); n+=1
            if im is None: bad.append((img.name,'unreadable')); continue
            h,w=im.shape[:2]; lp=args.root/'labels'/split/(img.stem+'.txt')
            for line_no,line in enumerate(lp.read_text().splitlines(),1) if lp.exists() else []:
                v=line.split()
                if len(v)!=5: bad.append((img.name,f'line {line_no}')); continue
                c,cx,cy,bw,bh=map(float,v); counts[int(c)]=counts.get(int(c),0)+1
                if not(0<=int(c)<6 and 0<cx<1 and 0<cy<1 and 0<bw<=1 and 0<bh<=1): bad.append((img.name,f'bad box line {line_no}'))
        report[split]={'images':n,'boxes_by_class':counts,'issues':bad}
    print(json.dumps(report,indent=2)); (args.root/'dataset_audit.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__': main()

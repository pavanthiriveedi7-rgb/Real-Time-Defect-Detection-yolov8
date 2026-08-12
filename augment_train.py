"""Create augmented YOLO training images. Validation/test are never augmented."""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import cv2, numpy as np
import albumentations as A

def read_yolo(p, w, h):
    boxes=[]; labels=[]
    if not p.exists(): return boxes, labels
    for line in p.read_text().splitlines():
        v=line.split();
        if len(v)!=5: continue
        c,cx,cy,bw,bh=map(float,v); labels.append(int(c))
        boxes.append([(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h])
    return boxes, labels

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True); ap.add_argument('--copies',type=int,default=1); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    rng=np.random.default_rng(args.seed)
    aug=A.Compose([A.OneOf([A.RandomBrightnessContrast(brightness_limit=.15,contrast_limit=.15,p=1),A.GaussNoise(std_range=(.02,.08),p=1)],p=.7),A.Affine(scale=(.95,1.05),translate_percent=(-.04,.04),rotate=(-7,7),shear=(-3,3),p=.6),A.MotionBlur(blur_limit=3,p=.08)], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=4,min_visibility=.25,clip=True))
    imgs=sorted((args.root/'images'/'train').glob('*')); count=0
    for path in imgs:
        im=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if im is None: continue
        h,w=im.shape[:2]; boxes, labels=read_yolo(args.root/'labels'/'train'/(path.stem+'.txt'),w,h)
        for k in range(args.copies):
            r=aug(image=im,bboxes=boxes,class_labels=labels); out_name=f'{path.stem}_aug{k:02d}{path.suffix}'
            cv2.imwrite(str(path.parent/out_name),r['image'],[cv2.IMWRITE_JPEG_QUALITY,95])
            rh,rw=r['image'].shape[:2]; lines=[]
            for b,c in zip(r['bboxes'],r['class_labels']):
                x1,y1,x2,y2=b; lines.append(f'{c} {((x1+x2)/2)/rw:.6f} {((y1+y2)/2)/rh:.6f} {(x2-x1)/rw:.6f} {(y2-y1)/rh:.6f}')
            (args.root/'labels'/'train'/(Path(out_name).stem+'.txt')).write_text('\n'.join(lines)+'\n' if lines else '')
            count+=1
    print(f'Created {count} augmented training images')
if __name__=='__main__': main()

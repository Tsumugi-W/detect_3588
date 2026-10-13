#!/usr/bin/env python3
"""Annotate images with raw YOLO predictions, without Tag or layout correction."""
import argparse
import json
from pathlib import Path
import random
import sys
import tempfile
import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src' / 'panel_detection'))
from panel_detection.default_config import DEFAULT_CONFIG
from panel_detection.panel_scene import merge_config
from panel_detection.detector_onnx import YoloV5ORT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--crop', nargs=4, type=int, metavar=('X','Y','WIDTH','HEIGHT'))
    args = parser.parse_args()
    files = sorted(p for p in args.image_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.bmp'))
    if not files:
        parser.error('No images found')
    destination = args.output_dir or args.image_dir/'yolo_raw'
    if destination.exists():
        parser.error(f'Refusing to overwrite: {destination}')
    cfg = merge_config(DEFAULT_CONFIG, yaml.safe_load(args.config.read_text()) or {} if args.config else {})
    model = Path(cfg['onnx_model'])
    if not model.is_absolute():
        model = ROOT/'src/panel_detection/panel_detection'/model
    random.seed(0)
    with tempfile.TemporaryDirectory(prefix='yolo-images-') as temp:
        config = Path(temp)/'config.yaml';config.write_text(yaml.safe_dump(cfg))
        detector = YoloV5ORT(str(model), str(config), threads=cfg['onnx_threads'])
    destination.mkdir(parents=True)
    counts = {};with_detections = 0
    with (destination/'predictions.jsonl').open('w') as log:
        for index, path in enumerate(files):
            frame = cv2.imread(str(path))
            if frame is None:
                raise RuntimeError(f'Cannot read {path}')
            height,width = frame.shape[:2]
            x,y,w,h = args.crop or (0,0,width,height)
            if x<0 or y<0 or w<=0 or h<=0 or x+w>width or y+h>height:
                raise ValueError(f'Invalid crop for {path}')
            annotated, ids, boxes, scores = detector.detect(frame[y:y+h,x:x+w])
            output = frame.copy();output[y:y+h,x:x+w] = annotated
            # Header goes in a new band; JSON coordinates remain relative to the original image.
            output = cv2.copyMakeBorder(output,32,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
            cv2.putText(output, 'RAW YOLO | '+path.name, (8,22), cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
            if not cv2.imwrite(str(destination/(path.stem+'.png')),output):
                raise RuntimeError(f'Cannot save {path.name}')
            predictions=[]
            for class_id,box,score in zip(ids,boxes,scores):
                name=detector.class_names[class_id];counts[name]=counts.get(name,0)+1
                predictions.append({'class_id':int(class_id),'class':name,'confidence':float(score),
                    'bbox_xyxy':[box[0]+x,box[1]+y,box[2]+x,box[3]+y]})
            with_detections+=bool(predictions)
            log.write(json.dumps({'image':path.name,'width':width,'height':height,'predictions':predictions},ensure_ascii=False)+'\n')
            if (index+1)%50==0:print(f'{index+1}/{len(files)} images',flush=True)
    summary={'images':len(files),'images_with_detections':with_detections,'prediction_counts':counts,
             'model':str(model.resolve()),'threshold':cfg['threshold'],'crop_xywh':args.crop,
             'coordinate_space':'original image; annotated PNG adds a 32px header',
             'class_names':detector.class_names,'tag_correction':False,
             'note':'Predictions only, not ground truth. No accuracy metric computed.'}
    (destination/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()

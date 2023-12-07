# yolo2coco

A tool for converting YOLO instance segmentation annotations to COCO format.

## Dependencies

```
geos>=3.11.0
matplotlib
opencv
pycocotools
shapely>=2.0
tqdm
```

## Convert COCO annotation json to instance segmentation masks for YoloV8

```
coco2yoloseg.py -o ./demo/yolo_masks/train/ demo/cvat_coco/cvat_coco2.json
```
It will convert the json into YoloV8 txts into the output directory.

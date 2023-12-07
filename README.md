# yolo2coco

A tool for converting YOLO instance segmentation annotations to COCO format.

Currently, the popular COCO and YOLO annotation format conversion tools are almost all aimed at object detection tasks, and there is no specific tool for instance segmentation tasks. The annotation format for YOLO instance segmentation differs greatly from that for object detection. Therefore, this code toolkit was developed.

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

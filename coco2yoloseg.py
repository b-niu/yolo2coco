import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import shapely
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from tqdm.auto import tqdm


def show_img(img):
    if img.ndim == 2:
        # 单通道图
        plt.imshow(img)
    else:
        plt.imshow(img[:, :, ::-1])
    plt.show()


def concave_hull(binary_img, ratio=0.05, debug=False):
    """_summary_

    Args:
        binary_img (numpy.ndarray): _description_
        ratio (int, optional): _description_. Defaults to 0.
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        numpy array: _description_
    """
    # 输入：二值化图像。前景白色，背景黑色。
    # 输出：新的轮廓坐标。不考虑原点。原点由其它程序做后处理。

    # 将连通域的点的(x, y)坐标整理成numpy.array
    ys, xs = np.where(binary_img > 0)  # 注意顺序
    # 对连通域求凹包：
    points = []
    for xi, yi in zip(xs, ys):
        points.append((xi, yi))
    # 使用shapely进行后续处理
    Points = shapely.geometry.MultiPoint(points)
    if debug:
        Points
    # 求凹包
    res = shapely.concave_hull(Points, ratio=ratio)
    xl, yl = res.exterior.coords.xy

    pts = []
    for xi, yi in zip(xl, yl):
        xi = round(xi)
        yi = round(yi)
        pts.append((xi, yi))
    # 整理成numpy array
    pts = np.array(pts, np.int32).reshape(1, -1, 2)
    return pts


def coco_to_yolo(
    coco_json="demo/cvat_coco/cvat_coco2.json", output_dir="demo/yolo_masks/train"
):
    coco_dict = COCO(coco_json)
    img_ids = coco_dict.getImgIds()

    os.makedirs(output_dir, exist_ok=True)

    for img_id in tqdm(img_ids, desc="Images"):
        img_infos = coco_dict.loadImgs(img_id)
        img_info = img_infos[0]
        h = img_info["height"]
        w = img_info["width"]
        img_name = img_info["file_name"]
        yolo_txt = os.path.join(output_dir, Path(img_name).with_suffix(".txt").name)
        annot_ids = coco_dict.getAnnIds(imgIds=img_id)
        annots = coco_dict.loadAnns(annot_ids)

        with open(yolo_txt, "w") as f:
            for annot in tqdm(annots, desc="Annotations", leave=False):
                # 获取分类号
                cls_id = annot["category_id"]
                # 整理坐标
                mask = coco_dict.annToMask(annot)
                # print(np.unique(mask, return_counts=True))
                # show_img(mask)
                contours, hierarchy = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if len(contours) > 1:
                    # show_img(mask)
                    pts = concave_hull(binary_img=mask, ratio=0.02)
                    # print(pts)
                    # new_mask = mask.copy()
                    # cv2.polylines(new_mask, pts, True, 128, 2)
                    # show_img(new_mask)
                else:
                    pts = contours[0]
                    pts = np.array(pts, np.int32).reshape(1, -1, 2)
                    # print(pts)
                    # new_mask = mask.copy()
                    # cv2.polylines(new_mask, pts, True, 128, 2)
                    # show_img(new_mask)
                # 对坐标进行处理
                # print(pts)
                # pts的shape为(1, -1, 2)
                yolo_points = []
                for x, y in pts[0]:
                    yolo_points.extend((str(x / w), str(y / h)))
                # 得到yolo的标注行：
                yolo_line = str(cls_id) + " " + " ".join(yolo_points) + "\n"
                # print(yolo_line)
                f.write(yolo_line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some weights and images.")
    parser.add_argument("coco_json", nargs=1, default=None, type=str)
    parser.add_argument("-o", "--output_dir", type=str)
    args = parser.parse_args()
    coco_to_yolo(coco_json=args.coco_json, output_dir=args.output_dir)

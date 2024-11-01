# 展示图像
from PIL import Image, ImageDraw
from sahi.utils.file import load_json
import matplotlib.pyplot as plt
import os

# coco图像集地址
image_path = "F:/data/origion"
# coco标注文件
coco_annotation_file_path="F:/datasets/json/coco.json"
# 加载数据集
coco_dict = load_json(coco_annotation_file_path)

f, axarr = plt.subplots(1, 1, figsize=(8, 8))
# 读取图像
img_ind = 0
img = Image.open(os.path.join(image_path,coco_dict["images"][img_ind]["file_name"])).convert('RGBA')
# 绘制标注框
for ann_ind in range(len(coco_dict["annotations"])):
    xywh = coco_dict["annotations"][ann_ind]["bbox"]
    xyxy = [xywh[0], xywh[1], xywh[0] + xywh[2], xywh[1] + xywh[3]]
    ImageDraw.Draw(img, 'RGBA').rectangle(xyxy, width=5)
axarr.imshow(img)


from sahi.slicing import slice_coco

# 保存的coco数据集标注文件名
output_coco_annotation_file_name="sliced"
# 输出文件夹
output_dir = "result"

# 切分数据集
coco_dict, coco_path = slice_coco(
    coco_annotation_file_path=coco_annotation_file_path,
    image_dir=image_path,
    output_coco_annotation_file_name=output_coco_annotation_file_name,
    ignore_negative_samples=True,
    output_dir=output_dir,
    slice_height=1280,
    slice_width=1280,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    min_area_ratio=0.2,
    verbose=False
)

print("切分子图{}张".format(len(coco_dict['images'])))
print("获得标注框{}个".format(len(coco_dict['annotations'])))

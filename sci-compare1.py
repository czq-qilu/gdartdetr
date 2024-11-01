import matplotlib.pyplot as plt
# 设置字体格式
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

size = 8  # 全局字体大小
# 设置英文字体
config = {
    "font.family": 'serif',
    "font.size": size,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
}
rcParams.update(config)
# 设置中文宋体
fontcn = {'family': 'SimSun', 'size': size}
label_size = size
text_size = size

# 数据1
rtdetr_mAP = [76.2, 77.5, 79.3, 80.5]
rtdetr_flop = [58.3, 88.6, 134.8, 257.7]

yolov8_mAP = [74.1,75.7, 78.2, 79.5]
yolov8_flop = [28.8, 79.3, 165.7, 258.5]

CANet_mAP = [75.4]
CANet_flop = [209]

yolov5_mAP = [72.2, 73.9, 74.8, 76.2]
yolov5_flop = [16.5, 49.0, 109.1, 205.7]

yolovx_mAP = [72.9,74.8, 77.4, 78.8]
yolovx_flop = [16.5, 49.0, 109.1, 205.7]

faster_mAP = [72.2, 73.2]
faster_flop = [91.4, 140]

rddyolo_mAP = [81.1]
rddyolo_flop = [123.2]

my_mAP = [79.3, 81.1, 82.4, 83.5]
my_flop = [51.2, 81.4, 103.7, 180.9]

# 参数设置
lw = 1
ms = 6
yolov8_text = ['S', 'M', 'L', 'X']
rtdetr_text = ['R18', 'R34', 'R50', 'R101']
CANet_text = ['CANet']
yolov5_text = ['S', 'M', 'L', 'X']
yolovx_text = ['S', 'M', 'L', 'X']
faster_text = ['VGG16', 'R101']
rddyolo_text = ['RDD-YOLO']
my_text = ['R18', 'R34', 'R50', 'R101']

# 绘制 mAP-Param
plt.figure(figsize=(6.4, 4.8))
plt.plot(yolov8_flop, yolov8_mAP, label='YOLOV8',
         c='coral',
         lw=lw,
         marker='o',
         markersize=ms,
         ls='--')
plt.plot(rtdetr_flop, rtdetr_mAP, label='RT-DETR',
         c='cadetblue',
         lw=lw,
         marker='v',
         markersize=ms,
         ls='--')
plt.plot(CANet_flop, CANet_mAP, label='CANet',
         c='dimgray',
         lw=lw,
         marker='*',
         markersize=10,
         ls='--')
plt.plot(yolov5_flop, yolov5_mAP, label='YOLOV5',
         c='orchid',
         lw=lw,
         marker='h',
         markersize=ms,
         ls='--')
plt.plot(yolovx_flop, yolovx_mAP, label='YOLOVX',
         c='pink',
         lw=lw,
         marker='s',
         markersize=ms,
         ls='--')
plt.plot(faster_flop, faster_mAP, label='FASTER-RCNN',
         c='purple',
         lw=lw,
         marker='D',
         markersize=ms,
         ls='--')
plt.plot(rddyolo_flop, rddyolo_mAP, label='RDD-YOLO',
         c='slateblue',
         lw=lw,
         marker='*',
         markersize=10,
         ls='--')
plt.plot(my_flop, my_mAP, label='GPA-DETR',
         c='limegreen',
         lw=lw,
         marker='^',
         markersize=ms,
         ls='--')
plt.legend(loc='lower right', prop=fontcn)
plt.ylabel('$\mathrm{mAP}$ (%)', fontsize=label_size)
plt.xlabel('$\mathrm{FLOPs}$ (G)', fontsize=label_size)
# 设置坐标轴间隔
x_major_locator = MultipleLocator(20)
y_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlim([-3, 270])
plt.ylim([70, 85])
plt.text(yolov8_flop[0] - 2.0, yolov8_mAP[0] + 0.3, yolov8_text[0], color="coral", fontsize=text_size)
plt.text(yolov8_flop[1] - 2.0, yolov8_mAP[1] + 0.3, yolov8_text[1], color="coral", fontsize=text_size)
plt.text(yolov8_flop[2] - 2.0, yolov8_mAP[2] + 0.3, yolov8_text[2], color="coral", fontsize=text_size)
plt.text(yolov8_flop[3] - 2.0, yolov8_mAP[3] + 0.3, yolov8_text[3], color="coral", fontsize=text_size)
plt.text(rtdetr_flop[0] - 5.0, rtdetr_mAP[0] + 0.3, rtdetr_text[0], color="cadetblue", fontsize=text_size)
plt.text(rtdetr_flop[1] - 5.0, rtdetr_mAP[1] + 0.3, rtdetr_text[1], color="cadetblue", fontsize=text_size)
plt.text(rtdetr_flop[2] - 5.0, rtdetr_mAP[2] + 0.3, rtdetr_text[2], color="cadetblue", fontsize=text_size)
plt.text(rtdetr_flop[3] - 6.0, rtdetr_mAP[3] + 0.3, rtdetr_text[3], color="cadetblue", fontsize=text_size)
plt.text(CANet_flop[0] - 9.0, CANet_mAP[0] + 0.3, CANet_text[0], color="dimgray", fontsize=text_size)
plt.text(yolov5_flop[0] - 2.0, yolov5_mAP[0] + 0.3, yolov5_text[0], color="orchid", fontsize=text_size)
plt.text(yolov5_flop[1] - 2.0, yolov5_mAP[1] + 0.3, yolov5_text[1], color="orchid", fontsize=text_size)
plt.text(yolov5_flop[2] - 2.0, yolov5_mAP[2] + 0.3, yolov5_text[2], color="orchid", fontsize=text_size)
plt.text(yolov5_flop[3] - 2.0, yolov5_mAP[3] + 0.3, yolov5_text[3], color="orchid", fontsize=text_size)
plt.text(yolovx_flop[0] - 2.0, yolovx_mAP[0] + 0.3, yolovx_text[0], color="orchid", fontsize=text_size)
plt.text(yolovx_flop[1] - 2.0, yolovx_mAP[1] + 0.3, yolovx_text[1], color="orchid", fontsize=text_size)
plt.text(yolovx_flop[2] - 2.0, yolovx_mAP[2] + 0.3, yolovx_text[2], color="orchid", fontsize=text_size)
plt.text(yolovx_flop[3] - 2.0, yolovx_mAP[3] + 0.3, yolovx_text[3], color="orchid", fontsize=text_size)
plt.text(faster_flop[0] - 10.0, faster_mAP[0] + 0.3, faster_text[0], color="purple", fontsize=text_size)
plt.text(faster_flop[1] - 5.0, faster_mAP[1] + 0.3, faster_text[1], color="purple", fontsize=text_size)
plt.text(rddyolo_flop[0] - 12.0, rddyolo_mAP[0] + 0.3, rddyolo_text[0], color="slateblue", fontsize=text_size)
plt.text(my_flop[0] - 5.0, my_mAP[0] + 0.3, my_text[0], color="limegreen", fontsize=text_size)
plt.text(my_flop[1] - 5.0, my_mAP[1] + 0.3, my_text[1], color="limegreen", fontsize=text_size)
plt.text(my_flop[2] - 5.0, my_mAP[2] + 0.3, my_text[2], color="limegreen", fontsize=text_size)
plt.text(my_flop[3] - 5.0, my_mAP[3] + 0.3, my_text[3], color="limegreen", fontsize=text_size)
plt.grid(linestyle='--')
plt.show()

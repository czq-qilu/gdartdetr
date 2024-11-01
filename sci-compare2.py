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
yolov5_mAP = [74.9, 75.7, 76.5, 77.2, 77.9]
yolov5_flop = [7.2, 24.0, 64.4, 135.3, 246.9]

yolov8_mAP = [77.7, 78.6, 79.2, 79.7, 80.2]
yolov8_flop = [8.9, 28.8, 79.3, 165.7,  258.5]

DETR_mAP = [76.4, 77.5, 78.3, 78.9]
DETR_flop = [86.0, 117, 165, 237]

DeformableDETR_mAP = [77.4, 78.2, 79.1, 80.2]
DeformableDETR_flop = [78.0, 109, 152, 205]

RT_DETR_mAP = [79.7, 80.5, 81.2, 81.9]
RT_DETR_flop = [58.3, 88.6, 134.8, 257.7]

DETR_BIFPN_mAP = [78.9, 79.6, 80.5, 81.8]
DETR_BIFPN_flop = [57.5, 87.4, 132.8, 255.3]

DETR_AFPN_mAP = [80.3, 81.0, 81.8, 82.7]
DETR_AFPN_flop = [63.4, 95.4, 117.7, 195.2]

my_mAP = [82.5, 83.4, 84.2, 85.5]
my_flop = [50.9, 81.4, 103.7, 180.9]

# 参数设置
lw = 1
ms = 6
yolov5_text = ['N', 'S', 'M', 'L', 'X']
yolov8_text = ['N', 'S', 'M', 'L', 'X']
DETR_text = ['R18', 'R34', 'R50', 'R101']
DeformableDETR_text = ['R18', 'R34', 'R50', 'R101']
DAB_DETR_text = ['R18', 'R34', 'R50', 'R101']
RT_DETR_text = ['R18', 'R34', 'R50', 'R101']
DETR_BIFPN_text = ['R18', 'R34', 'R50', 'R101']
DETR_AFPN_text = ['R18', 'R34', 'R50', 'R101']
my_text = ['R18', 'R34', 'R50', 'R101']

# 绘制 mAP-Param
plt.figure(figsize=(6.4, 4.8))
plt.plot(yolov5_flop, yolov5_mAP, label='YOLOV5',
         c='coral',
         lw=lw,
         marker='o',
         markersize=ms,
         ls='--')
plt.plot(yolov8_flop, yolov8_mAP, label='YOLOV8',
         c='cadetblue',
         lw=lw,
         marker='v',
         markersize=ms,
         ls='--')
plt.plot(DETR_flop, DETR_mAP, label='DETR',
         c='dimgray',
         lw=lw,
         marker='v',
         markersize=ms,
         ls='--')
plt.plot(DeformableDETR_flop, DeformableDETR_mAP, label='DeformableDETR',
         c='orchid',
         lw=lw,
         marker='h',
         markersize=ms,
         ls='--')
plt.plot(RT_DETR_flop, RT_DETR_mAP, label='RT-DETR',
         c='pink',
         lw=lw,
         marker='s',
         markersize=ms,
         ls='--')
plt.plot(DETR_BIFPN_flop, DETR_BIFPN_mAP, label='DETR-BiFPN',
         c='purple',
         lw=lw,
         marker='D',
         markersize=ms,
         ls='--')
plt.plot(DETR_AFPN_flop, DETR_AFPN_mAP, label='DETR-AFPN',
         c='slateblue',
         lw=lw,
         marker='*',
         markersize=ms,
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
plt.ylim([72, 88])
plt.text(yolov5_flop[0] - 2.0, yolov5_mAP[0] + 0.3, yolov5_text[0], color="coral", fontsize=text_size)
plt.text(yolov5_flop[1] - 2.0, yolov5_mAP[1] + 0.3, yolov5_text[1], color="coral", fontsize=text_size)
plt.text(yolov5_flop[2] - 2.0, yolov5_mAP[2] + 0.3, yolov5_text[2], color="coral", fontsize=text_size)
plt.text(yolov5_flop[3] - 2.0, yolov5_mAP[3] + 0.3, yolov5_text[3], color="coral", fontsize=text_size)
plt.text(yolov5_flop[4] - 2.0, yolov5_mAP[4] + 0.3, yolov5_text[4], color="coral", fontsize=text_size)
plt.text(yolov8_flop[0] - 5.0, yolov8_mAP[0] + 0.3, yolov8_text[0], color="cadetblue", fontsize=text_size)
plt.text(yolov8_flop[1] - 5.0, yolov8_mAP[1] + 0.3, yolov8_text[1], color="cadetblue", fontsize=text_size)
plt.text(yolov8_flop[2] - 5.0, yolov8_mAP[2] + 0.3, yolov8_text[2], color="cadetblue", fontsize=text_size)
plt.text(yolov8_flop[3] - 6.0, yolov8_mAP[3] + 0.3, yolov8_text[3], color="cadetblue", fontsize=text_size)
plt.text(yolov8_flop[4] - 6.0, yolov8_mAP[4] + 0.3, yolov8_text[4], color="cadetblue", fontsize=text_size)
plt.text(DETR_flop[0] - 2.0, DETR_mAP[0] + 0.3, DETR_text[0], color="dimgray", fontsize=text_size)
plt.text(DETR_flop[1] - 2.0, DETR_mAP[1] + 0.3, DETR_text[1], color="dimgray", fontsize=text_size)
plt.text(DETR_flop[2] - 2.0, DETR_mAP[2] + 0.3, DETR_text[2], color="dimgray", fontsize=text_size)
plt.text(DETR_flop[3] - 2.0, DETR_mAP[3] + 0.3, DETR_text[3], color="dimgray", fontsize=text_size)
plt.text(DeformableDETR_flop[0] - 2.0, DeformableDETR_mAP[0] + 0.3, DeformableDETR_text[0], color="orchid", fontsize=text_size)
plt.text(DeformableDETR_flop[1] - 2.0, DeformableDETR_mAP[1] + 0.3, DeformableDETR_text[1], color="orchid", fontsize=text_size)
plt.text(DeformableDETR_flop[2] - 2.0, DeformableDETR_mAP[2] + 0.3, DeformableDETR_text[2], color="orchid", fontsize=text_size)
plt.text(DeformableDETR_flop[3] - 2.0, DeformableDETR_mAP[3] + 0.3, DeformableDETR_text[3], color="orchid", fontsize=text_size)
plt.text(RT_DETR_flop[0] - 2.0, RT_DETR_mAP[0] + 0.3, RT_DETR_text[0], color="orchid", fontsize=text_size)
plt.text(RT_DETR_flop[1] - 2.0, RT_DETR_mAP[1] + 0.3, RT_DETR_text[1], color="orchid", fontsize=text_size)
plt.text(RT_DETR_flop[2] - 2.0, RT_DETR_mAP[2] + 0.3, RT_DETR_text[2], color="orchid", fontsize=text_size)
plt.text(RT_DETR_flop[3] - 2.0, RT_DETR_mAP[3] + 0.3, RT_DETR_text[3], color="orchid", fontsize=text_size)
plt.text(DETR_BIFPN_flop[0] - 5.0, DETR_BIFPN_mAP[0] + 0.3, DETR_BIFPN_text[0], color="purple", fontsize=text_size)
plt.text(DETR_BIFPN_flop[1] - 5.0, DETR_BIFPN_mAP[1] + 0.3, DETR_BIFPN_text[1], color="purple", fontsize=text_size)
plt.text(DETR_AFPN_flop[0] - 5.0, DETR_AFPN_mAP[0] + 0.3, DETR_AFPN_text[0], color="slateblue", fontsize=text_size)
plt.text(DETR_AFPN_flop[1] - 5.0, DETR_AFPN_mAP[1] + 0.3, DETR_AFPN_text[1], color="slateblue", fontsize=text_size)
plt.text(my_flop[0] - 5.0, my_mAP[0] + 0.3, my_text[0], color="limegreen", fontsize=text_size)
plt.text(my_flop[1] - 5.0, my_mAP[1] + 0.3, my_text[1], color="limegreen", fontsize=text_size)
plt.text(my_flop[2] - 5.0, my_mAP[2] + 0.3, my_text[2], color="limegreen", fontsize=text_size)
plt.text(my_flop[3] - 5.0, my_mAP[3] + 0.3, my_text[3], color="limegreen", fontsize=text_size)
plt.grid(linestyle='--')
plt.show()

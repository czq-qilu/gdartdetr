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
yolov5_mAP = [43.2, 45.4, 48.2, 49.5, 50.2]
yolov5_flop = [7.2, 24.0, 64.4, 135.3, 246.9]

yolov8_mAP = [46.7, 48.9, 50.8, 51.7, 52.5]
yolov8_flop = [8.9, 28.8, 79.3, 165.7,  258.5]

yolov8_bifpn_mAP = [45.9, 48.2, 49.2, 51.2, 51.9]
yolov8_bifpn_flop = [7.9, 26.5, 72.3, 135.7,  208.4]

yolov8_afpn_mAP = [46.7, 48.9, 49.8, 51.7, 52.5]
yolov8_afpn_flop = [5.4, 18.2, 49.5, 103.7,  160.6]


RT_DETR_mAP = [55.0, 56.7]
RT_DETR_flop = [110, 234]

DETR_BIFPN_mAP = [53.0, 54.8]
DETR_BIFPN_flop = [108.2, 223.5]

DETR_AFPN_mAP = [55.6, 57.3]
DETR_AFPN_flop = [91.4, 192.2]

my_mAP = [56.5, 58.7]
my_flop = [79.0, 141.4]

# 参数设置
lw = 1
ms = 6
yolov5_text = ['N', 'S', 'M', 'L', 'X']
yolov8_text = ['N', 'S', 'M', 'L', 'X']
yolov8_bifpn_text = ['N', 'S', 'M', 'L', 'X']
yolov8_afpn_text = ['N', 'S', 'M', 'L', 'X']
RT_DETR_text = ['L', 'X']
DETR_BIFPN_text = ['L', 'X']
DETR_AFPN_text = ['L', 'X']
my_text = ['L', 'X']

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
plt.plot(yolov8_bifpn_flop, yolov8_bifpn_mAP, label='YOLOV8-BiFPN',
         c='dimgray',
         lw=lw,
         marker='v',
         markersize=ms,
         ls='--')
plt.plot(yolov8_afpn_flop, yolov8_afpn_mAP, label='YOLOV8-AFPN',
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
plt.ylim([40, 60])
plt.text(yolov5_flop[0] - 2.0, yolov5_mAP[0] + 0.3, yolov5_text[0], color="coral", fontsize=text_size)
plt.text(yolov5_flop[1] - 2.0, yolov5_mAP[1] + 0.3, yolov5_text[1], color="coral", fontsize=text_size)
plt.text(yolov5_flop[2] - 2.0, yolov5_mAP[2] + 0.3, yolov5_text[2], color="coral", fontsize=text_size)
plt.text(yolov5_flop[3] - 2.0, yolov5_mAP[3] + 0.3, yolov5_text[3], color="coral", fontsize=text_size)
plt.text(yolov8_flop[0] - 5.0, yolov8_mAP[0] + 0.3, yolov8_text[0], color="cadetblue", fontsize=text_size)
plt.text(yolov8_flop[1] - 5.0, yolov8_mAP[1] + 0.3, yolov8_text[1], color="cadetblue", fontsize=text_size)
plt.text(yolov8_flop[2] - 5.0, yolov8_mAP[2] + 0.3, yolov8_text[2], color="cadetblue", fontsize=text_size)
plt.text(yolov8_flop[3] - 6.0, yolov8_mAP[3] + 0.3, yolov8_text[3], color="cadetblue", fontsize=text_size)
plt.text(yolov8_bifpn_flop[0] - 2.0, yolov8_bifpn_mAP[0] + 0.3, yolov8_bifpn_text[0], color="dimgray", fontsize=text_size)
plt.text(yolov8_bifpn_flop[1] - 2.0, yolov8_bifpn_mAP[1] + 0.3, yolov8_bifpn_text[1], color="dimgray", fontsize=text_size)
plt.text(yolov8_bifpn_flop[2] - 2.0, yolov8_bifpn_mAP[2] + 0.3, yolov8_bifpn_text[2], color="dimgray", fontsize=text_size)
plt.text(yolov8_bifpn_flop[3] - 2.0, yolov8_bifpn_mAP[3] + 0.3, yolov8_bifpn_text[3], color="dimgray", fontsize=text_size)
plt.text(yolov8_afpn_flop[0] - 2.0, yolov8_afpn_mAP[0] + 0.3, yolov8_afpn_text[0], color="orchid", fontsize=text_size)
plt.text(yolov8_afpn_flop[1] - 2.0, yolov8_afpn_mAP[1] + 0.3, yolov8_afpn_text[1], color="orchid", fontsize=text_size)
plt.text(yolov8_afpn_flop[2] - 2.0, yolov8_afpn_mAP[2] + 0.3, yolov8_afpn_text[2], color="orchid", fontsize=text_size)
plt.text(yolov8_afpn_flop[3] - 2.0, yolov8_afpn_mAP[3] + 0.3, yolov8_afpn_text[3], color="orchid", fontsize=text_size)
plt.text(RT_DETR_flop[0] - 2.0, RT_DETR_mAP[0] + 0.3, RT_DETR_text[0], color="orchid", fontsize=text_size)
plt.text(RT_DETR_flop[1] - 2.0, RT_DETR_mAP[1] + 0.3, RT_DETR_text[1], color="orchid", fontsize=text_size)
plt.text(DETR_BIFPN_flop[0] - 5.0, DETR_BIFPN_mAP[0] + 0.3, DETR_BIFPN_text[0], color="purple", fontsize=text_size)
plt.text(DETR_BIFPN_flop[1] - 5.0, DETR_BIFPN_mAP[1] + 0.3, DETR_BIFPN_text[1], color="purple", fontsize=text_size)
plt.text(DETR_AFPN_flop[0] - 5.0, DETR_AFPN_mAP[0] + 0.3, DETR_AFPN_text[0], color="slateblue", fontsize=text_size)
plt.text(DETR_AFPN_flop[1] - 5.0, DETR_AFPN_mAP[1] + 0.3, DETR_AFPN_text[1], color="slateblue", fontsize=text_size)
plt.text(my_flop[0] - 5.0, my_mAP[0] + 0.3, my_text[0], color="limegreen", fontsize=text_size)
plt.text(my_flop[1] - 5.0, my_mAP[1] + 0.3, my_text[1], color="limegreen", fontsize=text_size)
plt.grid(linestyle='--')
plt.show()

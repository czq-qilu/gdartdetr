
# matplotlib inline

# 导入相关库
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# colors = ["#4E79A7",  "#A0CBE8",  "#F28E2B",  "#FFBE7D",  "#59A14F",  "#8CD17D",  "#B6992D",
# "#F1CE63",  "#499894",  "#86BCB6",  "#E15759",  "#E19D9A"]

# 自定义每根柱子的颜色
colors = ["#4E79A7",  "#A0CBE8",  "#F28E2B",  "#FFBE7D" ]
labels = ['YOLOV5', 'YOLOV8', 'DeformableDETR', 'RT-DETR', 'DETR-BiFPN', 'DETR-AFPN', 'GDA_DETR']

a = [0.7, 0.6, 0.4, 0.4, 0.4, 0.5, 0.4]
b = [12.0, 11.4, 98.2, 22.3, 22.7, 26.2, 38.9]
c = [1.5, 1.4, 0.4, 0.3, 0.4, 0.5, 0.3]
d = [70.6, 74.4, 9.8, 42.9, 42.2, 36.7, 25.0]

x = np.arange(len(labels))  # 标签位置
width = 0.2  # 柱状图的宽度，可以根据自己的需求和审美来改

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5*width, a, width, label='preprocess', color="#4E79A7")
rects2 = ax.bar(x - 0.5*width, b, width, label='inference', color="#A0CBE8")
rects3 = ax.bar(x + 0.5*width, c, width, label='postprocess', color="#F28E2B")
rects4 = ax.bar(x + 1.5*width, d, width, label='FPS', color="#FFBE7D" )



# 为y轴、标题和x轴等添加一些文本。
ax.set_ylabel('', fontsize=15)
ax.set_xlabel('', fontsize=15)
ax.set_title('')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


def autolabel(rects):
    """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)


fig.tight_layout()

plt.show()
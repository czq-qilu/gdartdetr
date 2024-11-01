import matplotlib.pyplot as plt
import pandas as pd

# 绘制PR
def plot_PR():
    pr_csv_dict = {
        'YOLOV8': r'F:/ultralytics-main/runs/train/yolov8/PR_curve.csv',
        'RT-DETR': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-ResNet/PR_curve.csv',
        'DETR-BiFPN': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-l-HSFPN/PR_curve.csv',
        'DETR-AFPN': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-l-ResNet-cafpn-icmb/PR_curve.csv',
        'GDA-DETR': r'F:/ultralytics/runs/RT-DETR-train/EIOU(best)/PR_curve.csv',
    }

    # 绘制pr
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname in pr_csv_dict:
        res_path = pr_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[6]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='2')

    # 添加x轴和y轴标签
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()  # 显示网格线
    # 显示图像
    fig.savefig("pr.png", dpi=250)
    plt.show()
if __name__ == '__main__':
    plot_PR()   # 绘制PR
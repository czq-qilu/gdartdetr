import matplotlib.pyplot as plt
import pandas as pd

# 绘制PR
def plot_PR():
    pr_csv_dict = {
        'YOLOv5': r'F:/ultralytics/runs/lens/yolov5/PR_curve.csv',
        'YOLOv8': r'F:/ultralytics/runs/lens/yolov8/PR_curve.csv',
        # 'YOLOV8-BiFPN': r'F:/ultralytics/runs/train/yolov8-afpn/PR_curve.csv',
        # 'YOLOV8-AFPN': r'F:/ultralytics/runs/train/yolov8-gdafpn/PR_curve.csv',
        'DeformableDETR': r'F:/ultralytics/runs/lens/GDA-DETR(carafe)/PR_curve.csv',
        'RT-DETR': r'F:/ultralytics/runs/lens/RT-DETR/PR_curve.csv',
        'DETR-BiFPN': r'F:/ultralytics/runs/lens/DETR-BIFPN/PR_curve.csv',
        'DETR-AFPN': r'F:/ultralytics/runs/lens/DETR-AFPN/PR_curve.csv',
        'GDA-DETR': r'F:/ultralytics/runs/lens/GDA-DETR(EIOU+FOCAL best)/PR_curve.csv',

    }

    # 绘制pr
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname in pr_csv_dict:
        res_path = pr_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[5]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='1')

    # 添加x轴和y轴标签
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.grid()  # 显示网格线
    # 显示图像
    fig.savefig("pr.png", dpi=250)
    plt.show()



def plot_F1():
    f1_csv_dict = {
        'YOLOv5': r'F:/ultralytics/runs/lens/yolov5/F1_curve.csv',
        'YOLOv8': r'F:/ultralytics/runs/lens/yolov8/F1_curve.csv',
        # 'YOLOV8-BiFPN': r'F:/ultralytics/runs/train/yolov8-afpn/PR_curve.csv',
        # 'YOLOV8-AFPN': r'F:/ultralytics/runs/train/yolov8-gdafpn/PR_curve.csv',
        'DeformableDETR': r'F:/ultralytics/runs/lens/GDA-DETR(carafe)/F1_curve.csv',
        'RT-DETR': r'F:/ultralytics/runs/lens/RT-DETR/F1_curve.csv',
        'DETR-BiFPN': r'F:/ultralytics/runs/lens/GDA-DETR(反卷积)/F1_curve.csv',
        'DETR-AFPN': r'F:/ultralytics/runs/lens/DETR-AFPN/F1_curve.csv',
        'GDA-DETR': r'F:/ultralytics/runs/lens/GDA-DETR(256+lp)best/F1_curve.csv',
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname in f1_csv_dict:
        res_path = f1_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[5]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='1')

    # 添加x轴和y轴标签
    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()  # 显示网格线
    # 显示图像
    fig.savefig("F1.png", dpi=250)
    plt.show()


if __name__ == '__main__':
    plot_PR()  # 绘制PR
    plot_F1()  # 绘制F1

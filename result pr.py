import matplotlib.pyplot as plt
import pandas as pd

# 绘制PR
def plot_PR():
    pr_csv_dict = {
        'YOLOV8m': r'F:/ultralytics-main/runs/train/yolov8/PR_curve.csv',
        'RT-DETR': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-ResNet/PR_curve.csv',
        'DETR-HSFPN': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-l-HSFPN/PR_curve.csv',
        'DETR-AFPN': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-l-ResNet-cafpn-icmb/PR_curve.csv',
        'GDA-DETR': r'F:/ultralytics/runs/RT-DETR-train/EIOU(best)/PR_curve.csv',
    }

    # 绘制pr
    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname in pr_csv_dict:
        res_path = pr_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[3]).values.ravel()
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

# 绘制F1
def plot_F1():
    f1_csv_dict = {
        'YOLOV8m': r'F:/ultralytics-main/runs/train/yolov8/F1_curve.csv',
        'RT-DETR': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-ResNet/F1_curve.csv',
        'DETR-HSFPN': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-l-HSFPN/F1_curve.csv',
        'DETR-AFPN': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-l-ResNet-cafpn-icmb/F1_curve.csv',
        'GDA-DETR': r'F:/ultralytics/runs/RT-DETR-train/EIOU(best)/F1_curve.csv',

        # 'nah-datr': r'F:/ultralytics-main/runs/train/rtdetr-iafpn-improved/F1_curve.csv',
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)

    for modelname in f1_csv_dict:
        res_path = f1_csv_dict[modelname]
        x = pd.read_csv(res_path, usecols=[1]).values.ravel()
        data = pd.read_csv(res_path, usecols=[2]).values.ravel()
        ax.plot(x, data, label=modelname, linewidth='1')

    # 添加x轴和y轴标签
    ax.set_xlabel('Confidence')
    ax.set_ylabel('F1')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid()  # 显示网格线
    # 显示图像
    fig.savefig("F1.png", dpi=250)
    plt.show()

if __name__ == '__main__':
    plot_PR()   # 绘制PR
    plot_F1()   # 绘制F1



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # 列出待获取数据内容的文件位置
    # v5、v8都是csv格式的，v7是txt格式的
    result_dict = {
        'YOLOV8m': r'F:/ultralytics-main/runs/train/yolov8/results.csv',
        'RT-DETR': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-ResNet/results.csv',
        'DETR-HSFPN': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-l-HSFPN/results.csv',
        'DETR-AFPN': r'F:/ultralytics/runs/RT-DETR-train/rtdetr-l-ResNet-cafpn-icmb/results.csv',
        'GDA-DETR': r'F:/ultralytics/runs/RT-DETR-train/EIOU(best)/results.csv',
        # 'nah-datr': r'F:/ultralytics-main/runs/train/rtdetr-iafpn-improved/results.csv',
    }

    # 绘制map50
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[6]).values.ravel()    # 6是指map50的下标（每行从0开始向右数）
        else:   # 文件后缀是txt
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[10]))   # 10是指map50的下标（每行从0开始向右数）
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='1')   # 线条粗细设为1

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5')
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("mAP50.png", dpi=600)   # dpi可设为300/600/900，表示存为更高清的矢量图
    plt.show()


    # 绘制map50-95
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            data = pd.read_csv(res_path, usecols=[7]).values.ravel()    # 7是指map50-95的下标（每行从0开始向右数）
        else:
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[11]))   # 11是指map50-95的下标（每行从0开始向右数）
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='1')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('mAP@0.5:0.95')
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("mAP50-95.png", dpi=600)
    plt.show()

    # 绘制训练的总loss
    for modelname in result_dict:
        res_path = result_dict[modelname]
        ext = res_path.split('.')[-1]
        if ext == 'csv':
            box_loss = pd.read_csv(res_path, usecols=[1]).values.ravel()
            obj_loss = pd.read_csv(res_path, usecols=[2]).values.ravel()
            cls_loss = pd.read_csv(res_path, usecols=[3]).values.ravel()
            data = np.round(box_loss + obj_loss + cls_loss, 5)    # 3个loss相加并且保留小数点后5位（与v7一致）

        else:
            with open(res_path, 'r') as f:
                datalist = f.readlines()
                data = []
                for d in datalist:
                    data.append(float(d.strip().split()[5]))
                data = np.array(data)
        x = range(len(data))
        plt.plot(x, data, label=modelname, linewidth='1')

    # 添加x轴和y轴标签
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    # 显示图像
    plt.savefig("loss.png", dpi=600)
    plt.show()

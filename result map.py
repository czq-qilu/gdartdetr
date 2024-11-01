import pandas as pd
import matplotlib.pyplot as plt

# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('\s+', '_', regex=True)

#nonoresult.csv表示原始的结果图,csv文件在runs/train/exp中
RT_DETR_results = pd.read_csv("F:/ultralytics-main/runs/detect/rtdetr/results.csv")
RT_DETR_MobileViT_results = pd.read_csv("F:/ultralytics-main/runs/detect/rtdetr-MboileVit/results.csv")
#yesyesresult.csv表示提高后的结果图，csv文件在runs/train/exp中
RT_DETR_MobileViT_Carafe_results = pd.read_csv("F:/ultralytics-main/runs/detect/rtdetr-mobilevit-carafe/results.csv")
yolov8_results = pd.read_csv("F:/ultralytics-main/runs/detect/yolov8/results.csv")
yolov5_results = pd.read_csv("F:/yolov5-master/runs/train/new-v5-afpn/results.csv")
# Clean column names
clean_column_names(RT_DETR_results)
clean_column_names(RT_DETR_MobileViT_results)
clean_column_names(RT_DETR_MobileViT_Carafe_results)
clean_column_names(yolov8_results)
clean_column_names(yolov5_results)

# Plot mAP@0.5 curves
plt.figure()
#label属性为曲线名称，自己可以定义
plt.plot(RT_DETR_results['metrics/mAP50(B)'], label="RT-DETR")
plt.plot(RT_DETR_MobileViT_results['metrics/mAP50(B)'], label="RT-DETR-MobileViT")
plt.plot(RT_DETR_MobileViT_Carafe_results['metrics/mAP50(B)'], label="RT-DETR-MobileViT-Carafe")
plt.plot(yolov8_results['metrics/mAP50(B)'], label="yolov8")
plt.plot(yolov5_results['metrics/mAP_0.5'], label="yolov5-AFPN")

plt.xlabel("Epoch")
plt.ylabel("mAP@0.5")
plt.legend()
plt.title("mAP@0.5 Comparison")
plt.savefig("mAP_0.5_comparison.png")



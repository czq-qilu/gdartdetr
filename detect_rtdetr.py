from ultralytics import RTDETR
# 14_3072_0_4352_1280.jpg  15_2048_1024_3328_2304.jpg  jz31_3072_2048_4352_3328.jpg
# image (4002).jpg   image (797).jpg     image (261).jpg   image (3417).jpg
# 3_2048_1024_3328_2304
if __name__ == '__main__':
    # 加载模型
    model = RTDETR(r'F:/ultralytics/runs/train/RT-DETR(300)/weights/best.pt')  # YOLOv8n模型
    model.predict(
        source=r'F:/ultralytics/datasets/lens/valid/images/image (4002).jpg',
        save=True,  # 保存预测结果
        imgsz=640,  # 输入图像的大小，可以是整数或w，h
        conf=0.25,  # 用于检测的目标置信度阈值（默认为0.25，用于预测，0.001用于验证）
        iou=0.45,  # 非极大值抑制 (NMS) 的交并比 (IoU) 阈值
        show=False,  # 如果可能的话，显示结果
        project='runs/predict',  # 项目名称（可选）
        name='exp',  # 实验名称，结果保存在'project/name'目录下（可选）
        save_txt=False,  # 保存结果为 .txt 文件
        save_conf=True,  # 保存结果和置信度分数
        save_crop=False,  # 保存裁剪后的图像和结果
        show_labels=True,  # 在图中显示目标标签
        show_conf=True,  # 在图中显示目标置信度分数
        vid_stride=1,  # 视频帧率步长
        line_width=1,  # 边界框线条粗细（像素）
        visualize=False,  # 可视化模型特征
        augment=False,  # 对预测源应用图像增强
        agnostic_nms=False,  # 类别无关的NMS
        retina_masks=False,  # 使用高分辨率的分割掩码
        boxes=True,  # 在分割预测中显示边界框
    )


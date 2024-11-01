# -*- coding:utf-8 -*-
# @Time  : 2023/12/2319:57
# @Description：
# @Department ：电网自动化综合产品部
# @Author: Jiahao Li
# @File  : pt2onnx.py

import onnxruntime as rt
import os
import torch
# from networks.net import TransCDNet
# from networks import configs as cfg
from ultralytics import YOLO


def pt2onnx(pt_path, save_onnx_path):
    # model = torch.load(pt_path, map_location='cpu')
    # print(model)
    # model['model'].eval()
    # dummpy_input = torch.randn(1, 3, 640, 640)
    # torch.onnx.export(model, dummpy_input, save_onnx_path + '\\' + 'yolov8n.onnx', export_params=True,
    #                   input_names=['input'],
    #                   output_names=['output'])
    # return
    from ultralytics import YOLO
    model = YOLO(pt_path)
    success = model.export(format='onnx', simplify=True)  # export the model to onnx format
    assert success
    print("转换成功")


# transCD的模型导出
# def pth_to_onnx(input, pth_path, save_onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
#     ncfg = cfg.CONFIGS['SViT_E4_D4_32']
#     img_size = 512
#     model = TransCDNet(ncfg, img_size, False).to(device=device)
#     model.load_state_dict(torch.load(pth_path, map_location=device)['model_state_dict'])  # 初始化权重
#     print(model)
#     model.eval()
#     # model.to(device)
#
#     torch.onnx.export(model, (input,input), save_onnx_path + '\\' + 'SViT_E4_D4_32.onnx', verbose=True, input_names=input_names,
#                       output_names=output_names)  # 指定模型的输入，以及onnx的输出路径
#     print("Exporting .pth model to onnx model has been successful!")


if __name__ == '__main__':
    pt_path = r'F:\ultralytics\runs\detect\yolov8m\weights\best.pt'
    pth_path = r'E:\PythonProject\ultralytics\txpb_bj\SViT_E4_D4_32.pth'
    save_onnx_path = r'F:\ultralytics\runs\detect\yolov8m\weights'
    pt2onnx(pt_path, save_onnx_path)
    # TransCD的InputSize
    # input = torch.randn(1, 3, 512, 512)
    # pth_to_onnx(input, pth_path, save_onnx_path)

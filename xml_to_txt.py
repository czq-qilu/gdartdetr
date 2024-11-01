import os
# 使用xml.etree.ElementTree解析voc
import xml.etree.ElementTree as ET

def xml_to_txt(xml_path, txt_path, class_name):

    # 获取xml文件的名称，去除后缀名，并存放在列表中
    xml_name = os.listdir(xml_path)
    for i in range(len(xml_name)):
        xml_path_name = os.path.join(xml_path, xml_name[i])
        print(xml_path_name)

        # 逐个xml进行解析，计算，并将结果写入txt中
        tree = ET.parse(xml_path_name)
        root = tree.getroot()
        # 获取图像尺寸大小
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in class_name or int(difficult) == 1:
                continue
            # 获取标签id
            cls_id = class_name.index(cls)
            # 获取标注坐标
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            # 坐标转化
            bb = convert((w, h), b)
            with open(os.path.join(txt_path, xml_name[i].split(".")[0]+'.txt'), 'a+') as file:
                file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def convert(size, box):
    # print(box[0])
    # print(box[1])
    # print(box[2])
    # print(box[3])
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

if __name__ == '__main__':

    # xml的路径
    xml_path = r'F:/ultralytics/datasets/add/xml/'
    # txt的路径
    txt_path = r'F:/ultralytics/datasets/add/txt'

    if os.path.exists(txt_path) == 0:
        os.mkdir(txt_path)

    class_name =  ['black spot', 'scratch', 'wool']

    xml_to_txt(xml_path, txt_path, class_name)
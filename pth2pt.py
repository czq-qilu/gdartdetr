import torch
import torchvision
import torchvision.models as models

model = models.resnet18()


model.load_state_dict(torch.load("F:/ultralytics/resnet18.pth"))#保存的训练模型
model.eval()#切换到eval（）
example = torch.rand(1, 3, 320, 480)#生成一个随机输入维度的输入
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("resnet18.pt")

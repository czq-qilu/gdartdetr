import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 支持显示中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 读取菜品销售量数据
filepath = 'table.xlsx'
cor = pd.read_excel(filepath)
# 计算相关系数矩阵，包含了任意两个菜品间的相关系数
print('企业创新性评估：\n', cor.corr())

# 绘制相关性热力图
plt.subplots(figsize=(10, 10))  # 设置画面大小
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
plt.xlabel('x',fontsize=20)
plt.ylabel('y',fontsize=20)
sns.heatmap(cor.corr(), annot=True, vmax=1, square=True, cmap="Blues", annot_kws={"fontsize":20})
plt.title('pearson相关性热力图')
plt.show()


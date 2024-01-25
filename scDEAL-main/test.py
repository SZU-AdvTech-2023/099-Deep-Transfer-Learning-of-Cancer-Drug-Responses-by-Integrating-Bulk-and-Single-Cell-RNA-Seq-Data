# import matplotlib.pyplot as plt
#
# def draw_scatter_plot(x1, x2, y1, y2):
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
#     # 为了坐标轴负号正常显示。matplotlib默认不支持中文，设置中文字体后，负号会显示异常。需要手动将坐标轴负号设为False才能正常显示负号。
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.plot(x1, y1, 'o',color='blue', label='蛮力法')
#     plt.plot(x2, y2, 'v',color='red', label='分治法')
#     plt.xlabel('矩阵的阶')
#     plt.ylabel('时间(ms)')
#     plt.title('阶-时间散点图')
#     plt.legend()
#     plt.show()
#
# x1= [100, 1000, 2000, 3000, 4000, 5000]
# x2= [100, 1000,  2000, 3000, 4000, 5000, 10000, 25000, 40000, 50000,70000,80000, 100000]
# y1 = [14, 6738, 68027, 263226, 631707, 1273611]
# y2 = [21, 28, 35, 19, 25, 28, 47, 45, 98, 141, 129, 125, 149]
#
#
# # 绘制散点图
# draw_scatter_plot(x1,x2, y1, y2)
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
# 假设img1和img2是你的两张图片
# 请替换为你的实际图片数据或文件路径

# 创建一个 1 行 2 列的 subplot 网格
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
img1 = Image.open("fig/img1.png")
img3 = Image.open("fig/img4.png")
img2 = Image.open("fig/img2.png")
img4 = Image.open("fig/img3.png")
# 在第一个 subplot 中显示第一张图片
axs[0,0].imshow(img3)
axs[0,0].axis('off') # 关闭坐标轴

# 在第二个 subplot 中显示第二张图片
axs[0,1].imshow(img4)
axs[0,1].axis('off') # 关闭坐标轴

axs[1,0].imshow(img1)
axs[1,0].axis('off') # 关闭坐标轴

axs[1,1].imshow(img2)
axs[1,1].axis('off') # 关闭坐标轴
# 调整 subplot 之间的间距
plt.subplots_adjust(wspace=0, hspace=0)

# 显示图形
plt.show()

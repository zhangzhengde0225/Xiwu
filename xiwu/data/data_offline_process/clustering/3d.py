import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 词频数据
word_freq = {'Python': 100, 'programming': 50, 'language': 30, 'data science': 20, 'development': 10}

# 生成数据
labels = word_freq.keys()
sizes = word_freq.values()
colors = plt.cm.tab20c.colors  # 使用matplotlib内置的颜色映射

# 绘制饼图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制一个圆来模拟3D效果
for i in range(len(sizes)):
    ax.pie(sizes, labels=labels, colors=colors, startangle=140, wedgeprops=dict(width=0.5, edgecolor='w'))

    # "提升"饼块来创建3D效果
    ax.text(0, 0, i*0.1, str(i), color="k", va="center", ha="center")

ax.set_zlim(0, 0.5)  # 设置z轴高度

plt.title("3D Pie Chart with 'Lifted' Slices")
plt.savefig('3d_pie_chart.png')
plt.show()

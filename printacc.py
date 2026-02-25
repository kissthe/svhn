import json
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持 (如果需要显示中文标签)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans'] # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像时负号 '-' 显示为方块的问题

# 1. 加载 JSON 数据
file_path = 'all_multi_dataset_weights_config_parallel.json' # 请确保文件路径正确
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")
    exit()
except json.JSONDecodeError:
    print(f"错误：文件 {file_path} 不是有效的JSON格式")
    exit()

# 2. 提取性能指标
val_accuracies = []
val_losses = []

for item in data:
    performance = item.get('performance', {})
    val_accuracies.append(performance.get('val_accuracy'))
    val_losses.append(performance.get('val_loss'))

# 过滤掉可能的 None 值 (虽然根据结构应该没有)
val_accuracies = [acc for acc in val_accuracies if acc is not None]
val_losses = [loss for loss in val_losses if loss is not None]

print(f"加载了 {len(val_accuracies)} 个模型的验证准确率数据。")
print(f"加载了 {len(val_losses)} 个模型的验证损失数据。")

if not val_accuracies or not val_losses:
    print("错误：没有找到有效的性能数据。")
    exit()

# 3. 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 4. 绘制验证准确率分布
ax1.hist(val_accuracies, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax1.set_xlabel('验证准确率 (Val Accuracy)')
ax1.set_ylabel('频率 (Frequency)')
ax1.set_title('模型验证准确率分布')
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 添加统计数据文本框
stats_text_acc = f'均值: {np.mean(val_accuracies):.2f}\n标准差: {np.std(val_accuracies):.2f}\n范围: [{min(val_accuracies):.2f}, {max(val_accuracies):.2f}]'
ax1.text(0.95, 0.95, stats_text_acc, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

# 5. 绘制验证损失分布
ax2.hist(val_losses, bins=30, color='lightcoral', edgecolor='black', alpha=0.7)
ax2.set_xlabel('验证损失 (Val Loss)')
ax2.set_ylabel('频率 (Frequency)')
ax2.set_title('模型验证损失分布')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 添加统计数据文本框
stats_text_loss = f'均值: {np.mean(val_losses):.3f}\n标准差: {np.std(val_losses):.3f}\n范围: [{min(val_losses):.3f}, {max(val_losses):.3f}]'
ax2.text(0.05, 0.95, stats_text_loss, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

# 6. 调整布局并显示图形
plt.tight_layout()
plt.savefig('model_performance_distribution.png')
plt.show()
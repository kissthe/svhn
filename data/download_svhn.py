import torchvision
import torchvision.transforms as transforms

# 定义转换（这里使用简单的转换，仅用于下载）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("开始下载SVHN训练集...")
train_dataset = torchvision.datasets.SVHN(
    root='./data',
    split='train',
    download=True,
    transform=transform
)
print(f"训练集下载完成，包含 {len(train_dataset)} 个样本")

print("开始下载SVHN测试集...")
test_dataset = torchvision.datasets.SVHN(
    root='./data',
    split='test',
    download=True,
    transform=transform
)
print(f"测试集下载完成，包含 {len(test_dataset)} 个样本")

print("SVHN数据集下载完成！")

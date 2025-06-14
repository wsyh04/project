import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# 创建保存目录
os.makedirs('./results', exist_ok=True)

# 设定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 定义 CNN 模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8 * 8 * 128, 256), nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 训练函数
def train(model, optimizer, criterion, scheduler, trainloader, testloader, num_epochs=20):
    model.to(device)
    train_acc, test_acc = [], []

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        train_acc.append(acc)

        # 测试集上评估
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc.append(100. * correct / total)

        if scheduler is not None:
            scheduler.step()

        print(f"Epoch {epoch+1}: Train Acc = {train_acc[-1]:.2f}%, Test Acc = {test_acc[-1]:.2f}%")

    return train_acc, test_acc

# 启动实验
def run_experiments(trainloader, testloader, num_epochs=20):
    optimizers = {
        'SGD': lambda params: optim.SGD(params, lr=0.1, momentum=0.9),
        'SGD_StepLR': lambda params: (optim.SGD(params, lr=0.1, momentum=0.9), 'StepLR'),
        'SGD_CosineLR': lambda params: (optim.SGD(params, lr=0.1, momentum=0.9), 'CosineAnnealingLR'),
        'Adam': lambda params: optim.Adam(params, lr=0.001),
        'RMSprop': lambda params: optim.RMSprop(params, lr=0.001)
    }

    results = {}

    for name, opt_func in optimizers.items():
        print(f"\n===== 开始训练: {name} =====")
        model = SimpleCNN()

        if isinstance(opt_func(model.parameters()), tuple):
            optimizer, scheduler_type = opt_func(model.parameters())
            if scheduler_type == 'StepLR':
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            elif scheduler_type == 'CosineAnnealingLR':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        else:
            optimizer = opt_func(model.parameters())
            scheduler = None

        criterion = nn.CrossEntropyLoss()
        train_acc, test_acc = train(model, optimizer, criterion, scheduler, trainloader, testloader, num_epochs=num_epochs)
        results[name] = (train_acc, test_acc)

    return results

# 绘图函数
def plot_results(results):
    plt.figure(figsize=(12, 6))
    for name, (train_acc, test_acc) in results.items():
        plt.plot(test_acc, label=name)
    plt.title('Test Accuracy vs Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('./results/CNN_optimizer_comparison.png')
    plt.show()

# 保存结果
def save_results_txt(results):
    with open('./results/CNN_results.txt', 'w') as f:
        for name, (train_acc, test_acc) in results.items():
            f.write(f"{name}\n")
            f.write(f"Final Train Accuracy: {train_acc[-1]:.2f}%\n")
            f.write(f"Final Test Accuracy: {test_acc[-1]:.2f}%\n")
            f.write("-" * 40 + "\n")

# 主函数
if __name__ == "__main__":
    # 数据只加载一次
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # 设定训练轮数
    num_epochs = 100

    results = run_experiments(trainloader, testloader, num_epochs=num_epochs)
    plot_results(results)
    save_results_txt(results)


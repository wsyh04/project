import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import os

# 创建保存目录
os.makedirs('./results', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

#定义ResNet-18模型
class ResNet18Model(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18Model, self).__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train(model, optimizer, criterion, scheduler, trainloader, testloader, num_epochs=20):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()  # AMP梯度缩放器
    train_acc, test_acc = [], []

    for epoch in range(num_epochs):
        model.train()
        correct, total = 0, 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():  # 自动混合精度上下文
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100. * correct / total
        train_acc.append(acc)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
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
        model = ResNet18Model()

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
    plt.savefig('./results/ResNet18_optimizer_comparison.png')
    plt.show()

# 保存结果
def save_results_txt(results):
    with open('./results/ResNet18_results.txt', 'w') as f:
        for name, (train_acc, test_acc) in results.items():
            f.write(f"{name}\n")
            f.write(f"Final Train Accuracy: {train_acc[-1]:.2f}%\n")
            f.write(f"Final Test Accuracy: {test_acc[-1]:.2f}%\n")
            f.write("-" * 40 + "\n")

# 主函数
if __name__ == "__main__":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    num_epochs = 100

    results = run_experiments(trainloader, testloader, num_epochs=num_epochs)
    plot_results(results)
    save_results_txt(results)

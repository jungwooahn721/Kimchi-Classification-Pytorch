import torch

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def accuracy_(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target_index = torch.argmax(target, dim=1)
        assert pred.shape == target_index.shape
        correct = 0
        correct += torch.sum(pred == target_index).item()
    return correct / len(target_index)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


import torch
import torch.nn.functional as F

def eval_from_dataloader(model, device, data_loader, gpus=1, log=None):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(data_loader.dataset)

    data_size = len(data_loader.dataset)/gpus
    info = 'Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, data_size, 
        100. * correct / data_size)
    test_accuracy = correct / data_size
    return test_loss, test_accuracy, info
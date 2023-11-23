from tqdm import tqdm
import torch


def test_fn(config, model, test_loader):
    test_loss = 0
    test_accuracy = 0
    config.ACC_FUNC.reset()
    with torch.no_grad():       
        for images, labels in tqdm(test_loader, total=len(test_loader), desc="Testing "):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            pred = model(images)
            loss = config.LOSS_FUNC(pred, labels.squeeze(-1))
            test_loss += loss.item()
            config.ACC_FUNC.update(pred.cpu(), labels.cpu())
    
    pred = model(images)
    test_accuracy = config.ACC_FUNC.compute().item()
                 
    return test_accuracy, test_loss/config.TEST_SIZE
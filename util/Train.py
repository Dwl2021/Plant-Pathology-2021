from tqdm import tqdm
import torch
def train_fn(config, model, train_loader, epoch):
    tr_loss = 0
    tr_accuracy = 0
    config.ACC_FUNC.reset()
    for images, labels in tqdm(train_loader, total=len(train_loader), desc="Epoch {} Training".format(epoch + 1)):
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        config.OPTIM.zero_grad()
        pred = model(images)

        # calculate the loss
        loss = config.LOSS_FUNC(pred, labels.squeeze(-1))
        model.zero_grad()
        loss.backward()
        tr_loss += loss.item()
        config.ACC_FUNC.update(pred.cpu(), labels.cpu())
        config.OPTIM.step()
        
        # print("One image finished, running loss is" + str(tr_loss/TRAIN_SIZE))
    tr_accuracy = config.ACC_FUNC.compute().item()
    # print("One epoch finished,""Accuracy is",tr_accuracy,"Train loss is",tr_loss/TRAIN_SIZE)
    return tr_accuracy, tr_loss/config.TRAIN_SIZE


def valid_fn(config, model, val_loader, epoch):
    valid_loss = 0
    valid_accuracy = 0
    config.ACC_FUNC.reset()
    with torch.no_grad():       
        for images, labels in tqdm(val_loader, total=len(val_loader), desc="Epoch {} Validing".format(epoch + 1)):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            pred = model(images)
            loss = config.LOSS_FUNC(pred, labels.squeeze(-1))
            valid_loss += loss.item()
            config.ACC_FUNC.update(pred.cpu(), labels.cpu())
    
    pred = model(images)
    valid_accuracy = config.ACC_FUNC.compute().item()
                 
    return valid_accuracy, valid_loss/config.VAL_SIZE
import torch
import torch.utils
import torch.utils.data

from tqdm.auto import tqdm

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    '''
    Trains a model for a single epoch.

    Args:
        model: PyTorch model.
        data_loader: train dataloader.
        loss_fn: PyTorch loss function.
        optimizer: PyTorch optimizer.

    Returns:
        a tuple of (training_loss, training_accuracy)
    '''
    train_loss, train_acc = 0, 0
    model.train()
    for X, y in data_loader:
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss
        
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        predicted_classes = torch.argmax(y_pred, dim=1)
        train_acc += (predicted_classes == y).sum().item()/len(predicted_classes)
    
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module):
    '''
    test a model for a single epoch.

    Args:
        model: PyTorch model.
        data_loader: test dataloader.
        loss_fn: PyTorch loss function.

    Returns:
        a tuple of (testing_loss, testing_accuracy)
    '''
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            y_pred = model(X)

            test_loss += loss_fn(y_pred, y)

            predicted_classes = torch.argmax(y_pred, dim=1)
            test_acc += (predicted_classes == y).sum().item()/len(predicted_classes)

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int):
    '''
    trains and tests model fo a given number of epochs.

    Args:
        model: PyTorch model.
        train_dataloader: Train dataloader.
        test_dataloader: Test dataloader.
        loss_fn: loss function.
        optimizer: optimizer.
        epochs: number of epochs

    Returns:
        A dictionary with keys (train_loss, train_acc, test_loss, test_acc)
        and values (list(train_losses), list(train_accs), list(test_losses), list(test_accs))
    '''
    result = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }

    print("=================================================")

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model,
                                           train_dataloader,
                                           loss_fn,
                                           optimizer)
        
        test_loss, test_acc = test_step(model,
                                        test_dataloader,
                                        loss_fn)
        
        result['train_loss'].append(train_loss.item())
        result['train_acc'].append(train_acc)
        result['test_loss'].append(test_loss.item())
        result['test_acc'].append(test_acc)

        print(f" Epoch: {epoch+1}")
        print(f"Train Loss = {train_loss:.4f} || Train Accuracy = {train_acc*100:.2f}%")
        print(f"Test Loss = {test_loss:.4f} || Test Accuracy = {test_acc*100:.2f}%")
        print("=================================================")
        
    return result
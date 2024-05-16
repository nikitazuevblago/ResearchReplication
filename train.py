import os
import zipfile
import torch
import torchvision
from torchvision import datasets, transforms
import random
from tqdm.auto import tqdm
from pathlib import Path, PosixPath
import shutil
from typing import Tuple, Dict
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
warnings.filterwarnings('ignore')


def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device:torch.device,
               # if grad_clipping
               max_norm = False,
               # if scheduler
               epoch = False,
               scheduler = False) -> Tuple[float,float,float,float,float]:
    # train mode
    model.train()

    # Set up metrics
    train_loss, train_acc, train_precision, train_recall, train_f1 = 0, 0, 0, 0, 0

    # Loop through batches
    for batch_i, (X, y) in enumerate(dataloader):
        if scheduler:
            step = (epoch - 1) * len(dataloader) + batch_i
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        logits = model(X)

        # 2. Compute loss
        loss = loss_fn(logits, y)

        # 3. Zero gradients
        optimizer.zero_grad()

        # 4. Backward propagation
        loss.backward()

        # ***Gradient clipping
        if max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # 5. Update weights and biases
        optimizer.step()

        # **Step the learning rate scheduler
        if scheduler:
            scheduler.step(step)

        # Get the most confident predictions
        predictions = torch.argmax(logits,dim=1).cpu()
        y = y.cpu()

        # Calculate metrics
        train_loss += loss.item()
        train_acc += accuracy_score(predictions, y)
        train_precision += precision_score(predictions, y, average='macro')
        train_recall += recall_score(predictions, y, average='macro')
        train_f1 += f1_score(predictions, y, average='macro')

    # Averaging metrics
    train_loss = round(train_loss/len(dataloader),2)
    train_acc = round(train_acc/len(dataloader),2)
    train_precision = round(train_precision/len(dataloader),2)
    train_recall = round(train_recall/len(dataloader),2)
    train_f1 = round(train_f1/len(dataloader),2)

    return train_loss, train_acc, train_precision, train_recall, train_f1


def test_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              device:torch.device) -> Tuple[float,float,float,float,float]:

    # Set up metrics
    test_loss, test_acc, test_precision, test_recall, test_f1 = 0, 0, 0, 0, 0

    # Turn off updating gradients
    model.eval()
    with torch.inference_mode():
        # Loop through batches
        for X, y in dataloader:

            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            logits = model(X)

            # 2. Compute loss
            loss = loss_fn(logits, y)

            # Get the most confident predictions
            predictions = torch.argmax(logits,dim=1).cpu()
            y = y.cpu()

            # Calculate metrics
            test_loss += loss.item()
            test_acc += accuracy_score(predictions, y)
            test_precision += precision_score(predictions, y, average='macro')
            test_recall += recall_score(predictions, y, average='macro')
            test_f1 += f1_score(predictions, y, average='macro')

    # Averaging metrics
    test_loss = round(test_loss/len(dataloader),2)
    test_acc = round(test_acc/len(dataloader),2)
    test_precision = round(test_precision/len(dataloader),2)
    test_recall = round(test_recall/len(dataloader),2)
    test_f1 = round(test_f1/len(dataloader),2)

    return test_loss, test_acc, test_precision, test_recall, test_f1


def create_writer(model_name:str, batch_size:int, dropout_rate:float) -> SummaryWriter:
    experiment_dir = Path('experiments/')
    experiment_dir.mkdir(parents=True,exist_ok=True)
    log_dir = experiment_dir / model_name / f'BS{batch_size}' / f'DR{int(dropout_rate*100)}perc'
    return SummaryWriter(log_dir=log_dir)


def remove_empty_logs(model_name:str, batch_size:int, dropout_rate:float):
    log_dir = Path("experiments") / model_name / f'BS{batch_size}' / f'DR{int(dropout_rate*100)}perc'
    empty_logs = [file for file in os.listdir(log_dir) if (log_dir/file).is_file()]
    for empty_log in empty_logs:
        os.remove(log_dir/empty_log)

def create_dataloaders(data_dir:PosixPath, train_threshold:float, batch_size:int,
                    transforms:torchvision.transforms._presets.ImageClassification):
    # Create datasets
    dataset = datasets.ImageFolder(root=data_dir, transform=transforms)
    train_size = int(train_threshold * len(dataset))
    test_size = int(len(dataset) - train_size)
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def train(model:torch.nn.Module,
            train_dataloader:torch.utils.data.DataLoader,
            test_dataloader:torch.utils.data.DataLoader,
            loss_fn:torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            device:torch.device,
            batch_size:int,
            dropout_rate:float,
            epochs:int,
            # if grad_clipping
            max_norm = False,
            # if scheduler
            scheduler = False
            ):
    # Config
    model_name = model.__class__.__name__
    model_path = Path('models/') / f'{model_name}_BS{batch_size}_DR{int(dropout_rate*100)}perc.pt'
    Path('models/').mkdir(parents=True,exist_ok=True)

    lowest_test_loss = float('inf')
    for epoch in tqdm(range(1,epochs+1)):
        # Train step
        if scheduler:
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_step(model,train_dataloader,loss_fn,optimizer,
                                                                                        device,max_norm,epoch=epoch,scheduler=scheduler)
        else:
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_step(model,train_dataloader,loss_fn,optimizer,
                                                                                            device,max_norm)
        print(f'EPOCH_{epoch}:\n \
            Train metrics: loss = {train_loss}; acc = {train_acc}; precision = {train_precision}; recall = {train_recall}; f1 = {train_f1}')

        # Test step
        test_loss, test_acc, test_precision, test_recall, test_f1 = test_step(model,test_dataloader,loss_fn,device)
        print(f'EPOCH_{epoch}:\n \
            Test metrics: loss = {test_loss}; acc = {test_acc}; precision = {test_precision}; recall = {test_recall}; f1 = {test_f1}')

        writer = create_writer(model_name, batch_size, dropout_rate)

        # Add results to SummaryWriter
        writer.add_scalars(main_tag='CrossEntropyLoss',
                           tag_scalar_dict={'train':train_loss,
                                             'test':test_loss},
                            global_step=epoch)
        writer.add_scalars(main_tag='Accuracy',
                           tag_scalar_dict={'train':train_acc,
                                             'test':test_acc},
                            global_step=epoch)
        writer.add_scalars(main_tag='Precision',
                           tag_scalar_dict={'train':train_precision,
                                             'test':test_precision},
                            global_step=epoch)
        writer.add_scalars(main_tag='Recall',
                           tag_scalar_dict={'train':train_recall,
                                             'test':test_recall},
                            global_step=epoch)

        writer.add_scalars(main_tag='F1',
                            tag_scalar_dict={'train':train_f1,
                                                'test':test_f1},
                            global_step=epoch)

        writer.close()
        
        # Removing unnecessary files in log_dir
        remove_empty_logs(model_name, batch_size, dropout_rate)

        # Checking if test_loss decreased
        if test_loss<lowest_test_loss:
            print(f'\n{"-"*80}\nTest loss decreased, saving model to "{model_path}"\n{"-"*80}\n')
            torch.save(model, model_path)
            lowest_test_loss = test_loss

    print(f"\nTraining completed\nMetrics saved in 'experiments/' and can be seen using TensorBoard")
    return model

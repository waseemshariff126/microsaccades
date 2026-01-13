import argparse
import shutil
import os
import time
import datetime
import torch
import logging as logger
import torch.nn as nn
from torch import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from models.VGG_models import vgg11, vgg13, vgg16, vgg16_with_flow
from functions import seed_all, build_microsaccade
from tqdm import tqdm
import torch.nn.functional as F
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Neuromorphic Data Augmentation')
parser.add_argument('--log_dir', default='./logs', type=str, help='Path for TensorBoard logs')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--dset', default='uS', type=str, choices=['nc101', 'uS'], help='dataset')
parser.add_argument('--model', default='vgg16_flow', type=str, choices=['vgg11', 'vgg13', 'vgg16', 'vgg16_flow'], help='neural network architecture')
parser.add_argument('--batch_size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, metavar='LR', dest='lr', help='initial learning rate')
parser.add_argument('--seed', default=1000, type=int, help='seed for initializing training')
parser.add_argument('--time', default=10, type=int, metavar='N', help='snn simulation time')
parser.add_argument('--amp', action='store_true', help='if use amp training')
parser.add_argument('--nda', action='store_true', help='if use neuromorphic data augmentation')
args = parser.parse_args()

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
run_log_dir = os.path.join(args.log_dir, f"logs.{timestamp}")
os.makedirs(run_log_dir, exist_ok=True)

writer = SummaryWriter(log_dir=run_log_dir)

def events_to_frames(event_stream):
    B, T, C, H, W = event_stream.shape
    frames = np.zeros((B, T, H, W), dtype=np.uint8)
    for b in range(B):
        for t in range(T):
            frames[b, t] = (event_stream[b, t, 0] + event_stream[b, t, 1]).cpu().numpy().clip(0, 255)
    return frames

def compute_flow(frames):
    B, T, H, W = frames.shape
    flow = np.zeros((B, T-1, H, W, 2), dtype=np.float32)
    for b in range(B):
        for t in range(T-1):
            prev = frames[b, t].astype(np.uint8)
            next = frames[b, t+1].astype(np.uint8)
            flow[b, t] = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_padded = np.pad(flow, ((0, 0), (0, 1), (0, 0), (0, 0), (0, 0)), mode='edge')
    return flow_padded

def downsample_flow(flow_gt, target_size):
    B, T, H, W, C = flow_gt.shape
    flow_gt = flow_gt.permute(0, 1, 4, 2, 3)
    flow_gt = F.interpolate(flow_gt.view(B*T, 2, H, W), size=target_size, mode='bilinear', align_corners=False)
    return flow_gt.view(B, T, 2, *target_size)

def train(model, device, train_loader, class_criterion, flow_criterion, optimizer, epoch, scaler, args):
    running_class_loss = 0
    running_flow_loss = 0
    correct, total = 0, 0
    model.train()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch")
    for images, labels in progress_bar:
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        
        # Debugging: Print input shape
        if epoch == 0 and progress_bar.n == 0:
            print(f"Input shape: {images.shape}")
        
        # Generate flow ground truth for vgg16_flow
        if args.model == 'vgg16_flow':
            frames = events_to_frames(images)
            flow_gt = torch.tensor(compute_flow(frames), dtype=torch.float32).to(device)
        
        if args.amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                if args.model == 'vgg16_flow':
                    class_out, flow_out = outputs
                    # Debugging: Print output shapes
                    if epoch == 0 and progress_bar.n == 0:
                        print(f"Class out shape: {class_out.shape}, Flow out shape: {flow_out.shape}")
                    # Downsample flow_gt to match flow_out
                    flow_gt = downsample_flow(flow_gt, target_size=(flow_out.shape[3], flow_out.shape[4]))
                    if epoch == 0 and progress_bar.n == 0:
                        print(f"Flow gt shape after downsample: {flow_gt.shape}")
                    
                    class_loss = class_criterion(class_out, labels)
                    flow_loss = flow_criterion(flow_out, flow_gt)
                    loss = class_loss + 5 * flow_loss
                else:
                    class_out = outputs
                    loss = class_criterion(class_out, labels)
                    flow_loss = torch.tensor(0.0)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            outputs = model(images)
            if args.model == 'vgg16_flow':
                class_out, flow_out = outputs
                if epoch == 0 and progress_bar.n == 0:
                    print(f"Class out shape: {class_out.shape}, Flow out shape: {flow_out.shape}")
                flow_gt = downsample_flow(flow_gt, target_size=(flow_out.shape[3], flow_out.shape[4]))
                if epoch == 0 and progress_bar.n == 0:
                    print(f"Flow gt shape after downsample: {flow_gt.shape}")
                
                class_loss = class_criterion(class_out, labels)
                flow_loss = flow_criterion(flow_out, flow_gt)
                loss = class_loss + 0.5 * flow_loss
            else:
                class_out = outputs
                loss = class_criterion(class_out, labels)
                flow_loss = torch.tensor(0.0)
            
            loss.backward()
            optimizer.step()
        
        running_class_loss += class_loss.item()
        if args.model == 'vgg16_flow':
            running_flow_loss += flow_loss.item()
        total += labels.size(0)
        correct += (class_out.argmax(1) == labels).sum().item()
        
        progress_bar.set_postfix({
            "class_loss": running_class_loss / total,
            "flow_loss": running_flow_loss / total if args.model == 'vgg16_flow' else 0,
            "samples": total
        })
    
    acc = 100 * correct / total
    writer.add_scalar('Train/Class_Loss', running_class_loss / len(train_loader), epoch)
    if args.model == 'vgg16_flow':
        writer.add_scalar('Train/Flow_Loss', running_flow_loss / len(train_loader), epoch)
    writer.add_scalar('Train/Accuracy', acc, epoch)
    print("-------------------------------------------------------------------------------------------------")
    print(f'Epoch {epoch}: Train Class Loss: {running_class_loss / len(train_loader):.4f}, '
          f'Train Accuracy: {acc:.2f}%', end='')
    if args.model == 'vgg16_flow':
        print(f', Train Flow Loss: {running_flow_loss / len(train_loader):.4f}')
    else:
        print()
    return running_class_loss / len(train_loader), acc

def test(model, test_loader, device, epoch):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            total += targets.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()

    acc = 100 * correct / total
    writer.add_scalar('Test/Accuracy', acc, epoch)
    print(f'Epoch {epoch}: Test Accuracy: {acc:.2f}%')
    return acc

if __name__ == '__main__':
    seed_all(args.seed)
    best_acc = 0.0
    
    if args.dset == 'uS':
        train_dataset, val_dataset = build_microsaccade(transform=args.nda)
        num_cls = 7
        in_c = 2
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    if args.model == 'vgg11':
        model = vgg11(in_c=in_c, num_classes=num_cls)
    elif args.model == 'vgg13':
        model = vgg13(in_c=in_c, num_classes=num_cls)
    elif args.model == 'vgg16':
        model = vgg16(in_c=in_c, num_classes=num_cls)
    elif args.model == 'vgg16_flow':
        model = vgg16_with_flow(in_c=in_c, num_classes=num_cls)
    else:
        raise NotImplementedError(f"Model {args.model} is not implemented.")
    
    model.T = args.time
    model.cuda()
    device = next(model.parameters()).device
    model = nn.DataParallel(model).to(device)
    total_params = sum(p.numel() for p in model.module.parameters())
    trainable_params = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    scaler = GradScaler() if args.amp else None
    class_criterion = nn.CrossEntropyLoss().to(device)
    flow_criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/256 * args.batch_size, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    print('Start training!')
    for epoch in range(args.epochs):
        loss, acc = train(model, device, train_loader, class_criterion, flow_criterion, optimizer, epoch, scaler, args)
        test_acc = test(model, test_loader, device, epoch)
        scheduler.step()

        checkpoint_path = os.path.join(run_log_dir, f'model_epoch_{epoch}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Model saved: {checkpoint_path}')
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_path = os.path.join(run_log_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model updated: {best_model_path}')
        
        print("-------------------------------------------------------------------------------------------------")

    writer.close()
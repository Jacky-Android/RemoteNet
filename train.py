from torch.utils.data import DataLoader
from potsdam_data import *
from thop import profile,clever_format

from train_utils import train_one_epoch, evaluate, create_lr_scheduler
from tqdm import tqdm
import os
import torch.multiprocessing
from timm.optim.lion import Lion
from models.RemoteNet import RemoteNet
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "3,2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def create_model(num_classes):
    model = RemoteNet(dim=64,dims=(64, 128, 320, 512),num_classes=6)
    return model

def main():
    batch_size = 16
    train_batch_size = batch_size
    val_batch_size = int(batch_size*0.5)
    epochs = 100

    train_dataset = PotsdamDataset(data_root='potsdam', mode='train',img_dir='images/train', mask_dir='anns/train',
                                   mosaic_ratio=0.25, transform=train_aug)

    val_dataset = PotsdamDataset(data_root='potsdam',img_dir='images/val', mask_dir='anns/val',transform=val_aug)

    num_workers = 2

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=train_batch_size,
                              num_workers=num_workers,
                              pin_memory=True,
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            num_workers=num_workers,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=6)
    model.to(device)
    model = torch.nn.DataParallel(model)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW( params_to_optimize,lr=6e-4, weight_decay=0.01)
    #optimizer = Lion(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)
    amp = False
    #scaler = None
    scaler = torch.cuda.amp.GradScaler()  
     # 创建学习率更新策略，这里是每个step更新一次(不是每个epoch)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=True)
    #lr_scheduler=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    num_classes = 6
    best_dice = 0
    best_miou = 0
    
    for epoch in range(0,epochs):
        mean_loss, lr,con = train_one_epoch(model, optimizer, train_loader, device, 
                                            epoch,lr_scheduler=lr_scheduler, print_freq=50, 
                                            scaler=scaler,num_classes=num_classes)
        #torch.cuda.empty_cache()
        train_info = str(con)
        print('train###############################################################')
        print(train_info)
        print('#####################################################################')
        confmat = evaluate(model, val_loader, device=device, num_classes=num_classes)
        print('val*******************************************************************')
        val_info = str(confmat)
        print(val_info)


        print('************************************************************************')
        #print(val_info.split('\n')[-1].split(':'))
        miou = float(val_info.split('\n')[-3].split(': ')[-1])
        if best_miou<miou:
            best_miou = miou
            val_info = str(confmat)
            save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch}
            torch.save(save_file, "IRbest_model.pth")
        if epoch == epochs-1:
            best_miou = miou
            val_info = str(confmat)
            save_file = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "lr_scheduler": lr_scheduler.state_dict(),
                             "epoch": epoch}
            torch.save(save_file, "IRlast_model.pth")
        torch.cuda.empty_cache()  # 释放显存
if __name__ == '__main__':
    main()
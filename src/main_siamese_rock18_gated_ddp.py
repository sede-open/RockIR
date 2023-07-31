import os, sys, timeit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pytorch version tested till 2.0.1
import torch
import torch.distributed as dist
import torch.nn as nn
import torchvision.models as models
from torch.distributed import Backend
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import dataloader.dataset_rock_voxel_perm as dataset_rock
import model.model_siamese_rock18_gated as model_define
import parser_define


# Pytorch Distributed Data Parellel Settings 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'

train_args = parser_define.train_args_define()
rank = train_args.local_rank
world_size = 4
n_gpus = 4

############################################
#Counting trainable parameters
############################################ 
def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

'''
############################################
#Define a training function. 
############################################ 
model_resnet : a backbone network for blue part of the proposed network
model_siam : a Siamese network which is composed of fc layers 
model_rock : another backbone network for gray part of the proposed network
model_reg : a regression module for metadata prediction
train_loader : a dataloader for training data
optimizer : an optimizer for backpropagation
scheduler : a learning rate scheduler
'''
def train(model_resnet, model_siam, model_rock, model_reg, train_loader, optimizer, args):
    if rank == 0:
        print('training a network')
    model_resnet.train()
    model_siam.train()
    model_rock.train()
    model_reg.train()

    for epoch in range(1, args.epochs + 1):
        # print('epochs', epoch)
        train_loader.sampler.set_epoch(epoch)
        for i, data in enumerate(train_loader,0):
            img0, img1, d0, d1, label, label_rock = data # [img0,img1] = a pair of images, [d0,d1] = metadata of img0, and img1, label=whether a pair of images is similar or not, label_rock = label=whether rock property of a pair of images is similar or not
            img0, img1, d0, d1, label, label_rock = img0.to(args.device), img1.to(args.device), d0.to(args.device), d1.to(args.device), label.to(args.device), label_rock.to(args.device)

            dist.barrier()
            optimizer.zero_grad()
                        
            # Predict outputs with Siamese resnet : blue part
            output1, output2 = model_resnet(img0), model_resnet(img1)
            
            # Preprocess a embedding of rock properties and pass the embedding of rock properties to a neuron for regression : gray part
            output3, output4 = model_rock(img0), model_rock(img1)
            output_reg1, output_reg2 = model_reg(output3), model_reg(output4)

            # Pass outputs through Siamese fc model
            output, output_img, output_rock = model_siam(output1, output2, output3, output4)

            # Define loss functions.
            loss_func_rock = nn.SmoothL1Loss() # nn.MSELoss()
            loss_rock = loss_func_rock(output_reg1, d0.view(-1,2)) + loss_func_rock(output_reg2, d1.view(-1,2))
        
            loss_func = nn.BCELoss() # Binary Cross Entropy loss 
            loss_img = loss_func(output_img, label.view(-1,1)).to(args.device)
            loss_rock_sim = loss_func(output_rock, label_rock.view(-1,1)).to(args.device)
            loss = loss_func(output, label.view(-1,1)*label_rock.view(-1,1)).to(args.device) + loss_img + loss_rock_sim + 10*loss_rock # Final loss function, a scalar 10 is for balancing purpose
            # print("similarity loss", loss_func(output, label).to(args.device), "rock regression loss", loss_rock)

            loss.backward()
            optimizer.step()

        # Save checkpoints every 10 epochs
        if epoch % 10 == 0 :
            # First, create a directory to save checkpoints.
            model_dir = os.getcwd() + '/checkpoints/' # saved model path 
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            if rank == 0:
                print("Epoch number {}\n Current training loss {}\n".format(epoch,loss.item()))
                torch.save(model_resnet.state_dict(), os.path.join(model_dir, 'epoch-{}.pt'.format(epoch))) # Save a (blue) backbone model.
                torch.save(model_rock.state_dict(), os.path.join(model_dir, 'rock_epoch-{}.pt'.format(epoch))) # Save a (gray) backbone model.
                torch.save(model_siam.state_dict(), os.path.join(model_dir, 'siam_epoch-{}.pt'.format(epoch))) # Save Siamese parts.
                torch.save(model_reg.state_dict(), os.path.join(model_dir, 'reg_epoch-{}.pt'.format(epoch))) # Save a metadata regression module.
                print("Model saved")


           
# Define a main function with models, dataloaders, and DDP with a training function.
def main_func():
    # Time
    starttime = timeit.default_timer()
    # print("The start time is :",starttime)
    
    torch.distributed.init_process_group(backend=Backend.NCCL, init_method='env://')
    torch.manual_seed(train_args.seed)
    train_args.device = torch.device(f'cuda:{rank}')
    # print('current device', train_args.device)

    # Setting a MicroCT dataset
    folder_dataset = datasets.ImageFolder(root=train_args.datapath + "/train")
    print("folder_dataset", folder_dataset.root)
    train_dataset = dataset_rock.Microct_large_Dataset(imageFolderDataset=folder_dataset,
                                            transform=transforms.Compose([
                                                transforms.Resize((224, 224)),
                                                transforms.CenterCrop((224, 224)),
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomVerticalFlip(p=0.5),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5,), (0.5,))
                                                ])
                                        ,should_invert=False
                                        ,train=True)

    # Create distributed sampler pinned to rank
    sampler = DistributedSampler(train_dataset,
                            num_replicas=world_size,
                            rank=rank,
                            shuffle=True,  # May be True
                            seed=42)

    # Wrap train dataset into DataLoader
    train_loader = DataLoader(train_dataset,
                        batch_size=int(train_args.train_batch_size / n_gpus), # = /#of GPUS
                        shuffle=False,  # Must be False!
                        num_workers=4*n_gpus, # 4 *#of GPUS
                        sampler=sampler,
                        pin_memory=True)

    torch.cuda.set_device(rank)

    # A backbone network (blue part-resnet18)
    model_resnet = models.resnet18(pretrained=True) 
    model_resnet = torch.nn.Sequential(*(list(model_resnet.children())[:-2])) # Remove last fc layer
    model_resnet.to(train_args.device)
    
    # A Siamese network composed of FC layers
    model_siamese = model_define.Siamese()
    model_siamese.to(train_args.device)

    # A backbone network (gray part-resnet18)
    model_rock = models.resnet18(pretrained=True) 
    model_rock = torch.nn.Sequential(*(list(model_rock.children())[:-2])) # Remove last fc layer
    model_rock.to(train_args.device)

    # A regression module for metadata(rock properties)
    model_reg = model_define.Regression_rock()
    model_reg.to(train_args.device)

    model_resnet = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_resnet)
    model_rock = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_rock)

    # DDP settings 
    pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
    pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
    pg3 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
    pg4 = torch.distributed.new_group(range(torch.distributed.get_world_size()))  
    
    model_resnet = DistributedDataParallel(model_resnet, device_ids=[rank], process_group=pg1)
    model_rock = DistributedDataParallel(model_rock, device_ids=[rank], process_group=pg2)
    model_siamese = DistributedDataParallel(model_siamese, device_ids=[rank], process_group=pg3)
    model_reg = DistributedDataParallel(model_reg, device_ids=[rank], process_group=pg4)

    # Count the number of training parameters
    '''
    print(count_parameters(model_resnet))
    print(count_parameters(model_siamese))
    print(count_parameters(model_rock))
    print(count_parameters(model_reg))
    '''

    # Optimizer for backpropagation
    optimizer = torch.optim.AdamW([
                {'params': model_siamese.parameters(), 'lr': n_gpus*train_args.lr},
                {'params': model_resnet.parameters(), 'lr': n_gpus*train_args.lr},
                {'params': model_rock.parameters(), 'lr': n_gpus*train_args.lr},
                {'params': model_reg.parameters(), 'lr': n_gpus*train_args.lr}
            ], weight_decay=1e-5, amsgrad=True)
    
    # train_args.test_loader = test_loader
    train(model_resnet, model_siamese, model_rock, model_reg, train_loader, optimizer, train_args)
    print("The total time is :", timeit.default_timer() - starttime)

if __name__ == '__main__':
    main_func()

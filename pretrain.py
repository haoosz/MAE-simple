import torch
from models import ViT, MAE
from utils.data import get_dataloader
from torch.nn.utils import clip_grad_norm_
import math
from einops import rearrange
import argparse
import wandb
from tqdm import tqdm
import os
import torch.multiprocessing as mp

def get_args():
    parser = argparse.ArgumentParser('MAE pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--comment', default="pretrain", type=str)

    # Model parameters
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')

    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size for backbone')

    parser.add_argument('--num_patches', default=8, type=int,
                        help='number of patches per dimension')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1.5e-4, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--min_lr_n', type=float, default=1e-2, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Dataset parameters
    parser.add_argument('--data_dir', default='../data', type=str,
                        help='dataset path')
    parser.add_argument('--download', default=False, action="store_true")
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers')
    parser.add_argument('--num_classes', default=100,
                        help='number of classes')
    parser.add_argument("--dataset", default="CIFAR100", type=str, help="dataset")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    # distributed parameters
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    return parser.parse_args()

class trainer():
    def __init__(self, args, gpu):
        self.args = args
        self.gpu = gpu
        self.patch_size = int(self.args.input_size/self.args.num_patches) # 32/8=4
        self.v = ViT(
            image_size = self.args.input_size,
            patch_size = self.patch_size,
            num_classes = self.args.num_classes,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048
        )

        self.mae = MAE(
            encoder = self.v,
            masking_ratio = self.args.mask_ratio,   # the paper recommended 75% masked patches
            decoder_dim = 512,      # paper showed good results with just 512
            decoder_depth = 6       # anywhere from 1 to 8
        )

        self.to_cuda()
        self.config_optim()
    
    def to_cuda(self):
        self.v.cuda(self.gpu)
        self.mae.cuda(self.gpu)
        self.mae = torch.nn.parallel.DistributedDataParallel(self.mae, device_ids=[self.gpu], find_unused_parameters=True)

    def config_optim(self):
        params = [x for x in self.mae.parameters() if x.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=self.args.lr, betas=(0.9,0.95), eps=1e-8, weight_decay=self.args.weight_decay, amsgrad=False)

        lambda_sch = lambda e: (1-self.args.min_lr_n)*e/self.args.warmup_epochs+self.args.min_lr_n if e<self.args.warmup_epochs \
            else (1-self.args.min_lr_n)*(1+math.cos(math.pi*(e-self.args.warmup_epochs)/(self.args.epochs-self.args.warmup_epochs)))/2+self.args.min_lr_n
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_sch)

    def forward_and_backward(self, image):
        self.mae.train()
        # make optimizer zero grad
        self.optimizer.zero_grad()
        # forward and backward
        loss, self.recon_patch, self.org_patch = self.mae(image)
        loss.backward()
        # step optimizier
        if self.args.clip_grad:
            clip_grad_norm_(self.mae.parameters(), self.args.clip_grad)
        self.optimizer.step()
        return loss
    
    def update_lr(self):
        self.scheduler.step()
    
    def get_image_results(self):
        recon_patch = rearrange(self.recon_patch, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = self.args.num_patches, p1 = self.patch_size, p2 = self.patch_size)
        org_patch = rearrange(self.org_patch, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h = self.args.num_patches, p1 = self.patch_size, p2 = self.patch_size)
        return recon_patch, org_patch
    
    def save_checkpoints(self, epoch):
        torch.save(self.mae.state_dict(), './checkpoints/trained-mae-{:0>4d}.pt'.format(epoch))
    
    def evaluate(self, image):
        with torch.no_grad():
            self.mae.eval()
            loss, self.recon_patch, self.org_patch = self.mae(image)
            recon_patch, org_patch = self.get_image_results()
        return loss, recon_patch, org_patch

def train(gpu, args):
    rank = args.nr * args.gpus + gpu	                          
    torch.distributed.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )      
    torch.cuda.set_device(gpu)
    torch.manual_seed(args.seed)
    
    if gpu == 0:
        wandb.init(project="MAE", name=args.comment, config=args)

    dl_train, dl_val = get_dataloader(args, rank, "pretrain")
    trner = trainer(args, gpu)
    for e in range(args.epochs):
        loss_epoch = 0
        pbar = tqdm(enumerate(dl_train),total=len(dl_train))
        for i, (image, _) in pbar:
            image = image.cuda(gpu)
            loss = trner.forward_and_backward(image)
            loss_epoch += loss
            pbar.set_postfix({"loss":"{:.3f}".format(loss)})

        trner.update_lr()
        recon_patch, org_patch = trner.get_image_results()

        if gpu == 0:
            wandb.log({"train_loss": loss_epoch/len(dl_train)})

            # save checkpoints
            if e % args.save_ckpt_freq == 0 and e != 0:
                trner.save_checkpoints(e)
                # logging
                wandb.log({"train_recon": [wandb.Image(im) for im in recon_patch[:20]]})
                wandb.log({"train_org": [wandb.Image(im) for im in org_patch[:20]]})

            # evaluate
            if e % args.eval_freq == 0 and e != 0:
                loss_eval = 0
                for _, (image, _) in enumerate(dl_val):
                    image = image.cuda(gpu)
                    loss, recon_patch, org_patch = trner.evaluate(image)
                    loss_eval += loss
                # logging
                wandb.log({"eval_recon": [wandb.Image(im) for im in recon_patch[:20]]})
                wandb.log({"eval_org": [wandb.Image(im) for im in org_patch[:20]]})
                wandb.log({"eval_loss": loss_eval/len(dl_val)})

    torch.save(trner.v.state_dict(), './trained-vit.pt')

if __name__ == "__main__":
    args = get_args()
    args.world_size = args.gpus * args.nodes

    os.environ['MASTER_ADDR'] = '10.64.45.220'
    os.environ['MASTER_PORT'] = '8888'
    
    mp.spawn(train, nprocs=args.gpus, args=(args,)) 
import torch
from models import ViT
from utils.data import get_dataloader
from torch.nn.utils import clip_grad_norm_
import math
from einops import rearrange
import argparse
import wandb
from tqdm import tqdm
import os
import torch.multiprocessing as mp
import torch.nn.functional as F
from timm.loss import SoftTargetCrossEntropy
from timm.data.mixup import Mixup
import json

def get_args():
    parser = argparse.ArgumentParser('Encoder supervised-training script', add_help=False)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--comment', default="sup", type=str)

    # Model parameters
    parser.add_argument('--input_size', default=32, type=int,
                        help='images input size for backbone')

    parser.add_argument('--num_patches', default=8, type=int,
                        help='number of patches per dimension')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.5e-4)')
    parser.add_argument('--min_lr_n', type=float, default=1e-2, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--lr_decay', type=float, default=0.65,
                        help='lr decay (default: 0.05)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")

    # Dataset parameters
    parser.add_argument('--data_dir', default='../data', type=str,
                        help='dataset path')
    parser.add_argument('--download', default=False, action="store_true")
    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of workers')
    parser.add_argument('--num_classes', default=80,
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

def get_layer(name):
    name_ls = name.split(".")
    return(int(name_ls[3]))

def get_parameter_groups(model, base_lr, lr_decay, n_layers=6):
    parameter_group_names = {}
    parameter_group_vars = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        elif "mlp" in name:
            group_name = "mlp"
            decay_scale = 0
        elif "transformer" in name:
            layer_id = get_layer(name)
            if layer_id == n_layers - 1:
                group_name = "layer{:0>2d}".format(layer_id)
                decay_scale = n_layers - layer_id
            else:
                continue
        # elif "to_patch" in name:
        #     group_name = "to_patch"
        #     decay_scale = n_layers + 1
        else:
            continue

        if group_name not in parameter_group_names:

            parameter_group_names[group_name] = {
                "params": [],
                "lr": (lr_decay**decay_scale) * base_lr,
            }
            parameter_group_vars[group_name] = {
                "params": [],
                "lr": (lr_decay**decay_scale) * base_lr,
            }

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)
    for k, v in parameter_group_names.items():   
        print("{}:{}".format(k, v["lr"]))
    return list(parameter_group_vars.values())

class trainer():
    def __init__(self, args, gpu):
        self.args = args
        self.gpu = gpu
        self.patch_size = int(self.args.input_size/self.args.num_patches) # 32/8=4

        self.vit = ViT(
            image_size = self.args.input_size,
            patch_size = self.patch_size,
            num_classes = self.args.num_classes,
            dim = 1024,
            depth = 6,
            heads = 8,
            mlp_dim = 2048
        )
        
        self.load_pretrain()
        self.to_cuda()
        self.config_optim()
        self.criterion = SoftTargetCrossEntropy()

        self.mixup_fn = None
        self.mixup_active = self.args.mixup > 0 or self.args.cutmix > 0. or self.args.cutmix_minmax is not None
        if self.mixup_active:
            print("Mixup is activated!")
            self.mixup_fn = Mixup(
                mixup_alpha=self.args.mixup, cutmix_alpha=self.args.cutmix, cutmix_minmax=self.args.cutmix_minmax,
                prob=self.args.mixup_prob, switch_prob=self.args.mixup_switch_prob, mode=self.args.mixup_mode,
                label_smoothing=self.args.smoothing, num_classes=self.args.num_classes)

    def load_pretrain(self):
        state_dict = torch.load("trained-vit.pt", map_location=torch.device('cpu'))
        self.vit.load_state_dict(state_dict, strict=False)

    def to_cuda(self):
        self.vit.cuda(self.gpu)
        self.vit = torch.nn.parallel.DistributedDataParallel(self.vit, device_ids=[self.gpu], find_unused_parameters=True)

    def config_optim(self):
        # params = [x for x in self.vit.parameters() if x.requires_grad]
        params_group = get_parameter_groups(self.vit, self.args.lr, self.args.lr_decay)
        self.optimizer = torch.optim.AdamW(params_group, betas=(0.9,0.999), eps=1e-8, weight_decay=self.args.weight_decay, amsgrad=False)

        lambda_sch = lambda e: (1-self.args.min_lr_n)*e/self.args.warmup_epochs+self.args.min_lr_n if e<self.args.warmup_epochs \
            else (1-self.args.min_lr_n)*(1+math.cos(math.pi*(e-self.args.warmup_epochs)/(self.args.epochs-self.args.warmup_epochs)))/2+self.args.min_lr_n
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_sch)

    def forward_and_backward(self, image, target):
        if self.mixup_fn is not None:
            image, target = self.mixup_fn(image, target)

        self.vit.train()
        # make optimizer zero grad
        self.optimizer.zero_grad()
        # forward and backward
        pred = self.vit(image)
        loss = self.criterion(pred / self.args.temperature, target)
        loss.backward()
        # step optimizier
        if self.args.clip_grad:
            clip_grad_norm_(self.vit.parameters(), self.args.clip_grad)
        self.optimizer.step()
        return loss
    
    def update_lr(self):
        self.scheduler.step()
    
    def save_checkpoints(self, epoch):
        dir_save = './checkpoints/{}'.format(self.args.comment)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        torch.save(self.vit.state_dict(), dir_save+'/trained-encoder-{:0>4d}.pt'.format(epoch))
    
    def evaluate(self, image, target):
        with torch.no_grad():
            self.vit.eval()
            pred = self.vit(image)
            acc = sum(torch.argmax(pred,1) == target) / target.size(0)
        return acc

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
    
    if rank == 0:
        wandb.init(project="MAE", name=args.comment, config=args)
    torch.distributed.barrier()

    dl_train, dl_val = get_dataloader(args, rank, "supervise")
    trner = trainer(args, gpu)
    for e in range(args.epochs):
        loss_epoch = 0
        pbar = tqdm(enumerate(dl_train),total=len(dl_train))
        for _, (image, target) in pbar:
            pbar.set_description("Processing Epoch %d" % e)
            image = image.cuda(gpu)
            target = target.cuda(gpu)
            loss = trner.forward_and_backward(image, target)
            loss_epoch += loss
            pbar.set_postfix({"loss":"{:.3f}".format(loss)})

        trner.update_lr()

        if rank == 0:
            wandb.log({"train_loss": loss_epoch/len(dl_train)})

            # save checkpoints
            if e % args.save_ckpt_freq == 0 and e != 0:
                trner.save_checkpoints(e)

            # evaluate
            if e % args.eval_freq == 0 and e != 0:
                acc_eval = 0
                for _, (image, target) in enumerate(dl_val):
                    image = image.cuda(gpu)
                    target = target.cuda(gpu)
                    acc = trner.evaluate(image, target)
                    acc_eval += acc
                # logging
                wandb.log({"eval_acc": acc_eval/len(dl_val)})
        torch.distributed.barrier()
    
    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    args = get_args()
    args.world_size = args.gpus * args.nodes

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '4444'

    mp.spawn(train, nprocs=args.gpus, args=(args,)) 

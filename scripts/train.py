import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import sys

sys.path.append("/home/compu/cleancode/simswap")
from utils import utils
from core.checkpoint import ckptIO
from core.loss import LossCollector
from core.options import train_options
from core.dataset import FaceDataset
from nets.simswap import Generator_Adain_Upsample, Discriminator


def train(gpu, args): 
    torch.cuda.set_device(gpu)

    G = Generator_Adain_Upsample(input_nc=3, output_nc=3, style_dim=512, n_blocks=9).cuda(gpu).train()
    D1 = Discriminator(input_nc=3).cuda(gpu).train()
    D2 = Discriminator(input_nc=3).cuda(gpu).train()

    dataset = FaceDataset(args.dataset_list, same_prob=args.same_prob)
    sampler = None

    if args.use_mGPU:
        args.isMaster = gpu==0

        utils.setup_ddp(gpu, args.gpu_num)

        # Data parallelism is required to use multi-GPU
        G = torch.nn.parallel.DistributedDataParallel(G, device_ids=[gpu], broadcast_buffers=False, find_unused_parameters=True)
        D1 = torch.nn.parallel.DistributedDataParallel(D1, device_ids=[gpu])
        D2 = torch.nn.parallel.DistributedDataParallel(D2, device_ids=[gpu])
        G = G.module
        D1 = D1.module
        D2 = D2.module
   
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, sampler=sampler, num_workers=8, drop_last=True)
    batch_iterator = iter(dataloader)

    # Load parameters from checkpoint
    ckptio = ckptIO(args)
    ckptio.load_ckpt(args, G)
    
    opt_G = torch.optim.Adam(G.parameters(), lr=args.lr_G, betas=(0, 0.999))
    opt_D = torch.optim.Adam([*D1.parameters(), *D2.parameters()], lr=args.lr_D, betas=(0, 0.999))

    loss_collector = LossCollector(args)

    # Initialize wandb to gather and display loss on dashboard 
    if args.isMaster:
        wandb.init(project=args.project_id, name=args.run_id)

    downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    # Iteration loop
    global_step = -1
    while global_step < args.max_step:
        global_step += 1

        # Load next batch
        try:
            I_source, I_target, same_person = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(dataloader)
            I_source, I_target, same_person = next(batch_iterator)
        
        # Transfer data to the GPU
        I_source, I_target, same_person = I_source.to(gpu), I_target.to(gpu), same_person.to(gpu)

        ###########
        # train G #
        ###########

        # Run G
        G_list = [I_source, I_target, same_person]

        I_swapped = G(I_source, I_target)
        G_list += [I_swapped]

        # Down sample
        I_swapped_downsampled = downsample(I_swapped)
        I_target_downsampled = downsample(I_target)

        # Run D
        g_fake1 = D1.forward(I_swapped)
        g_fake2 = D2.forward(I_swapped_downsampled)
        g_real1 = D1.forward(I_target)
        g_real2 = D2.forward(I_target_downsampled)
        G_list += [g_fake1, g_fake2, g_real1, g_real2]
        
        # Run arc face
        id_swapped = G.get_id(I_swapped)
        id_source = G.get_id(I_source)
        G_list += [id_swapped, id_source]

        # Calculate G Loss
        loss_G = loss_collector.get_loss_G(*G_list)

        # Update generator
        utils.update_net(opt_G, loss_G)

        ###########
        # train D #
        ###########

        # run D
        D_list = []

        # D_Real
        d_real1 = D1.forward(I_target)
        d_real2 = D2.forward(I_target_downsampled)
        D_list += [d_real1, d_real2]

        # D_Fake
        d_fake1 = D1.forward(I_swapped.detach())
        d_fake2 = D2.forward(I_swapped_downsampled.detach())
        D_list += [d_fake1, d_fake2]

        # Calculate D Loss
        loss_D = loss_collector.get_loss_D(*D_list)

        # Update Discriminator
        utils.update_net(opt_D, loss_D)

        ################
        # log and save #
        ################

        # Save and print loss
        if args.isMaster and global_step % args.loss_cycle==0:
            wandb.log(loss_collector.loss_dict)
            loss_collector.print_loss(global_step)

        # Save image
        if args.isMaster and global_step % args.image_cycle == 0:
            utils.save_image(args, global_step, "imgs", [I_source, I_target, I_swapped])

        # Save checkpoint parameters 
        if global_step % args.ckpt_cycle == 0:
            ckptio.save_ckpt(args, global_step, G)


if __name__ == "__main__":
    args = train_options()
    os.makedirs(args.save_root, exist_ok=True)

    # Set up multi-GPU training
    if args.use_mGPU:  
        args.gpu_num = torch.cuda.device_count()
        args.batch_size = int(args.batch_size / args.gpu_num)
        torch.multiprocessing.spawn(train, nprocs=args.gpu_num, args=(args, ))

    # Set up single GPU training
    else:  
        args.isMaster = True
        train(args.gpu_id, args)

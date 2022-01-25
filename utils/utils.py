import torch
import torchvision
import cv2
import os

def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X).mean()
    else:
        return torch.relu(X+1).mean()

def get_grid_row(images):

    # get 8 images
    images = images[:8]

    # make one row
    grid_row = torchvision.utils.make_grid(images.detach().cpu(), nrow=images.shape[0]) * 0.5 + 0.5

    return grid_row

def save_image(args, global_step, dir, images):

    # make dir
    os.makedirs(f'{args.save_root}/{args.run_id}/{dir}', exist_ok=True)
    
    # make grid
    sample_image = make_image(images).transpose([1,2,0])*255
    
    # set path
    save_path = f'{args.save_root}/{args.run_id}/{dir}/e{global_step}.jpg'

    # save image
    cv2.imwrite(save_path, sample_image[:,:,::-1])

def make_image(images):

    grid_rows = []

    # convert each image tensor to row
    for image in images:

        # get one row
        grid_row = get_grid_row(image)

        # append row
        grid_rows.append(grid_row)

    # make grid
    grid = torch.cat(grid_rows, dim=1).numpy()

    return grid

def setup_ddp(gpu, ngpus_per_node):

    # setup ddp
    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=ngpus_per_node,
            rank=gpu)

def update_net(optimizer, loss):

    # clear old gradients
    optimizer.zero_grad()
    
    # computes derivative of loss 
    loss.backward()

    # take one step
    optimizer.step()
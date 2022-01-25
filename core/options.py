
import argparse

def train_options():
    parser = argparse.ArgumentParser()

    # experiment id
    parser.add_argument('--run_id', type=str, required=True) 
    parser.add_argument('--gpu_id', type=int, default=0) 
    parser.add_argument('--ckpt_id', type=str, default=None)
    parser.add_argument('--project_id', type=str, default="simsawp")

    # hyperparameters
    parser.add_argument('--batch_size', type=str, default=32)
    parser.add_argument('--max_step', type=str, default=200000)
    parser.add_argument('--same_prob', type=float, default=0.2)

    # dataset
    parser.add_argument('--dataset_list', type=list, \
        # default=['/home/compu/dataset/k_face_aligned_cv2_1024'])
        default=['/home/compu/dataset/CelebHQ'])

    # learning rate
    parser.add_argument('--lr_G', type=str, default=4e-4)
    parser.add_argument('--lr_D', type=str, default=4e-4)

    # log
    parser.add_argument('--loss_cycle', type=str, default=10)
    parser.add_argument('--image_cycle', type=str, default=1000)
    parser.add_argument('--ckpt_cycle', type=str, default=10000)
    parser.add_argument('--save_root', type=str, default="training_result")

    # weight
    parser.add_argument('--W_adv', type=float, default=1)
    parser.add_argument('--W_id', type=float, default=10)
    parser.add_argument('--W_attr', type=float, default=0)
    parser.add_argument('--W_recon', type=float, default=10)
    parser.add_argument('--W_fm', type=float, default=10)
    parser.add_argument('--W_GP', type=float, default=1e-5)

    # multi GPU
    parser.add_argument('--isMaster', default=False)
    parser.add_argument('--use_mGPU', action='store_false')

    # use wandb
    parser.add_argument('--use_wandb', action='store_false')

    # args
    return parser.parse_args()
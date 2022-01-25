import torch
import os

class ckptIO():
    def __init__(self, args):
        super(ckptIO, self).__init__()
        self.args = args
        
    def load_ckpt(self, args, G):
        try:
            # set path
            G_ckpt_path = f'{args.save_root}/{args.ckpt_id}/ckpt/G_latest.pt'
            # load ckpt
            G_ckpt = torch.load(G_ckpt_path, map_location=torch.device('cuda'))

            # load state dict
            G.load_state_dict(G_ckpt, strict=False)

        except Exception as e:
            print(e)

    def save_ckpt(self, args, global_step, G):
        os.makedirs(f'{args.save_root}/{args.run_id}/ckpt', exist_ok=True)
        
        G_ckpt_path = f'{args.save_root}/{args.run_id}/ckpt/G_{global_step}.pt'
        torch.save(G.state_dict(), G_ckpt_path)

        G_ckpt_path_latest = f'{args.save_root}/{args.run_id}/ckpt/G_latest.pt'
        torch.save(G.state_dict(), G_ckpt_path_latest)
        
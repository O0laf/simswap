import torch
import time


class lossCollector():
    def __init__(self, args):
        super(lossCollector, self).__init__()
        self.args = args
        self.start_time = time.time()
        self.loss_dict = {}
        self.L1 = torch.nn.L1Loss()

    def get_id_loss(self, a, b):
        return (1 - torch.cosine_similarity(a, b, dim=1)).mean()

    def get_L1_loss_with_same_person(self, a, b, same_person):
        return torch.sum(0.5 * torch.mean(torch.abs(a - b).reshape(self.args.batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

    def get_L2_loss_with_same_person(self, a, b, same_person):
        return torch.sum(0.5 * torch.mean(torch.pow(a - b, 2).reshape(self.args.batch_size, -1), dim=1) * same_person) / (same_person.sum() + 1e-6)

    def get_attr_loss(self, a, b):
        L_attr = 0
        for i in range(len(a)):
            L_attr += torch.mean(torch.pow((a[i] - b[i]), 2).reshape(self.args.batch_size, -1), dim=1).mean()
        L_attr /= 2.0

        return L_attr
        
    def get_hinge_loss(self, input, label, for_discriminator=True):
        if isinstance(input, list):
            input = input[-1]
        if for_discriminator:
            if label:
                minval = torch.min(input - 1, torch.zeros_like(input, device='cuda'))
                loss = -torch.mean(minval)
            else:
                minval = torch.min(-input - 1, torch.zeros_like(input, device='cuda'))
                loss = -torch.mean(minval)
        else:
            loss = -torch.mean(input)
        return loss
        
    def get_loss_G(self, I_source, I_target, same_person, I_swapped, g_fake1, g_fake2, g_real1, g_real2, id_swapped, id_source):
        L_G = 0.0
        
        # adv loss
        if self.args.W_adv:
            L_adv = 0
            L_adv += self.get_hinge_loss(g_fake1, True, for_discriminator=False)
            L_adv += self.get_hinge_loss(g_fake2, True, for_discriminator=False)
            L_G += self.args.W_adv * L_adv
            self.loss_dict["L_adv"] = round(L_adv.item(), 4)
        
        # id loss
        if self.args.W_id:
            L_id = self.get_id_loss(id_source, id_swapped)
            L_G += self.args.W_id * L_id
            self.loss_dict["L_id"] = round(L_id.item(), 4)

        # attr loss
        if self.args.W_attr:
            L_attr = self.get_attr_loss(I_target, I_swapped)
            L_G += self.args.W_attr * L_attr
            self.loss_dict["L_attr"] = round(L_attr.item(), 4)

        # recon_loss
        if self.args.W_recon:
            L_recon = self.get_L1_loss_with_same_person(I_swapped, I_target, same_person)
            L_G += self.args.W_recon * L_recon
            self.loss_dict["L_recon"] = round(L_recon.item(), 4)
        
        # feature matching loss 
        if self.args.W_fm:
            L_fm = 0
            n_layers_D = 4
            num_D = 2
            feat_weights = 4.0 / (n_layers_D + 1)
            D_weights = 1.0 / num_D
            for i in range(0, n_layers_D):
                L_fm += D_weights * feat_weights * self.L1(g_fake1[i], g_real1[i].detach())
                L_fm += D_weights * feat_weights * self.L1(g_fake2[i], g_real2[i].detach())
            L_G += self.args.W_fm * L_fm
            self.loss_dict["L_fm"] = round(L_recon.item(), 4)

        # save in dict
        self.loss_dict["L_G"] = round(L_G.item(), 4)

        return L_G

    def get_loss_D(self, d_real1, d_real2, d_fake1, d_fake2):
        # real 
        L_D_real = 0
        L_D_real += self.get_hinge_loss(d_real1, True, for_discriminator=True)
        L_D_real += self.get_hinge_loss(d_real2, True, for_discriminator=True)

        # fake
        L_D_fake = 0
        L_D_fake += self.get_hinge_loss(d_fake1, False, for_discriminator=True)
        L_D_fake += self.get_hinge_loss(d_fake2, False, for_discriminator=True)

        L_D = 0.5*(L_D_real.mean() + L_D_fake.mean())
        
        # save in dict
        self.loss_dict["L_D_real"] = round(L_D_real.mean().item(), 4)
        self.loss_dict["L_D_fake"] = round(L_D_fake.mean().item(), 4)
        self.loss_dict["L_D"] = round(L_D.item(), 4)

        return L_D
    
    def print_loss(self, global_step):
        seconds = int(time.time() - self.start_time)
        print("")
        print(f"[ {seconds//3600//24:02}d {(seconds//3600)%24:02}h {(seconds//60)%60:02}m {seconds%60:02}s ]")
        print(f'steps: {global_step:06} / {self.args.max_step}')
        print(f'lossD: {self.loss_dict["L_D"]} | lossG: {self.loss_dict["L_G"]}')
    
import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image

import sys
sys.path.append("/home/compu/samplecode_faceshifter")
sys.path.append("/home/compu/samplecode_faceshifter/submodules")

from realesrgan import RealESRGANer
from gfpgan import GFPGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

from nets.BiSeNet import BiSeNet
from nets.AEI_Net import *
from nets.arcface import Backbone

from facenet_pytorch import MTCNN

import torchvision.transforms as transforms
import torch.nn.functional as F
from utils.align_trans import warp_and_crop_face, get_reference_facial_points

class FS():
    def __init__(self):
        self.device = 'cuda'
        self.do_superresolution = True
        self.load_FD_model()
        self.load_AF_model()
        self.load_FS_model()
        self.load_SR_model()
        self.load_FP_model()
        self.get_mask(1024)
        self.ref5points = get_reference_facial_points(default_square=True)
        self.test_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def do_swap(self, Xs, Xt):
        # input = -1~1
        # output = 0~1
        with torch.no_grad():
            Yt, _, _ = self.FS_model(Xs, Xt)
        return Yt

    def do_sr(self, Yt):
        with torch.no_grad():
            Yt = Yt.squeeze().detach().cpu().numpy().transpose([1, 2, 0])*0.5 + 0.5
            cv2.imwrite("Yt.jpg", Yt[:,:,::-1]*255)
            Yt = self.SR_model.enhance(Yt*255, outscale=4)[0]/255
        return Yt

    def do_eye(self, Xt, Yt):
        with torch.no_grad():
            Yt_ = transforms.ToTensor()(Yt).unsqueeze(0).float()
            parsing = self.FP_model(Yt_).max(1)[1]
            
            mask = np.ones((1024, 1024))
            mask -= torch.where(parsing==4, 1, 0).squeeze().cpu().numpy()
            mask -= torch.where(parsing==5, 1, 0).squeeze().cpu().numpy()
            mask = mask.clip(0,1)
            mask = cv2.blur(mask.astype(np.float64)*255, (5,5)).astype(np.uint8)/255
            mask = np.expand_dims(mask, 2)
            Yt = mask*Yt + (1-mask)*Xt
        return Yt

    def load_AF_model(self):
        arcface = Backbone(50, 0.6, 'ir_se')
        arcface.eval()
        arcface.load_state_dict(torch.load('ptnn/model_ir_se50.pth', map_location=self.device), strict=False)

        self.AF_model = arcface.to(self.device)

    def load_FD_model(self): 
        self.FD_model = MTCNN()

    def load_FS_model(self):
        G = AEI_Net(c_id=512).to(self.device).eval()
        G.load_state_dict(torch.load('/home/compu/samplecode_faceshifter/training_result/id7/ckpt/G_200000.pt', map_location=self.device))
        self.FS_model = G.to(self.device)

    def load_SR_model(self):
        print("Loading SR model,,,")
        SR_model_path = "ptnn/RealESRGAN_x4plus.pth"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
                scale=4,
                model_path=SR_model_path,
                model=model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False)

        # face enhancement
        face_enhancer = GFPGANer(
            model_path='https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth',
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=upsampler)

        print("Success!")
        print("")
        
        self.SR_model = upsampler
            
    def load_FP_model(self):
        # segmentation 
        segmentation_net = BiSeNet(n_classes=19)
        segmentation_net.load_state_dict(torch.load('ptnn/79999_iter.pth', map_location=self.device))
        segmentation_net.eval()
        self.FP_model = segmentation_net

    def get_mask(self, size):
        half = size // 2
        mask_grad = np.zeros([size, size], dtype=float)
        for i in range(size):
            for j in range(size):
                dist = np.sqrt((i-half)**2 + (j-half)**2)/half
                dist = np.minimum(dist, 1)
                mask_grad[i, j] = 1-dist
        mask_grad = cv2.dilate(mask_grad, None, iterations=20)
        self.mask_grad = mask_grad

    def combine(self, Yt, trans_inv, Xt_raw):
        mask = cv2.warpAffine(self.mask_grad, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        Ytt = cv2.warpAffine(Yt, trans_inv, (np.size(Xt_raw, 1), np.size(Xt_raw, 0)), borderValue=(0, 0, 0))
        mask = np.expand_dims(mask, 2)
        Ytt = mask*Ytt + (1-mask)*np.array(Xt_raw).astype(float)/255.0
        return Ytt

    def get_face(self, img):
        try:
            _, _, lms = self.FD_model.detect(img, landmarks=True)

        except Exception as e:
            print(e)
            print('Please change the image')

        lm = np.array(lms).astype(np.int32)[0]
        face1024, trans_inv = warp_and_crop_face(np.array(img), lm, self.ref5points, crop_size=(1024, 1024))
        face256 = self.test_transform(Image.fromarray(face1024))
        face256 = face256.unsqueeze(0).cuda()

        return face1024/255, face256, trans_inv

if __name__ == "__main__":
    Xs_path = "samples/ref/00021.png"
    Xt_path = "samples/inputs/PKI00585.jpg"
    save_path = "samples/outputs/hey.jpg"
    models = FS()
    
    # get Xta
    Xt_raw = Image.open(Xt_path)
    Xt1024, Xt256, trans_inv = models.get_face(Xt_raw)

    # get Xs
    Xs_raw = Image.open(Xs_path)
    _, Xs, _ = models.get_face(Xs_raw)

    # get Yt
    Yt = models.do_swap(Xs, Xt256)
    Yt = models.do_sr(Yt)
    Yt = models.do_eye(Xt1024, Yt)
    Ytt = models.combine(Yt, trans_inv, Xt_raw)
    
    # get Yt
    cv2.imwrite(save_path, Ytt[:, :, ::-1]*255)

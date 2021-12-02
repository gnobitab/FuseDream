import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision
import BigGAN_utils.utils as utils
import clip
import torch.nn.functional as F
from DiffAugment_pytorch import DiffAugment
import numpy as np
import lpips

LATENT_NOISE = 0.01
Z_THRES = 2.0
POLICY = 'color,translation,resize,cutout'
TEST_POLICY = 'color,translation,resize,cutout'
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()

def AugmentLoss(img, clip_model, text, replicate=10, interp_mode='bilinear', policy=POLICY):

    clip_c = clip_model.logit_scale.exp()
    img_aug = DiffAugment(img.repeat(replicate, 1, 1, 1), policy=policy)
    img_aug = (img_aug+1.)/2.
    img_aug = F.interpolate(img_aug, size=224, mode=interp_mode)
    img_aug.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

    logits_per_image, logits_per_text = clip_model(img_aug, text)
    logits_per_image = logits_per_image / clip_c
    concept_loss = (-1.) * logits_per_image 
     
    return concept_loss.mean(dim=0, keepdim=False)

def NaiveSemanticLoss(img, clip_model, text, interp_mode='bilinear'):

    clip_c = clip_model.logit_scale.exp()
    img = (img+1.)/2.
    img = F.interpolate(img, size=224, mode=interp_mode)
    img.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

    logits_per_image, logits_per_text = clip_model(img, text)
    logits_per_image = logits_per_image / clip_c
    concept_loss = (-1.) * logits_per_image 
     
    return concept_loss.mean(dim=0, keepdim=False)

def get_gaussian_mask(size=256):
    x, y = np.meshgrid(np.linspace(-1,1, size), np.linspace(-1,1,size))
    dst = np.sqrt(x*x+y*y)
      
    # Intializing sigma and muu
    sigma = 1
    muu = 0.000
      
    # Calculating Gaussian array
    gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
    
    return gauss

def save_image(img, path, n_per_row=1):
    with torch.no_grad():
        torchvision.utils.save_image(
            torch.from_numpy(img.cpu().numpy()), ##hack, to turn Distribution back to tensor
            path,
            nrow=n_per_row,
            normalize=True,
        )

def get_G(resolution=256):
    if resolution == 256:
        parser = utils.prepare_parser()
        parser = utils.add_sample_parser(parser)
        config = vars(parser.parse_args())

        # See: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/sample_BigGAN_bs256x8.sh.
        config["resolution"] = utils.imsize_dict["I128_hdf5"]
        config["n_classes"] = utils.nclass_dict["I128_hdf5"]
        config["G_activation"] = utils.activation_dict["inplace_relu"]
        config["D_activation"] = utils.activation_dict["inplace_relu"]
        config["G_attn"] = "128"
        config["D_attn"] = "128"
        config["G_ch"] = 96
        config["D_ch"] = 96
        config["hier"] = True
        config["dim_z"] = 140
        config["shared_dim"] = 128
        config["G_shared"] = True
        config = utils.update_config_roots(config)
        config["skip_init"] = True
        config["no_optim"] = True
        config["device"] = "cuda"
        config["resolution"] = 256

        # Set up cudnn.benchmark for free speed.
        torch.backends.cudnn.benchmark = True

        # Import the model.
        model = __import__(config["model"])
        G = model.Generator(**config).to(config["device"])
        utils.count_parameters(G)

        # Load weights.
        weights_path = "./BigGAN_utils/weights/biggan-256.pth"  # Change this.
        G.load_state_dict(torch.load(weights_path), strict=False)
    elif resolution == 512:
        parser = utils.prepare_parser()
        parser = utils.add_sample_parser(parser)
        config = vars(parser.parse_args())

        # See: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/scripts/sample_BigGAN_bs128x8.sh.
        config["resolution"] = 512
        config["n_classes"] = utils.nclass_dict["I128_hdf5"]
        config["G_activation"] = utils.activation_dict["inplace_relu"]
        config["D_activation"] = utils.activation_dict["inplace_relu"]
        config["G_attn"] = "64"
        config["D_attn"] = "64"
        config["G_ch"] = 96
        config["D_ch"] = 64
        config["hier"] = True
        config["dim_z"] = 128
        config["shared_dim"] = 128
        config["G_shared"] = True
        config = utils.update_config_roots(config)
        config["skip_init"] = True
        config["no_optim"] = True
        config["device"] = "cuda"

        # Set up cudnn.benchmark for free speed.
        torch.backends.cudnn.benchmark = True

        # Import the model.
        model = __import__(config["model"])
        #print(config["model"])
        G = model.Generator(**config).to(config["device"])
        utils.count_parameters(G)
        #print('G parameters:')
        #for p, m in G.named_parameters():
        #    print(p)
        # Load weights.
        weights_path = "./BigGAN_utils/weights/biggan-512.pth"  # Change this.
        G.load_state_dict(torch.load(weights_path), strict=False)

    return G, config

class FuseDreamBaseGenerator():
    def __init__(self, G, G_config, G_batch_size=10, clip_mode="ViT-B/32", interp_mode='bilinear'):

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.G = G
        self.clip_model, _ = clip.load(clip_mode, device=device) 
        
        (self.z_, self.y_) = utils.prepare_z_y(
            G_batch_size,
            self.G.dim_z,
            G_config["n_classes"],
            device=G_config["device"],
            fp16=G_config["G_fp16"],
            z_var=G_config["z_var"],
        )

        self.G.eval()

        for p in self.G.parameters():
            p.requires_grad = False
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.interp_mode = interp_mode 
  
    def generate_basis(self, text, init_iters=500, num_basis=5):
        text_tok = clip.tokenize([text]).to(self.device)
        clip_c = self.clip_model.logit_scale.exp() 

        z_init_cllt = []
        y_init_cllt = []
        z_init = None
        y_init = None
        score_init = None
        with torch.no_grad():
            for i in tqdm(range(init_iters)):
                self.z_.sample_()
                self.y_.sample_()

                self.z_.data = torch.clamp(self.z_.data.detach().clone(), min=-Z_THRES, max=Z_THRES)

                image_tensors = self.G(self.z_, self.G.shared(self.y_))
                image_tensors = (image_tensors+1.) / 2.
                image_tensors = F.interpolate(image_tensors, size=224, mode=self.interp_mode)
                image_tensors.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
                
                logits_per_image, logits_per_text = self.clip_model(image_tensors, text_tok)
                logits_per_image = logits_per_image/clip_c
                if z_init is None:
                    z_init = self.z_.data.detach().clone()
                    y_init = self.y_.data.detach().clone()
                    score_init = logits_per_image.squeeze()
                else:
                    z_init = torch.cat([z_init, self.z_.data.detach().clone()], dim=0)
                    y_init = torch.cat([y_init, self.y_.data.detach().clone()], dim=0)
                    score_init = torch.cat([score_init, logits_per_image.squeeze()])

                sorted, indices = torch.sort(score_init, descending=True)
                z_init = z_init[indices]
                y_init = y_init[indices]
                score_init = score_init[indices]
                z_init = z_init[:num_basis]
                y_init = y_init[:num_basis]
                score_init = score_init[:num_basis]
        
        #save_image(self.G(z_init, self.G.shared(y_init)), 'samples/init_%s.png'%text, 1)

        z_init_cllt.append(z_init.detach().clone())
        y_init_cllt.append(self.G.shared(y_init.detach().clone()))

        return z_init_cllt, y_init_cllt


    def optimize_clip_score(self, z_init_cllt, y_init_cllt, text, latent_noise=False, augment=True, opt_iters=500, optimize_y=False):

        text_tok = clip.tokenize([text]).to(self.device)
        clip_c = self.clip_model.logit_scale.exp()

        z_init_ans = torch.stack(z_init_cllt)
        y_init_ans = torch.stack(y_init_cllt)
        z_init_ans = z_init_ans.view(-1, z_init_ans.shape[-1])
        y_init_ans = y_init_ans.view(-1, y_init_ans.shape[-1])

        w_z = torch.randn((z_init_ans.shape[0], z_init_ans.shape[1])).to(self.device)
        w_y = torch.randn((y_init_ans.shape[0], y_init_ans.shape[1])).to(self.device)
        w_z.requires_grad = True
        w_y.requires_grad = True

        opt_y = torch.zeros(y_init_ans.shape).to(self.device)
        opt_y.data = y_init_ans.data.detach().clone()
        opt_z = torch.zeros(z_init_ans.shape).to(self.device)
        opt_z.data = z_init_ans.data.detach().clone()
        opt_z.requires_grad = True
        
        if not optimize_y:
            optimizer = torch.optim.Adam([w_z, w_y, opt_z], lr=5e-3, weight_decay=0.0)
        else:
            opt_y.requires_grad = True
            optimizer = torch.optim.Adam([w_z, w_y,opt_y,opt_z], lr=5e-3, weight_decay=0.0)

        for i in tqdm(range(opt_iters)):
            #print(w_z.shape, w_y.shape)
            optimizer.zero_grad()
            
            if not latent_noise:
                s_z = torch.softmax(w_z, dim=0)
                s_y = torch.softmax(w_y, dim=0)
                #print(s_z)
            
                cur_z = s_z * opt_z
                cur_y = s_y * opt_y
                cur_z = cur_z.sum(dim=0, keepdim=True)
                cur_y = cur_y.sum(dim=0, keepdim=True)

                image_tensors = self.G(cur_z, cur_y)
            else:
                s_z = torch.softmax(w_z, dim=0)
                s_y = torch.softmax(w_y, dim=0)
            
                cur_z = s_z * opt_z
                cur_y = s_y * opt_y
                cur_z = cur_z.sum(dim=0, keepdim=True)
                cur_y = cur_y.sum(dim=0, keepdim=True)
                cur_z_aug = cur_z + torch.randn(cur_z.shape).to(cur_z.device) * LATENT_NOISE
                cur_y_aug = cur_y + torch.randn(cur_y.shape).to(cur_y.device) * LATENT_NOISE
                
                image_tensors = self.G(cur_z_aug, cur_y_aug)
            
            loss = 0.0
            for j in range(image_tensors.shape[0]):
                if augment:
                    loss = loss + AugmentLoss(image_tensors[j:(j+1)], self.clip_model, text_tok, replicate=50, interp_mode=self.interp_mode)
                else:
                    loss = loss + NaiveSemanticLoss(image_tensors[j:(j+1)], self.clip_model, text_tok) 

            loss.backward()
            optimizer.step()

            opt_z.data = torch.clamp(opt_z.data.detach().clone(), min=-Z_THRES, max=Z_THRES)

        z_init_ans = cur_z.detach().clone()
        y_init_ans = cur_y.detach().clone()

        #save_image(self.G(z_init_ans, y_init_ans), 'samples/opt_%s.png'%text, 1)
        return self.G(z_init_ans, y_init_ans), z_init_ans, y_init_ans    

    def measureAugCLIP(self, z, y, text, augment=False, num_samples=20):
        text_tok = clip.tokenize([text]).to(self.device)
        avg_loss = 0.0
        for itr in range(num_samples):
            image_tensors = self.G(z, y)

            for j in range(image_tensors.shape[0]):
                if augment:
                    loss = AugmentLoss(image_tensors[j:(j+1)], self.clip_model, text_tok, replicate=50, interp_mode=self.interp_mode, policy=TEST_POLICY)
                else:
                    loss = NaiveSemanticLoss(image_tensors[j:(j+1)], self.clip_model, text_tok) 
            avg_loss += loss.item()

        avg_loss /= num_samples
        return avg_loss * (-1.)


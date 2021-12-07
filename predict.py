import tempfile
from pathlib import Path
import torch
import cog
import BigGAN_utils.utils as utils
from fusedream_utils import FuseDreamBaseGenerator, save_image


class Predictor(cog.Predictor):
    def setup(self):
        G_512, config_512 = get_G(512)
        generator_512 = FuseDreamBaseGenerator(G_512, config_512, 10)
        G_256, config_256 = get_G(256)
        generator_256 = FuseDreamBaseGenerator(G_256, config_256, 10)
        self.generators = {
            512: generator_512,
            256: generator_256
        }
        pass

    @cog.input("sentence", type=str)
    @cog.input("init_iterations", type=int, default=300, min=100, max=500,
               help='number of iterations for getting the basis images')
    @cog.input("opt_iterations", type=int, default=300, min=100, max=500,
               help='number of iterations for optimizing the latents')
    @cog.input("seed", type=int, default=0, help="0 for random seed")
    @cog.input("model_size", type=int, default=512, options=[512, 256], help="output image size")
    def predict(self, sentence, init_iterations, opt_iterations, seed, model_size):
        generator = self.generators[model_size]
        utils.seed_rng(seed)
        print('Generating:', sentence)

        z_cllt, y_cllt = generator.generate_basis(sentence, init_iters=init_iterations, num_basis=5)
        img, z, y = generator.optimize_clip_score(z_cllt, y_cllt, sentence, latent_noise=True, augment=True,
                                                  opt_iters=opt_iterations, optimize_y=True)
        out_path = Path(tempfile.mkdtemp()) / "out.png"

        save_image(img, str(out_path))
        return out_path


def get_G(resolution=256):
    if resolution == 256:
        parser = utils.prepare_parser()
        parser = utils.add_sample_parser(parser)
        config = vars(parser.parse_args(''))

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
        weights_path = "BigGAN_utils/weights/biggan-256.pth"  # Change this.
        G.load_state_dict(torch.load(weights_path), strict=False)
    elif resolution == 512:
        parser = utils.prepare_parser()
        parser = utils.add_sample_parser(parser)
        config = vars(parser.parse_args(''))

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
        G = model.Generator(**config).to(config["device"])
        utils.count_parameters(G)

        weights_path = "BigGAN_utils/weights/biggan-512.pth"  # Change this.
        G.load_state_dict(torch.load(weights_path), strict=False)

    return G, config

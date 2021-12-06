import tempfile
from pathlib import Path
import cog
import BigGAN_utils.utils as utils
from fusedream_utils import FuseDreamBaseGenerator, get_G, save_image


class Predictor(cog.Predictor):
    def setup(self):
        parser = utils.prepare_parser()
        parser = utils.add_sample_parser(parser)
        self.args = parser.parse_args()
        G_512, config_512 = get_G(512)
        generator_512 = FuseDreamBaseGenerator(G_512, config_512, 10)
        G_256, config_256 = get_G(256)
        generator_256 = FuseDreamBaseGenerator(G_256, config_256, 10)
        self.generators = {
            512: generator_512,
            256: generator_256
        }

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

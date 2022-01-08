import torch
#from torch import nn
from torchvision import utils
from model import Generator
from tqdm import tqdm


import argparse

def generate(args , generator , device , mean_latent):

    with torch.no_grad():
        generator.eval()

        for i in tqdm(range(args.pics)):
            noise = torch.randn(args.sample , args.latent , device=device)

            imgs , latent = generator(
                [noise] , truncation=args.truncation , truncation_latent=mean_latent
            )

            utils.save_image(
                imgs,
                f"sample/{str(i).zfill(6)}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )



if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description="Generator for testing")

    parser.add_argument(
        "--size" , type=int , default=1024 , help='Output Size'
    )

    parser.add_argument(
        "--sample" , 
        type=int , 
        default=1 , 
        help="Number of samples "
    )

    parser.add_argument(
        "--pics" , 
        type=int , 
        default=20 , 
        help="Number of images"
    )

    parser.add_argument(
        "--truncation" , 
        type=float , 
        default=0.5 , 
        help="truncation rate"
    )

    parser.add_argument(
        "--truncation_mean" , 
        type=int , 
        default=4096 , 
        help="Number of Vectors to calculcate mean for truncation"
    )

    parser.add_argument(
        "--ckpt" , 
        type=str , 
        default="stylegan2-ffhq-config-f.pt" , 
        help="path to the model checkpoint"
    )

    parser.add_argument(
        "--channel_multiplier" , 
        type=int , 
        default=2 , 
        help="channels multiplier of the generator"
    )

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    generator = Generator(
        args.size , 
        args.latent , 
        args.n_mlp , 
        args.channel_multiplier
    ).to(device)

    ckpt = torch.load(args.ckpt , map_location=device)
    generator.load_state_dict(ckpt["g_ema"] , strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = generator.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    
    generate(args , generator , device , mean_latent)
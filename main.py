"""
Main programe

NOTE: The "run" in running mode here means we do both train and infer 

@yuningw
"""

import      torch 
from        lib import init
from        lib.runners import vaeRunner, latentRunner
import      argparse


parser = argparse.ArgumentParser()
parser.add_argument('-nn',default="easy", type=str,   help="Choose the model for time-series prediction: easy, self OR lstm")
parser.add_argument('-re',default=40,     type=int,   help="40 OR 100, Choose corresponding Reynolds number for the case")
parser.add_argument('-m', default="test",type=str,   help='Switch the mode between train, infer and run')
parser.add_argument('-t', default="pre", type=str,    help='The type of saved model: pre/val/final')
args  = parser.parse_args()

device = ('cuda' if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":

    ## Env INIT
    datafile = init.init_env(args.re)


    ## Beta-VAE 
    bvae   = vaeRunner(device,datafile)
    if args.m == 'train':
        bvae.train()
    elif args.m == 'test':
        bvae.infer(args.t)
    elif args.m == 'run':
        bvae.run()


    # Time-series prediction runner 
    lruner = latentRunner(args.nn,device)
    if args.m == 'train':
        lruner.train()
    elif args.m == 'test':
        lruner.infer(args.t)
    elif args.m == 'run':
        lruner.run()

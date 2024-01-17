import torch 
from lib.runners import vaeRunner, latentRunner
from lib import init
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-nn',default="easy", type=str,   help="Choose the model for time-series prediction: easy, self OR lstm")
parser.add_argument('-re',default=40,     type=int,   help="40 OR 100, Choose corresponding Reynolds number for the case")
parser.add_argument('-m', default="train",type=str,   help='Switch the mode between train, infer and run')
args  = parser.parse_args()


device = ('cuda' if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    ## Beta-VAE 
    
    datafile = init.init_env(args.re)

    bvae   = vaeRunner(device,datafile)
    if args.m == 'train':
        bvae.train()
    elif args.m == 'test':
        bvae.infer()
    elif args.m == 'run':
        bvae.run()


    # bvae.get_data()
    # bvae.compile()
    # bvae.run()

    ## Time-series prediction runner 
    
    # runner = latentRunner(name,device)
    # print(runner.model.eval)
    # runner.train()
    # runner.post_process()
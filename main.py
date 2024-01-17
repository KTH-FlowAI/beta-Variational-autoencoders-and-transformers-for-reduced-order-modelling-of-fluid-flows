import torch 
from lib.runners import vaeRunner, latentRunner
from lib import init
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-model',default="easy",type=str,help="Choose the model for time-series prediction: easy, self OR lstm")
parser.add_argument('-re',default=40,type=str,help="40 OR 100, Choose corresponding Reynolds number for the case")
args  = parser.parse_args()
# name  = args.model
device = ('cuda' if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    ## Beta-VAE 
    datafile = init.init_env(args.re)
    bvae   = vaeRunner(device,datafile)
    # bvae.get_data()
    # bvae.compile()
    # bvae.run()

    ## Time-series prediction runner 
    
    # runner = latentRunner(name,device)
    # print(runner.model.eval)
    # runner.train()
    # runner.post_process()
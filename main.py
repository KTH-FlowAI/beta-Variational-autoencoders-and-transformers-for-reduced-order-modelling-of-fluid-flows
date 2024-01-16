import torch 
from lib.runners import vaeRunner, latentRunner
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-model',default="easy",type=str,help="Choose the model for time-series prediction: easy, self OR lstm")
args  = parser.parse_args()
name  = args.model
device = ('cuda' if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    ## Beta-VAE 
    
    bvae   = vaeRunner(device)
    bvae.get_data()

    ## Time-series prediction runner 
    
    # runner = latentRunner(name,device)
    # print(runner.model.eval)
    # runner.train()
    # runner.post_process()
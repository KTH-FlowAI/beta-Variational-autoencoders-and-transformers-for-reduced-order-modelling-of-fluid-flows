import torch 
from lib.runner_lantent_predictor import latentRunner
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-model',default="easy",type=str,help="Choose the model for time-series prediction: easy, self OR lstm")
args  = parser.parse_args()
name  = args.model
device = ('cuda' if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    runner = latentRunner(name,device)
    runner.train()
    runner.post_process()
import torch
from pathlib import Path
import argparse
import wandb
import logging
from datetime import datetime
from dataset import get_data_loaders
from model import create_model
from training import TrainingPipeline

def parse_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('--data_dir', type=str, default='../balanced_processed_data')
   parser.add_argument('--output_dir', type=str, default='../outputs')
   parser.add_argument('--batch_size', type=int, default=16)
   parser.add_argument('--epochs', type=int, default=100)
   parser.add_argument('--patience', type=int, default=10)
   parser.add_argument('--learning_rate', type=float, default=1e-4)
   parser.add_argument('--num_workers', type=int, default=4)
   return parser.parse_args()

def setup_experiment(args):
   timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
   experiment_dir = Path(args.output_dir) / timestamp
   experiment_dir.mkdir(parents=True, exist_ok=True)
   
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s',
       handlers=[
           logging.FileHandler(experiment_dir / 'training.log'),
           logging.StreamHandler()
       ]
   )
   return experiment_dir

def main():
   args = parse_args()
   experiment_dir = setup_experiment(args)
   logger = logging.getLogger(__name__)
   device = torch.device('cpu')
   
   logger.info(f"Using device: {device}")
   logger.info("Creating datasets and dataloaders...")
   data_dir = Path(args.data_dir).resolve()
   
   train_loader, val_loader, test_loader = get_data_loaders(
       data_dir=data_dir,
       batch_size=args.batch_size,
       num_workers=args.num_workers
   )
   
   logger.info("Creating model...")
   model, trainer = create_model(
       num_classes=3,
       learning_rate=args.learning_rate,
       device=device
   )
   
   pipeline = TrainingPipeline(
       model=model,
       trainer=trainer,
       train_loader=train_loader,
       val_loader=val_loader,
       test_loader=test_loader,
       config={
           'epochs': args.epochs,
           'patience': args.patience,
           'log_interval': 10,
       },
       output_dir=experiment_dir
   )
   
   try:
       pipeline.train()
       eval_results = pipeline.evaluate()
       logger.info(f"Test Loss: {eval_results['test_loss']:.4f}")
       
   except Exception as e:
       logger.error(f"Error during execution: {str(e)}")
       raise

if __name__ == '__main__':
   main()
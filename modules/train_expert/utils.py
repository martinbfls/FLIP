import torch
import os

def checkpoint_callback(model, opt, epoch, iteration, save_iter, output_dir):
    '''Saves model and optimizer state dicts at fixed intervals.'''
    if iteration % save_iter == 0 and iteration != 0:
        os.makedirs(output_dir, exist_ok=True)

        checkpoint_path = os.path.join(output_dir, f'model_{epoch}_{iteration}.pth')
        opt_path = os.path.join(output_dir, f'model_{epoch}_{iteration}_opt.pth')
        torch.save(model.state_dict(), checkpoint_path)
        torch.save(opt.state_dict(), opt_path)

import argparse
import os
import sys
import warnings
import torch
import torch.multiprocessing as mp
import subprocess
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_list(s):
    if s is None: return []
    if isinstance(s, list): return s
    items = s.split(',')
    return items

def pre_parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    
    parser.add_argument('-lm', '--load_model', type=int, default=0, help='0: New Train, 1: Resume Train')
    parser.add_argument('-v', '--version', type=str, default=1, help='Version number (e.g. 5)')
    parser.add_argument('-ckpt', '--ckpt_num', type=str, default=0, help='Checkpoint number (e.g. 115)')
    
    parser.add_argument('-c', '--config', type=str, default='./config/AMI2GMI.json')
    parser.add_argument('-p', '--phase', type=str, default='train')
    parser.add_argument('-b', '--batch', type=int, default=None)
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-s', '--sampler', type=str, default='ddpm')
    parser.add_argument('-step', '--infer_step', type=int, default=50)
    parser.add_argument('-obj', '--objective', type=str, default='pred_x0', choices=['pred_noise', 'pred_x0'], help='Prediction objective: pred_noise or pred_x0')
    args, _ = parser.parse_known_args()
    return args

args = pre_parse_args()

if args.load_model == 1:
    if args.version is None or args.ckpt_num is None:
        raise ValueError(">>> [Error] If load_model is 1, you must provide --version and --ckpt_num.")
    
    base_exp_dir = f'experiments/train_AMI2GMI_v{args.version}'
    saved_code_path = os.path.join(os.getcwd(), base_exp_dir, 'code')
    
    if os.path.exists(saved_code_path):
        print(f">>> [System] Resuming Mode: Loading backup code from {saved_code_path}")
        sys.path.insert(0, saved_code_path)
        backup_config_path = os.path.join(saved_code_path, 'config', 'AMI2GMI.json')
        
        if os.path.exists(backup_config_path):
            args.config = backup_config_path # 강제로 Config 경로 덮어쓰기
            print(f">>> [System] Config Path Overwritten: {args.config}")
        else:
            print(f">>> [Warning] Backup config not found at {backup_config_path}. Using CURRENT config.")
    else:
        print(f">>> [Warning] Backup code path not found: {saved_code_path}")
        print(">>> [System] Using CURRENT code instead.")

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric


def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend = 'nccl', 
            init_method = opt['init_method'],
            world_size = opt['world_size'], 
            rank = opt['global_rank'],
            group_name='mtorch'
        )
    
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader = define_dataloader(phase_logger, opt)
    networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt = opt,
        networks = networks,
        phase_loader = phase_loader,
        val_loader = val_loader,
        losses = losses,
        metrics = metrics,
        logger = phase_logger,
        writer = phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        phase_writer.close()
        
        
if __name__ == '__main__':
    
    opt = Praser.parse(args)
    if args.load_model == 1:
        base_exp_dir = f'experiments/train_AMI2GMI_v{args.version}'
        ckpt_path = os.path.join(os.getcwd(), base_exp_dir, 'checkpoint', f'{args.ckpt_num}')
        
        opt['path']['resume_state'] = ckpt_path
        print(f">>> [System] Resume State Overwritten: {opt['path']['resume_state']}")
    
    else:
        opt['path']['resume_state'] = None
        print(">>> [System] New Train Mode: 'resume_state' has been cleared.")
    
    opt['sampler'] = args.sampler
    opt['infer_step'] = args.infer_step
    opt['objective'] = args.objective
    
    ''' cuda devices '''
    if opt['gpu_ids'] is not None:
        gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
        print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) '''
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids'])
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)

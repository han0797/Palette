from functools import partial
import numpy as np

from torch.utils.data.distributed import DistributedSampler
from torch import Generator, randperm
from torch.utils.data import DataLoader, Subset

import core.util as Util
from core.praser import init_obj


def define_dataloader(logger, opt):
    """ create train/test dataloader and validation dataloader,  validation dataloader is None when phase is test or not GPU 0 """
    '''create dataset and set random seed'''
    dataloader_args = opt['datasets'][opt['phase']]['dataloader']['args']
    worker_init_fn = partial(Util.set_seed, gl_seed=opt['seed'])

    phase_dataset, val_dataset = define_dataset(logger, opt)

    '''create datasampler'''
    data_sampler = None
    if opt['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=dataloader_args.get('shuffle', False), num_replicas=opt['world_size'], rank=opt['global_rank'])
        dataloader_args.update({'shuffle':False}) # sampler option is mutually exclusive with shuffle 
    
    ''' create dataloader and validation dataloader '''
    dataloader = DataLoader(phase_dataset, sampler=data_sampler, worker_init_fn=worker_init_fn, **dataloader_args)
    ''' val_dataloader don't use DistributedSampler to run only GPU 0! '''
    if opt['global_rank']==0 and val_dataset is not None:
        dataloader_args.update(opt['datasets'][opt['phase']]['dataloader'].get('val_args',{}))
        val_dataloader = DataLoader(val_dataset, worker_init_fn=worker_init_fn, **dataloader_args) 
    else:
        val_dataloader = None
    return dataloader, val_dataloader

def define_dataset(logger, opt):
    """ loading Dataset() class from given file's name """
    # 1. 메인 데이터셋(학습용) 로드
    dataset_opt = opt['datasets'][opt['phase']]['which_dataset']
    phase_dataset = init_obj(dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
    val_dataset = None

    # 2. 별도의 검증(Validation) 데이터셋 로드 
    # [수정] 디버그 모드가 아닐 때만('debug' not in opt['name']) 별도 검증 셋을 로드합니다.
    if opt['phase'] == 'train' and 'val' in opt['datasets'] and 'debug' not in opt['name']:
        val_dataset_opt = opt['datasets']['val']['which_dataset']
        val_dataset = init_obj(val_dataset_opt, logger, default_file_name='data.dataset', init_type='Dataset')
        logger.info('Validation dataset loaded from separate config.')

    # 3. 데이터셋 길이 결정 (전체 길이 vs 디버그 길이)
    data_len = len(phase_dataset)
    if 'debug' in opt['name']:
        debug_split = opt['debug'].get('debug_split', 1.0)
        if isinstance(debug_split, int):
            data_len = debug_split  # 예: 50
        else:
            data_len = int(data_len * debug_split)

    # 4. 데이터 분할 설정
    dataloder_opt = opt['datasets'][opt['phase']]['dataloader']
    valid_split = dataloder_opt.get('validation_split', 0)
    valid_len = 0

    # 5. 데이터 자르기 및 분할 로직
    # Case A: 검증 셋이 이미 따로 로드된 경우 (디버그가 아닐 때)
    if val_dataset is not None:
        # 데이터 길이 자르기 (혹시 필요하다면)
        if data_len < len(phase_dataset):
            phase_dataset = subset_split(dataset=phase_dataset, lengths=[data_len], generator=Generator().manual_seed(opt['seed']))[0]
            logger.info(f'Train dataset truncated to {data_len} samples.')

    # Case B: 검증 셋이 없어서 학습 데이터에서 떼어내야 하는 경우 (디버그 모드 포함)
    elif valid_split > 0.0 or 'debug' in opt['name']: 
        # [수정] 디버그 모드인데 Config상 split이 0이면, 강제로 10%를 검증용으로 할당
        if 'debug' in opt['name'] and valid_split == 0:
            valid_len = max(1, int(data_len * 0.1)) # 최소 1장 이상
        elif isinstance(valid_split, int):
            assert valid_split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = valid_split
        else:
            valid_len = int(data_len * valid_split)
        
        # 학습용 데이터 길이 조정 (전체 - 검증용)
        train_len = data_len - valid_len
        phase_dataset, val_dataset = subset_split(dataset=phase_dataset, lengths=[train_len, valid_len], generator=Generator().manual_seed(opt['seed']))
        
        if 'debug' in opt['name']:
             logger.info(f'Debug Mode: Split {data_len} samples into Train({train_len}) / Val({valid_len}).')

    # 6. 결과 로그 출력
    logger.info('Dataset for {} have {} samples.'.format(opt['phase'], len(phase_dataset)))
    if opt['phase'] == 'train' and val_dataset is not None:
        logger.info('Dataset for {} have {} samples.'.format('val', len(val_dataset)))   
        
    return phase_dataset, val_dataset

def subset_split(dataset, lengths, generator):
    """
    split a dataset into non-overlapping new datasets of given lengths. main code is from random_split function in pytorch
    """
    indices = randperm(sum(lengths), generator=generator).tolist()
    Subsets = []
    for offset, length in zip(np.add.accumulate(lengths), lengths):
        if length == 0:
            Subsets.append(None)
        else:
            Subsets.append(Subset(dataset, indices[offset - length : offset]))
    return Subsets

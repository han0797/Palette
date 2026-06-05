import torch
import tqdm
from core.base_model import BaseModel
from core.logger import LogTracker
import copy
from models.metric import Nor2tbh
import inspect
import torch.optim.lr_scheduler as lr_scheduler

class EMA():
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta
    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)
    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Palette(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, ema_scheduler=None, lr_schedulers=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(Palette, self).__init__(**kwargs)
        
        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.opt['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.opt['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        self.resume_training() 
        
        if self.opt['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
            self.netG.module.set_objective(self.opt.get('objective', 'pred_noise'))
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)
            self.netG.set_objective(self.opt.get('objective', 'pred_noise'))

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        self.best_score = 0 # CSI 기준이라 0
        self.best_epoch = 0
        
        self.schedulers = []
        if lr_schedulers is not None:
            optimizer = self.optimizers[0] 
            for conf in lr_schedulers:
                if conf['name'][1] == 'CosineAnnealingLR':
                    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, **conf['args'])
                    self.schedulers.append(scheduler)
                    self.logger.info(f"Scheduler created: CosineAnnealingLR with {conf['args']}")
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
        
        if 'stats' in data:
            self.stats = {}
            for k, v in data['stats'].items():
                if not torch.is_tensor(v):
                    v = torch.tensor(v)
                
                v = self.set_device(v).float()
                self.stats[k] = v.view(self.batch_size, 1, 1, 1)
        else:
            self.stats = None
    
    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        }
        if self.task in ['inpainting','uncropping']:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                'mask_image': (self.mask_image+1)/2,
            })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}.png'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}.png'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}.png'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
        
        if self.task in ['inpainting','uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in tqdm.tqdm(self.phase_loader):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
            loss.backward()
            self.optG.step()

            self.iter += self.batch_size
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.opt['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals().items():
                    if value.shape[1] > 3:
                        value = value[:, :3, :, :]
                    elif value.shape[1] == 1:
                        value = value.repeat(1, 3, 1, 1)
                    
                    self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def val_step(self, sampler='ddim', infer_step=50):
        self.netG.eval()
        self.val_metrics.reset()
        sampler = self.opt.get('sampler', 'ddim')
        t_step = self.opt.get('infer_step', 50)
        
        with torch.no_grad():
            for val_data in tqdm.tqdm(self.val_loader):
                self.set_input(val_data)
                
                if self.opt['distributed']:
                    netG = self.netG.module
                else:
                    netG = self.netG
                
                if self.task in ['inpainting','uncropping']:
                    self.output, self.visuals = netG.restoration(
                        self.cond_image, y_t=self.cond_image, y_0=self.gt_image, mask=self.mask, 
                        sample_num=self.sample_num, 
                        sampler=sampler, infer_step=t_step
                    )
                else:
                    self.output, self.visuals = netG.restoration(
                        self.cond_image, 
                        sample_num=self.sample_num,
                        sampler=sampler,
                        infer_step=t_step
                    )
                
                self.iter += self.batch_size
                
                # 모델 출력(-1~1) -> 밝기온도(tbh, K)
                gt_tbh = Nor2tbh(self.gt_image)
                pr_tbh = Nor2tbh(self.output)
                
                for met in self.metrics:
                    key = met.__name__
                    # ★ 수정한 metric 함수들에 맞게 mask를 세 번째 인자로 넘겨줌
                    value = met(gt_tbh, pr_tbh, mask=self.mask) 
                    self.val_metrics.update(key, value, n=self.batch_size)

                    # self.val_metrics.update(key, value)
                    # self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='val').items():
                    if value.shape[1] > 3:
                        value = value[:, :3, :, :]
                    elif value.shape[1] == 1:
                        value = value.repeat(1, 3, 1, 1)
                    self.writer.add_images(key, value)
                # self.writer.save_images(self.save_current_results())
        
        # return self.val_metrics.result()
        
        val_results = self.val_metrics.result()
        for key, value in val_results.items():
            self.writer.add_scalar(key, value, self.epoch)
        
        current_score = val_results['val/csi']
        if current_score > self.best_score:
            self.best_score = current_score
            self.best_epoch = self.epoch
            
            if self.opt['distributed']:
                    netG_label = self.netG.module.__class__.__name__
            else:
                netG_label = self.netG.__class__.__name__
            self.save_network(network=self.netG, network_label=f'best_{netG_label}')
            
            self.logger.info(f"★ New Best Model found at Epoch {self.epoch}! CSI: {current_score:.6f}")
            
            # (3) [추가] TensorBoard 'Text' 탭에 기록!
            self.writer.add_text(
                'Best_Model_Info', 
                f"**New Best CSI**: {current_score:.6f} at **Epoch {self.epoch}**", 
                self.epoch
            )
        
        return val_results
        
    
    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in tqdm.tqdm(self.phase_loader):
                self.set_input(phase_data)
                if self.opt['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                        
                self.iter += self.batch_size
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                
                # 모델 출력(-1~1) -> 강우율(mm/h)
                gt_tbh = Nor2tbh(self.gt_image)
                pr_tbh = Nor2tbh(self.output)
                
                for met in self.metrics:
                    key = met.__name__
                    # ★ test 단계에서도 반드시 mask를 넘겨서 결측치를 빼고 채점!
                    value = met(gt_tbh, pr_tbh, mask=self.mask) 
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    if value.shape[1] > 3:
                        value = value[:, :3, :, :]
                    elif value.shape[1] == 1:
                        value = value.repeat(1, 3, 1, 1)
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results())
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.opt['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()

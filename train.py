# This is the main training file we are using
import time
import os
from pathlib import Path
import argparse
from tqdm import tqdm
import random
from PIL import Image
import numpy as np
import cv2
import torch
import torch.cuda.amp as amp
import wandb

from datasets import OrientedDataset
import utils
import api


def get_args():
    """ Command line arguments
    """
    parser = argparse.ArgumentParser()
    # wandb setting
    parser.add_argument('--project',    type=str,  default='fisheye')
    parser.add_argument('--group',      type=str,  default='default')
    parser.add_argument('--wbmode',     type=str,  default='disabled')
    parser.add_argument('--proxy',      action='store_true')
    # model setting
    parser.add_argument('--model',      type=str,  default='rapid_pL1')
    parser.add_argument('--initial',    type=str,  default='') # runs/xxx/yyy.pt
    parser.add_argument('--resume',     type=str,  default='') # run name
    # dataset and loader setting
    parser.add_argument('--dataset',    type=str,  default='coco')
    parser.add_argument('--img_size',   type=int,  default=608, choices=[608, 1024])
    parser.add_argument('--epochs',     type=int,  default=80)
    parser.add_argument('--bs',         type=int,  default=16) # batch_size
    # optimization setting
    parser.add_argument('--lr',         type=float,default=0.001)
    parser.add_argument('--amp',        type=str,  default='true')
    parser.add_argument('--ema',        type=str,  default='true')
    # verbose setting
    parser.add_argument('--skip_0eval', action='store_true')
    # device setting
    parser.add_argument('--device',     type=str,  default='0') # '0', '1', ... or 'cpu'
    parser.add_argument('--workers',    type=int,  default=4)
    parser.add_argument('--fix_seed',   action='store_true')
    return parser.parse_args()

def set_default_args(cfg: argparse.Namespace):
    """ Default cfg/arguments that are not often chenged
    """
    # optimization
    cfg.momentum = 0.9
    cfg.weight_decay = 0.0001
    cfg.accum_bs = 64 # equivalent batch size after gradient accumulation.
    cfg.amp = True if cfg.amp.lower() == 'true' else False
    cfg.ema = True if cfg.ema.lower() == 'true' else False
    # Decrease cfg.accum_bs if you like faster training
    cfg.accum_num = max(1, round(cfg.accum_bs // cfg.bs))
    # Exponential moving averaging (EMA)
    cfg.ema_warmup_epochs = 4
    return cfg


def get_dataset_dirs(name):
    coco_dir = Path('D:/datasets/coco')
    mwr_dir = Path('D:/datasets/fisheye/MW-R')
    habbof_dir = Path('D:/datasets/fisheye/HABBOF')
    cepdof_dir = Path('D:/datasets/fisheye/CEPDOF')

    if name.lower() == 'coco':
        train_img_dir = coco_dir / 'train2017'
        train_jspath = coco_dir / 'annotations/instances_train2017.json'
        val_img_dir = cepdof_dir / 'Lunch3'
        val_jspath = cepdof_dir / 'annotations/Lunch3.json'
    elif name.lower() == 'hbcp':
        raise NotImplementedError()
        videos = ['Meeting1', 'Meeting2', 'Lab2',
                  'Lunch1', 'Lunch2', 'Lunch3', 'Edge_cases', 'IRill', 'Activity']
        # if cfg.high_resolution:
        #     videos += ['All_off', 'IRfilter', 'IRill']
        train_img_dir = [f'../../../COSSY/{s}/' for s in videos]
        train_json = [f'../../../COSSY/annotations/{s}.json' for s in videos]
        val_img_dir = '../../../COSSY/Lab1/'
        val_json = '../../../COSSY/annotations/Lab1.json'
    elif name.lower() == 'hbmw':
        raise NotImplementedError()
        train_img_dir = [
            '../Datasets/fisheye/HABBOF/Meeting1',
            '../Datasets/fisheye/HABBOF/Meeting2',
            '../Datasets/fisheye/HABBOF/Lab2',
            '../Datasets/fisheye/MW-R'
        ]
        train_json = [
            '../Datasets/fisheye/annotations/Meeting1.json',
            '../Datasets/fisheye/annotations/Meeting2.json',
            '../Datasets/fisheye/annotations/Lab2.json',
            '../Datasets/fisheye/annotations/MW-R.json'
        ]
        val_img_dir = '../Datasets/fisheye/HABBOF/Lab1/'
        val_json = '../Datasets/fisheye/annotations/Lab1.json'
    elif name.lower() == 'cpmw':
        raise NotImplementedError()
        videos = ['Lunch1', 'Lunch2', 'Edge_cases', 'IRill', 'Activity',
                  'MW']
        # if cfg.high_resolution:
        #     videos += ['All_off', 'IRfilter']
        train_img_dir = [f'../../../COSSY/{s}/' for s in videos]
        train_json = [f'../../../COSSY/annotations/{s}.json' for s in videos]
        val_img_dir = '../../../COSSY/Lunch3/'
        val_json = '../../../COSSY/annotations/Lunch3.json'
    return train_img_dir, train_jspath, val_img_dir, val_jspath


def get_optimizer(model, cfg):
    # different optimization setting for different layers
    pgb, pgw = [], []
    for k, v in model.named_parameters():
        if ('.bn' in k) or ('.bias' in k): # batchnorm or bias
            pgb.append(v)
        else: # conv weights
            assert '.weight' in k, f'name={k}, shape={v.shape}'
            pgw.append(v)

    parameters = [
        {'params': pgb, 'lr': cfg.lr, 'weight_decay': 0.0},
        {'params': pgw, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay},
    ]
    print('Parameter groups:', [len(pg['params']) for pg in parameters])

    optimizer = torch.optim.SGD(parameters, lr=cfg.lr, momentum=cfg.momentum, 
                                weight_decay=cfg.weight_decay)
    return optimizer


def initialize_wandb(cfg):
    log_parent = Path(f'runs/{cfg.project}')
    if cfg.resume: # resume from an existing experiment
        log_dir = log_parent / cfg.resume
        wb_id = open(log_dir / 'wandb_id.txt', 'r').read()
        print(f'resumming from {log_dir}...')
    else: # new experiment
        _base = f'{cfg.model}_{cfg.dataset}'
        run_name = utils.increment_dir(dir_root=log_parent, name=_base)
        log_dir = log_parent / run_name # wandb logging dir
        print(f'logging to {log_dir}...')
        os.makedirs(log_dir, exist_ok=False)
        wb_id = None

    # initialize wandb
    if cfg.proxy:
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    wbrun = wandb.init(project=cfg.project, group=cfg.group, name=run_name, config=cfg,
                       dir='runs/', resume='allow', id=wb_id, mode=cfg.wbmode)
    cfg = wbrun.config
    cfg.log_dir = log_dir
    cfg.wandb_id = wbrun.id
    with open(log_dir / 'wandb_id.txt', 'w') as f:
        f.write(wbrun.id)
    return wbrun, log_dir


class Evaluator():
    def __init__(self, val_img_dir, val_jspath, img_size):
        self.img_dir = val_img_dir
        self.val_func = utils.CustomEval(val_jspath, iou_method='rle')
        self.img_size = img_size

    def validate_log(self, model):
        print(f'Evaluating {type(model)}...')
        tic = time.time()
        model.eval()
        model_eval = api.Detector(conf_thres=0.01, model=model)
        dts = model_eval.detect_imgSeq(self.val_img_dir, input_size=self.img_size)
        msg = self.val_func.evaluate_dtList(dts, metric='AP')
        # print(msg)
        print(f'\nValidation elapsed time: {time.time()-tic:.1f}s')
        results = {
            'ap50': self.val_func._getAP(0.5),
            'ap75': self.val_func._getAP(0.75),
            'ap': self.val_func._getAP()
        }
        return results


def main():
    # get config
    cfg = get_args()
    cfg = set_default_args(cfg)
    print(cfg, '\n')

    if cfg.fix_seed: # fix random seeds for reproducibility
        utils.set_random_seeds(1)

    # device setting
    if cfg.device == 'cpu':
        device = torch.device('cpu')
        print(f'Using CPU for training... (not recommended)')
    else:
        assert cfg.device.isdigit(), 'device should be a int'
        _id = int(cfg.device)
        assert torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        device = torch.device(f'cuda:{_id}')
        print(f'Using device {_id}:', torch.cuda.get_device_properties(_id))
    print(f'Gradient accmulation: every {cfg.accum_num} backwards() -> one step()')
    print(f'Effective batch size: {cfg.bs} * {cfg.accum_num} = {cfg.accum_bs}', '\n')

    # dataset setting
    print('initialing dataloader...')
    train_img_dir, train_jspath, val_img_dir, val_jspath = get_dataset_dirs(cfg.dataset)
    trainset = OrientedDataset(train_img_dir, train_jspath, cfg.img_size, True,
                                only_person=True, debug_mode=cfg.debug)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.bs, 
                shuffle=True, num_workers=cfg.workers, pin_memory=True, drop_last=False)
    # validation set
    evaluator = Evaluator(val_img_dir, val_jspath, cfg.img_size)

    if cfg.model == 'rapid_pL1':
        from models.rapid import RAPiD
        model = RAPiD(backbone='dark53', img_norm=False,
                       loss_angle='period_L1')
    elif cfg.model == 'rapid_pL2':
        from models.rapid import RAPiD
        model = RAPiD(backbone='dark53', img_norm=False,
                       loss_angle='period_L2')
    else: raise NotImplementedError()
    model = model.to(device=device)

    optimizer = get_optimizer()

    # AMP
    scaler = amp.GradScaler(enabled=cfg.amp)

    # wandb
    wbrun, log_dir = initialize_wandb(cfg)
    if cfg.resume: # resume run
        assert not cfg.initial
        checkpoint = torch.load(log_dir / 'last.pt')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start_epoch = checkpoint['epoch'] + 1
        best_fitness = checkpoint['ap50']
    else:
        start_epoch, best_fitness = 0, 0.0
        if cfg.initial: # initialize network from weights
            checkpoint = torch.load(cfg.initial)
            model.load_state_dict(checkpoint['model'], strict=False)

    # ======================== start training ========================
    pbar_title = ('%-8s' * 9) % (
        'Epoch', 'GPU_mem', 'lr', 'xy', 'wh', 'angle','conf', 'loss', 'ap50'
    )
    n_iter = start_epoch * len(trainloader)
    print('Start training...\n')
    for n_epoch in range(start_epoch, cfg.epochs):
        # evaluation and save checkpoint
        if not (cfg.skip_0eval and (n_epoch == 0)):
            _log_dic = {'general/epoch': n_epoch}
            results = evaluator.validate_log(model)
            _log_dic.update({'metric/plain_val_'+k: v for k,v in results.items()})
            if cfg.ema:
                results = results = evaluator.validate_log(ema.ema)
                _log_dic.update({f'metric/ema_val_'+k: v for k,v in results.items()})
            model.train()

        train_loss, train_acc = 0.0, 0.0
        time.sleep(0.1)
        print('\n' + pbar_title) # title
        pbar = tqdm(enumerate(trainloader), total=len(trainloader))
        for bi, (imgs, targets) in pbar:
            n_iter = n_epoch * len(trainloader) + bi

        # subdivision loop
        optimizer.zero_grad()
        for inner_iter_i in range(subdivision):
            try:
                imgs, targets, cats, _, _ = next(dataiterator)  # load a batch
            except StopIteration:
                dataiterator = iter(dataloader)
                imgs, targets, cats, _, _ = next(dataiterator)  # load a batch
            # visualization.imshow_tensor(imgs)
            imgs = imgs.cuda()
            loss = model(imgs, targets, labels_cats=cats)
            loss.backward()
        optimizer.step()
        scheduler.step()

        # logging
        if iter_i % cfg.print_interval == 0:
            sec_used = timer.tic() - start_time
            time_used = timer.sec2str(sec_used)
            avg_iter = timer.sec2str(sec_used/(iter_i+1-start_iter))
            avg_epoch = avg_iter / batch_size / subdivision * 118287
            print(f'\nTotal time: {time_used}, iter: {avg_iter}, epoch: {avg_epoch}')
            current_lr = scheduler.get_last_lr()[0] * batch_size * subdivision
            print(f'[Iteration {iter_i}] [learning rate {current_lr:.3g}]',
                  f'[Total loss {loss:.2f}] [img size {dataset.img_size}]')
            print(model.loss_str)
            max_cuda = torch.cuda.max_memory_allocated(0) / 1024 / 1024 / 1024
            print(f'Max GPU memory usage: {max_cuda} GigaBytes')
            torch.cuda.reset_peak_memory_stats(0)

        # random resizing
        if multiscale and iter_i > 0 and (iter_i % multiscale_interval == 0):
            if cfg.high_resolution:
                imgsize = random.randint(16, 34) * 32
            else:
                low = 10 if cfg.dataset == 'COCO' else 16
                imgsize = random.randint(low, 21) * 32
            dataset.img_size = imgsize
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_cpu, pin_memory=True, drop_last=False)
            dataiterator = iter(dataloader)

        # save checkpoint
        if iter_i > 0 and (iter_i % cfg.checkpoint_interval == 0):
            state_dict = {
                'iter': iter_i,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_path = os.path.join('./weights', f'{job_name}_{today}_{iter_i}.ckpt')
            torch.save(state_dict, save_path)

        # save detection
        if iter_i > 0 and iter_i % cfg.img_interval == 0:
            for img_path in eval_img_paths:
                eval_img = Image.open(img_path)
                dts = api.detect_once(model, eval_img, conf_thres=0.1, input_size=target_size)
                np_img = np.array(eval_img)
                visualization.draw_dt_on_np(np_img, dts)
                np_img = cv2.resize(np_img, (416,416))
                # cv2.imwrite(f'./results/eval_imgs/{job_name}_{today}_{iter_i}.jpg', np_img)
                logger.add_image(img_path, np_img, iter_i, dataformats='HWC')

            model.train()


if __name__ == '__main__':
    main()

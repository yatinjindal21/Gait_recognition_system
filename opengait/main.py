import os
import argparse
import torch
import torch.nn as nn
from modeling import models
from utils import config_loader, init_seeds, params_count, get_msg_mgr

parser = argparse.ArgumentParser(description='Main program for opengait.')
parser.add_argument('--cfgs', type=str,
                    default='configs/default.yaml', help="path of config file")
parser.add_argument('--phase', default='train',
                    choices=['train', 'test'], help="choose train or test phase")
parser.add_argument('--log_to_file', action='store_true',
                    help="log to file, default path is: output/<dataset>/<model>/<save_name>/<logs>/<Datetime>.txt")
parser.add_argument('--iter', default=0, help="iter to restore")
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialization(cfgs, training):
    msg_mgr = get_msg_mgr()
    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    output_path = os.path.join('output/', cfgs['data_cfg']['dataset_name'],
                               cfgs['model_cfg']['model'], engine_cfg['save_name'])
    if training:
        msg_mgr.init_manager(output_path, opt.log_to_file, engine_cfg['log_iter'],
                             engine_cfg['restore_hint'] if isinstance(engine_cfg['restore_hint'], (int)) else 0)
    else:
        msg_mgr.init_logger(output_path, opt.log_to_file)

    msg_mgr.log_info(engine_cfg)

    seed = 42
    init_seeds(seed)


def run_model(cfgs, training):
    msg_mgr = get_msg_mgr()
    model_cfg = cfgs['model_cfg']
    msg_mgr.log_info(model_cfg)
    Model = getattr(models, model_cfg['model'])
    model = Model(cfgs, training)
    if training and cfgs['trainer_cfg']['sync_BN'] and torch.cuda.device_count() > 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if cfgs['trainer_cfg']['fix_BN']:
        model.fix_BN()
    model = model.to(device)
    msg_mgr.log_info(params_count(model))
    msg_mgr.log_info("Model Initialization Finished!")

    engine_cfg = cfgs['trainer_cfg'] if training else cfgs['evaluator_cfg']
    restore_hint = engine_cfg.get("restore_hint", 0)
    if restore_hint:
        model.resume_ckpt(restore_hint)

    if training:
        Model.run_train(model)
    else:
        Model.run_test(model)


if __name__ == '__main__':
    cfgs = config_loader(opt.cfgs)
    if opt.iter != 0:
        cfgs['evaluator_cfg']['restore_hint'] = int(opt.iter)
        cfgs['trainer_cfg']['restore_hint'] = int(opt.iter)

    training = (opt.phase == 'train')
    initialization(cfgs, training)
    run_model(cfgs, training)
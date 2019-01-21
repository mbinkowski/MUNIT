"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
from utils import get_all_data_loaders, prepare_sub_folder, write_html, write_loss, get_config, write_2images, Timer
import argparse
from torch.autograd import Variable
from trainer import MUNITDD_Trainer, MUNIT_Trainer, UNIT_Trainer
import torch.backends.cudnn as cudnn
import torch
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
import os
import sys
import tensorboardX
import shutil
import time
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/edges2handbags_folder.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--requeue", type=float, default=-1)
parser.add_argument('--trainer', type=str, default='MUNIT', help="MUNIT|UNIT")
opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']
display_size = config['display_size']

# Setup model and data loader
if opts.trainer == 'MUNIT':
    trainer = MUNIT_Trainer(config)
elif opts.trainer == 'UNIT':
    trainer = UNIT_Trainer(config)
elif opts.trainer == 'MUNITDD':
    trainer = MUNITDD_Trainer(config)
else:
    sys.exit("Only support MUNITDD|MUNIT|UNIT")
trainer.cuda()
train_loader_a, train_loader_b, test_loader_a, test_loader_b = loaders = get_all_data_loaders(config)
print('Dataset sizes: train A: %d, train B: %d, test A: %d, test B: %d' % (len(train_loader_a),
    len(train_loader_b), len(test_loader_a), len(test_loader_b)))
def sample4display(loader, n):
    samples = [loader.dataset[i % n] for i in range(n)]
    if hasattr(samples[0], '__len__'):
        samples = [s[0] for s in samples]
    return torch.stack(samples).cuda()
train_display_images_a = sample4display(train_loader_a, display_size)
train_display_images_b = sample4display(train_loader_b, display_size)
test_display_images_a = sample4display(test_loader_a, display_size)
test_display_images_b = sample4display(test_loader_b, display_size)

# Setup logger and output folders
def new_sample_path(path, name):
    name += '_' + time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime(time.time()))
    while os.path.exists(os.path.join(path, name)):
        name += np.random.choice(list(string.uppercase))
    return name

success = False
#def check_folders(path, name, config):
basename = os.path.splitext(os.path.basename(opts.config))[0]
folders = glob(os.path.join(opts.output_path, "logs", basename + '*'))
skip = ['image_save_iter', 'image_display_iter', 'display_size', 'snapshot_save_iter', 'log_iter', 'resume', 'max_iter']
config_ = dict([(k,v) for k, v in config.items() if k not in skip])
for f in folders:
    f_config = os.path.join(f.replace('/logs/', '/outputs/'), 'config.yaml')
    if not os.path.exists(f_config):
        print('Config ' + repr(f_config) + ' does not exist')
        continue
    config2_ = dict([(k, v) for k, v in get_config(f_config).items() if k not in skip])
    if config2_ != config_:
        print('Config ' + repr(f_config) + ' does not match')
#        for k,v in config_2.items():
#            if k not in config:
#                print('key %s does not exist in current config')
#            elif v != config[k]:
#                print(k, ': ', v, config[k])
#        print('A\B', [k for k in config if k not in config_2])
        continue
    if len(os.listdir(f)) == 0:
        print('No checkpoint saved in dir ' + repr(f))
        continue
    try:
        model_name = f.split('/')[-1]
        print(model_name + ' seems good')
        train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
        output_directory = os.path.join(opts.output_path + "/outputs", model_name)
        checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
        print('Resuming ...')
        iterations = trainer.resume(checkpoint_directory, hyperparameters=config) if config['resume'] else 0
        success = True
        print('Model loaded at iteration %d' % iterations)
        break    
    except Exception as e:
        print('Load failed for %s' % model_name)
        print(e)

if not success:
    model_name = new_sample_path(opts.output_path + "/logs", basename)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder
    iterations = 0

# Start training
config['vgg_model_path'] = opts.output_path
if 'd_steps' not in config:
    config['d_steps'] = 1
timer = Timer("Elapsed time in update: %f", config['log_iter'], iterations) 
trainer.logger = timer
while True:
    for it, (images_a, images_b) in enumerate(zip(train_loader_a, train_loader_b)):
        if hasattr(images_a, '__len__'):
            images_a, labels_a = images_a
            images_b, labels_b = images_b
#    it_a, it_b = iter(train_loader_a), iter(train_loader_b)
#    for it in range(max(len(it_a), len(it_b))):
        t0 = time.time()
#        images_a, images_b = next(it_a), next(it_b)
        trainer.update_learning_rate()
        images_a, images_b = images_a.cuda().detach(), images_b.cuda().detach()

        with timer:
            # Main training code
            trainer.dis_update(images_a, images_b, config, iterations)
            if iterations % config['d_steps'] == 0:
                trainer.gen_update(images_a, images_b, config, iterations)
            torch.cuda.synchronize()

            # Dump training stats in log file
            if (iterations + 1) % config['log_iter'] == 0:
                timer.print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
                write_loss(iterations, trainer, train_writer)

        # Write images
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample_many(test_display_images_a, test_display_images_b)
                train_image_outputs = trainer.sample_many(train_display_images_a, train_display_images_b)
            write_2images(test_image_outputs, display_size, image_directory, 'test_%08d' % (iterations + 1))
            write_2images(train_image_outputs, display_size, image_directory, 'train_%08d' % (iterations + 1))
            # HTML
            write_html(output_directory + "/index.html", iterations + 1, config['image_save_iter'], 'images')

        if (iterations + 1) % config['image_display_iter'] == 0:
            with torch.no_grad():
                image_outputs = trainer.sample(train_display_images_a, train_display_images_b)
            write_2images(image_outputs, display_size, image_directory, 'train_current')

        # Save network weights
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)
            if (opts.requeue > 0) and (timer.clock() > opts.requeue):
                timer.print('Requeing job after iteration %d.' % (iterations + 1))
                try:
                    subprocess.call('scontrol requeue $SLURM_JOB_ID', shell=True)
                except Exception as e:
                    timer.print('[!] Requeueing failed')
                    print(e)

        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')


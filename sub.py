from itertools import product
import argparse
import subprocess
import os

def parse(f):
    lines = f.split('\n')
    new_files, file_names = [[]], ['']
    for line in lines:
        if ',' in line:
            line_split = line.split(' ')
            n = 0
            for word in line_split:
                if ':' in word:
                    break
                n += 1
            name, vals = line_split[n][:-1], line_split[n + 1]
            new_files_, file_names_ = [], []
            for val in vals.split(','):
                new_files_ += [f + [line.replace(vals, val)] for f in new_files]
                file_names_ += [f + '_' + name[:6] + '=' + val[:6] for f in file_names]
            new_files, file_names = new_files_, file_names_
        else:
            for f in new_files:
                f.append(line)
    return ['\n'.join(f) for f in new_files], file_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('script', type=str)
    parser.add_argument('-c', type=str, default='sbatch --gres=gpu:1 --mem=8192M --cpus-per-task=4 --account=def-bengioy --requeue')
    parser.add_argument('-t', type=str, default='24:00:00')
    parser.add_argument('--trainer', type=str, default='MUNITDD')
    
    config = parser.parse_args()
    
    with open(config.script, 'r') as f:
        main_script = f.read()

    scripts, names = parse(main_script)

    for name, script in zip(names, scripts):
        yaml = os.path.join(os.getcwd(), config.script.replace('.yaml', '%s.yaml' % name))
        yaml_name = config.script.replace('.yaml', '').split('/')[-1]
        runsh = os.path.join(os.getcwd(), 'run_%s%s.sh' % (yaml_name, name))
        print(yaml)
        with open(yaml, 'w') as f:
            f.write(script)

        runcommand = '\n'.join([
            '#!/bin/bash',
            'module use ~/projects/rpp-bengioy/modules/*/Core',
            'source activate munit',
            'python train.py --config %s --trainer=%s --output_path %s'
        ]) % (yaml, config.trainer, os.getcwd())

        with open(runsh, 'w') as f:
            f.write(runcommand)

        bash = '%s -t %s %s' % (config.c, config.t, runsh)
        command = os.popen(bash)
        command.read()
        command.close()
        #subprocess.call(bash)

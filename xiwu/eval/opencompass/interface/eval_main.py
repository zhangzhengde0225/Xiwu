


import hepai
from dataclasses import dataclass, field
import argparse
import copy
import getpass
import os
import os.path as osp
from datetime import datetime

from .base_args import GeneralArgs

def main(args: GeneralArgs):
    from mmengine.config import Config, DictAction
    from opencompass.registry import PARTITIONERS, RUNNERS, build_from_cfg
    from opencompass.runners import SlurmRunner
    from opencompass.summarizers import DefaultSummarizer
    from opencompass.utils import LarkReporter, get_logger
    from opencompass.utils.run import (fill_eval_cfg, fill_infer_cfg,
                                    get_config_from_arg)
   

    if args.dry_run:
        args.debug = True
    # initialize logger
    logger = get_logger(log_level='DEBUG' if args.debug else 'INFO')

    cfg = get_config_from_arg(args)
    if args.work_dir is not None:
        cfg['work_dir'] = args.work_dir
    else:
        cfg.setdefault('work_dir', os.path.join('outputs', 'default'))

    # cfg_time_str defaults to the current time
    cfg_time_str = dir_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.reuse:
        if args.reuse == 'latest':
            if not os.path.exists(cfg.work_dir) or not os.listdir(
                    cfg.work_dir):
                logger.warning('No previous results to reuse!')
            else:
                dirs = os.listdir(cfg.work_dir)
                dir_time_str = sorted(dirs)[-1]
        else:
            dir_time_str = args.reuse
        logger.info(f'Reusing experiements from {dir_time_str}')
    elif args.mode in ['eval', 'viz']:
        raise ValueError('You must specify -r or --reuse when running in eval '
                         'or viz mode!')

    # update "actual" work_dir
    cfg['work_dir'] = osp.join(cfg.work_dir, dir_time_str)
    current_workdir = cfg['work_dir']
    logger.info(f'Current exp folder: {current_workdir}')

    os.makedirs(osp.join(cfg.work_dir, 'configs'), exist_ok=True)

    # dump config
    output_config_path = osp.join(cfg.work_dir, 'configs',
                                  f'{cfg_time_str}_{os.getpid()}.py')
    cfg.dump(output_config_path)
    # Config is intentally reloaded here to avoid initialized
    # types cannot be serialized
    cfg = Config.fromfile(output_config_path, format_python_code=False)

    # report to lark bot if specify --lark
    if not args.lark:
        cfg['lark_bot_url'] = None
    elif cfg.get('lark_bot_url', None):
        content = f'{getpass.getuser()}\'s task has been launched!'
        LarkReporter(cfg['lark_bot_url']).post(content)

    if args.mode in ['all', 'infer']:
        # When user have specified --slurm or --dlc, or have not set
        # "infer" in config, we will provide a default configuration
        # for infer
        if (args.dlc or args.slurm) and cfg.get('infer', None):
            logger.warning('You have set "infer" in the config, but '
                           'also specified --slurm or --dlc. '
                           'The "infer" configuration will be overridden by '
                           'your runtime arguments.')

        if args.dlc or args.slurm or cfg.get('infer', None) is None:
            fill_infer_cfg(cfg, args)

        if args.partition is not None:
            if RUNNERS.get(cfg.infer.runner.type) == SlurmRunner:
                cfg.infer.runner.partition = args.partition
                cfg.infer.runner.quotatype = args.quotatype
        else:
            logger.warning('SlurmRunner is not used, so the partition '
                           'argument is ignored.')
        if args.debug:
            cfg.infer.runner.debug = True
        if args.lark:
            cfg.infer.runner.lark_bot_url = cfg['lark_bot_url']
        cfg.infer.partitioner['out_dir'] = osp.join(cfg['work_dir'],
                                                    'predictions/')
        partitioner = PARTITIONERS.build(cfg.infer.partitioner)
        tasks = partitioner(cfg)
        if args.dry_run:
            return
        runner = RUNNERS.build(cfg.infer.runner)
        # Add extra attack config if exists
        if hasattr(cfg, 'attack'):
            for task in tasks:
                cfg.attack.dataset = task.datasets[0][0].abbr
                task.attack = cfg.attack
        runner(tasks)

    # evaluate
    if args.mode in ['all', 'eval']:
        # When user have specified --slurm or --dlc, or have not set
        # "eval" in config, we will provide a default configuration
        # for eval
        if (args.dlc or args.slurm) and cfg.get('eval', None):
            logger.warning('You have set "eval" in the config, but '
                           'also specified --slurm or --dlc. '
                           'The "eval" configuration will be overridden by '
                           'your runtime arguments.')

        if args.dlc or args.slurm or cfg.get('eval', None) is None:
            fill_eval_cfg(cfg, args)
        if args.dump_eval_details:
            cfg.eval.runner.task.dump_details = True
        if args.dump_extract_rate:
            cfg.eval.runner.task.cal_extract_rate = True
        if args.partition is not None:
            if RUNNERS.get(cfg.eval.runner.type) == SlurmRunner:
                cfg.eval.runner.partition = args.partition
                cfg.eval.runner.quotatype = args.quotatype
            else:
                logger.warning('SlurmRunner is not used, so the partition '
                               'argument is ignored.')
        if args.debug:
            cfg.eval.runner.debug = True
        if args.lark:
            cfg.eval.runner.lark_bot_url = cfg['lark_bot_url']
        cfg.eval.partitioner['out_dir'] = osp.join(cfg['work_dir'], 'results/')
        partitioner = PARTITIONERS.build(cfg.eval.partitioner)
        tasks = partitioner(cfg)
        if args.dry_run:
            return
        runner = RUNNERS.build(cfg.eval.runner)

        # For meta-review-judge in subjective evaluation
        if isinstance(tasks, list) and len(tasks) != 0 and isinstance(
                tasks[0], list):
            for task_part in tasks:
                runner(task_part)
        else:
            runner(tasks)

    # visualize
    if args.mode in ['all', 'eval', 'viz']:
        summarizer_cfg = cfg.get('summarizer', {})

        # For subjective summarizer
        if summarizer_cfg.get('function', None):
            main_summarizer_cfg = copy.deepcopy(summarizer_cfg)
            grouped_datasets = {}
            for dataset in cfg.datasets:
                prefix = dataset['abbr'].split('_')[0]
                if prefix not in grouped_datasets:
                    grouped_datasets[prefix] = []
                grouped_datasets[prefix].append(dataset)
            all_grouped_lists = []
            for prefix in grouped_datasets:
                all_grouped_lists.append(grouped_datasets[prefix])
            dataset_score_container = []
            for dataset in all_grouped_lists:
                temp_cfg = copy.deepcopy(cfg)
                temp_cfg.datasets = dataset
                summarizer_cfg = dict(type=dataset[0]['summarizer']['type'], config=temp_cfg)
                summarizer = build_from_cfg(summarizer_cfg)
                dataset_score = summarizer.summarize(time_str=cfg_time_str)
                if dataset_score:
                    dataset_score_container.append(dataset_score)
            main_summarizer_cfg['config'] = cfg
            main_summarizer = build_from_cfg(main_summarizer_cfg)
            main_summarizer.summarize(time_str=cfg_time_str, subjective_scores=dataset_score_container)
        else:
            if not summarizer_cfg or summarizer_cfg.get('type', None) is None:
                summarizer_cfg['type'] = DefaultSummarizer
            summarizer_cfg['config'] = cfg
            summarizer = build_from_cfg(summarizer_cfg)
            summarizer.summarize(time_str=cfg_time_str)


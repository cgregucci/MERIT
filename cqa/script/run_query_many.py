import os
import sys
import time
import random
import pprint
import argparse

import torch
import torch_geometric as pyg
from torch.utils import data as torch_data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import datasets_query, tasks, util, query_utils
from timeit import default_timer as timer
from script.run_query import train_and_validate, test, build_query_model

separator = ">" * 30
line = "-" * 30

def set_seed(seed):
    random.seed(seed + util.get_rank())
    torch.manual_seed(seed + util.get_rank())
    torch.cuda.manual_seed(seed + util.get_rank())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    seeds = [1024, 42, 1337, 512, 256]

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-d", "--datasets", help="target datasets", default='InductiveFB15k237Query:550,InductiveFB15k237Query:300', type=str, required=True)
    parser.add_argument("-reps", "--repeats", help="number of times to repeat each exp", default=1, type=int)
    parser.add_argument("-ft", "--finetune", help="finetune the checkpoint on the specified datasets", action='store_true')
    parser.add_argument("-s", "--seed", help="override the base seed, which also seeds the RNI noise at evaluation", default=None, type=int)
    args, unparsed = parser.parse_known_args()
    # the dataset loop os.chdir()s into the output working dir after the first dataset, so a
    # relative config path breaks on every subsequent dataset in the loop. Resolve
    # it up front so multi-dataset sweeps work regardless of cwd.
    args.config = os.path.abspath(args.config)

    datasets = args.datasets.split(",")
    results_file = None

    for graph in datasets:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        for i in range(args.repeats):
            seed = args.seed if args.seed is not None else (seeds[i] if i < len(seeds) else random.randint(0, 10000))
            print(f"Running on {graph}, iteration {i+1} / {args.repeats}, seed: {seed}")

            # get dynamic arguments defined in the config file
            vars = util.detect_variables(args.config)
            parser = argparse.ArgumentParser()
            for var in vars:
                parser.add_argument("--%s" % var)
            vars = parser.parse_known_args(unparsed)[0]
            vars = {k: util.literal_eval(v) for k, v in vars._get_kwargs()}

            if args.finetune:
                epochs, batch_per_epoch = 1, 1000
            else:
                epochs, batch_per_epoch = 0, 'null'
            vars['epochs'] = epochs
            vars['bpe'] = batch_per_epoch
            vars['dataset'] = ds
            if version is not None:
                vars['version'] = version

            cfg = util.load_config(args.config, context=vars)
            cfg.eval_seed = seed  # eval RNI seed
            root_dir = os.path.expanduser(cfg.output_dir) # resetting the path to avoid inf nesting
            os.makedirs(root_dir, exist_ok=True)
            if results_file is None:
                results_file = os.path.join(root_dir, f"results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")
            os.chdir(root_dir)
            working_dir = util.create_working_directory(cfg)
            set_seed(seed)

            logger = util.get_root_logger()
            if util.get_rank() == 0:
                logger.warning("Random seed: %d" % seed)
                logger.warning("Config file: %s" % args.config)
                logger.warning(pprint.pformat(cfg))
            
            task_name = cfg.task["name"]
            dataset = query_utils.build_query_dataset(cfg)
            device = util.get_device(cfg)
            
            
            train_data, valid_data, test_data = dataset.split()
            train_graph, valid_graph, test_graph = dataset.train_graph, dataset.valid_graph, dataset.test_graph

            model = build_query_model(cfg)

            if "fast_test" in cfg.train:
                if util.get_rank() == 0:
                    logger.warning("Quick test mode on. Only evaluate on %d samples for valid" % cfg.train.fast_test)
                g = torch.Generator()
                g.manual_seed(1024)
                valid_data = torch_data.random_split(valid_data, [cfg.train.fast_test, len(valid_data) - cfg.train.fast_test], generator=g)[0]
                

            model = model.to(device)
            
            train_and_validate(cfg, model, train_graph, train_data, valid_graph, valid_data, query_id2type=dataset.id2type, device=device, batch_per_epoch=cfg.train.batch_per_epoch, logger=logger)
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Evaluate on valid")
            start = timer()
            val_metrics = test(cfg, model, valid_graph, valid_data, query_id2type=dataset.id2type, device=device, logger=logger)
            end = timer()
            logger.warning(f"Valid time: {end - start}")
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Evaluate on test")
            metrics = test(cfg, model, test_graph, test_data, query_id2type=dataset.id2type, device=device, logger=logger)
            
            # write to the log file
            if util.get_rank() == 0:
                metrics['dataset'] = str(dataset)
                query_utils.print_metrics_to_file(metrics, results_file)

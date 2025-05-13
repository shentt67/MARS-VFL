import argparse
import logging

import pickle
import torch.nn as nn
from torch.nn.utils import clip_grad_norm

from dataset.dataset import *
from method.trainer_efficiency import Trainer_Efficiency
from method.trainer_robustness import Trainer_Robustness
from method.trainer_security import Trainer_Security

from utils.utils import *
from dataset.utils.utils import _process_emo, _process_none, _process_emo_robustness, _process_none_robustness, SubsetRandomSampler
import functools

from model.NUSWIDE_models import GlobalModelForNUSWIDE, LocalModelForNUSWIDE
from model.MOSI_models import GlobalModelForMOSI, LocalModelForMOSI
from model.URFUNNY_models import GlobalModelForURFUNNY, LocalModelForURFUNNY
from model.MUSTARD_models import GlobalModelForMUSTARD, LocalModelForMUSTARD
from model.MOSEI_models import GlobalModelForMOSEI, LocalModelForMOSEI
from model.MIMIC_models import GlobalModelForMIMIC, LocalModelForMIMIC
from model.MUJOCO_models import GlobalModelForMUJOCO, LocalModelForMUJOCO
from model.VISIONTOUCH_models import GlobalModelForVISIONTOUCH, LocalModelForVISIONTOUCH
from model.UCIHAR_models import GlobalModelForUCIHAR, LocalModelForUCIHAR
from model.MMIMDB_models import GlobalModelForMMIMDB, LocalModelForMMIMDB
from model.KUHAR_models import GlobalModelForKUHAR, LocalModelForKUHAR
from model.PTBXL_models import GlobalModelForPTBXL, LocalModelForPTBXL

def main(args):
    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")

    # create logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
    logger.setLevel(level=logging.DEBUG)

    # for seed in args.seeds:
    for seed in [100]:
        set_seed(seed)
        args.seed = seed
        current_time = datetime.now()
        # create handler for writing logs
        args.results_dir = '/data/vfl_benchmark_log/' + args.dataset + '/' + str(args.method_name) + '_seed_' + str(seed) + '_' + str(current_time)
        if not os.path.isdir(args.results_dir):
            os.mkdir(args.results_dir)
        fh = logging.FileHandler(args.results_dir + '/experiment.log')

        # add handler
        logger.addHandler(fh)

        # params
        logger.info(args)

        dataset = {'MOSI': MOSI_VFL, 'MOSEI': MOSEI_VFL, 'URFUNNY': URFUNNY_VFL, 'MUSTARD': MUSTARD_VFL, 'MIMIC': MIMIC_VFL,
                   'PTBXL': PTBXL_VFL, 'MUJOCO': MUJOCO_VFL, 'MMIMDB': MMIMDB_VFL, 'VISIONTOUCH': VISIONTOUCH_VFL, 'NUSWIDE': NUSWIDE_VFL,
                   'UCIHAR': UCIHAR_VFL, 'KUHAR': KUHAR_VFL}

        # create dataset
        logger.info("=> Preparing Data...")
        if args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD', 'MIMIC', 'PTBXL', 'MUJOCO', 'MMIMDB']:
            train_data = dataset[args.dataset](args, dtype='train')
            valid_data = dataset[args.dataset](args, dtype='valid')
            test_data = dataset[args.dataset](args, dtype='test')
        elif args.dataset in ['VISIONTOUCH', 'NUSWIDE', 'UCIHAR', 'KUHAR']:
            train_data = dataset[args.dataset](args, dtype='train')
            valid_data = dataset[args.dataset](args, dtype='valid')
            test_data = None
        else:
            raise_dataset_exception()

        # build vfl models
        global_models = {'MOSI': GlobalModelForMOSI, 'MOSEI': GlobalModelForMOSEI, 'URFUNNY': GlobalModelForURFUNNY, 'MUSTARD': GlobalModelForMUSTARD,
                         'MIMIC': GlobalModelForMIMIC, 'PTBXL': GlobalModelForPTBXL, 'MUJOCO': GlobalModelForMUJOCO, 'MMIMDB': GlobalModelForMMIMDB,
                         'VISIONTOUCH': GlobalModelForVISIONTOUCH, 'NUSWIDE': GlobalModelForNUSWIDE, 'UCIHAR': GlobalModelForUCIHAR, 'KUHAR': GlobalModelForKUHAR}
        local_models = {'MOSI': LocalModelForMOSI, 'MOSEI': LocalModelForMOSEI, 'URFUNNY': LocalModelForURFUNNY, 'MUSTARD': LocalModelForMUSTARD,
                         'MIMIC': LocalModelForMIMIC, 'PTBXL': LocalModelForPTBXL, 'MUJOCO': LocalModelForMUJOCO, 'MMIMDB': LocalModelForMMIMDB,
                         'VISIONTOUCH': LocalModelForVISIONTOUCH, 'NUSWIDE': LocalModelForNUSWIDE, 'UCIHAR': LocalModelForUCIHAR, 'KUHAR': LocalModelForKUHAR}
        model_list = []
        model_list.append(global_models[args.dataset](args))
        for i in range(args.client_num):
            model_list.append(local_models[args.dataset](args, i))
        model_list = [model.to(device) for model in model_list]
        # Adam
        if args.optimizer == 'adam':
            optimizer_list = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in model_list]
        # SGD
        elif args.optimizer == 'sgd':
            optimizer_list = [
                torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay) for
                model in model_list]
        # loss function
        if args.dataset in ['MOSI', 'MOSEI']:
            criterion = nn.L1Loss().to(device)
        elif args.dataset in ['URFUNNY', 'VISIONTOUCH', 'MUSTARD', 'MIMIC', 'NUSWIDE', 'UCIHAR', 'KUHAR']:
            criterion = nn.CrossEntropyLoss().to(device)
        elif args.dataset in ['MUJOCO']:
            criterion = nn.MSELoss().to(device)
        elif args.dataset in ['MMIMDB', 'PTBXL']:
            criterion = nn.BCEWithLogitsLoss().to(device)
        else:
            raise_dataset_exception()
        # dataloader
        if args.eval_mode == 'robustness':
            process = _process_emo_robustness if args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD'] else _process_none_robustness
        else:
            process = _process_emo if args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD'] else _process_none
        collate_fn_with_args = functools.partial(process, args=args)

        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                                                   collate_fn=collate_fn_with_args)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=args.batch_size, shuffle=True,
                                                   collate_fn=collate_fn_with_args)
        if test_data is None:
            test_loader = None
        else:
            test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True,
                                                      collate_fn=collate_fn_with_args)

        # init trainer
        trainers = {'efficiency': Trainer_Efficiency, 'robustness': Trainer_Robustness, 'security': Trainer_Security}
        trainer = trainers[args.eval_mode](device, model_list, optimizer_list, criterion, train_loader, valid_loader, test_loader, logger, args, checkpoint=None)

        trainer.train()
        if args.eval_mode == 'efficiency':
            # save the results
            results_dict = {'metric': trainer.metric_per_ep, 'metric_test': trainer.metric_test_per_ep, 'communication_cost': trainer.communication_cost_per_ep, 'execution_time': trainer.execution_time_per_ep}
            result_file_name = './results/{}/{}_seed_{}_{}.pk'.format(args.dataset, args.method_name, seed, current_time)
            f = open(result_file_name, 'wb')
            # save
            pickle.dump(results_dict, f)
            # close file
            f.close()

if __name__ == '__main__':
    currentDateAndTime = datetime.now()
    parser = argparse.ArgumentParser()

    parser = parser_add(parser)

    args = parser.parse_args()

    main(args)

from dataset.utils.utils import prepare_data_vfl
from method.security.LFBA import Trainer_LFBA
from utils.utils import *
from utils.performance import f1_score, eval_multi_affect
from method.security.AMC_PMC import Trainer_AMC_PMC
from method.security.GRNA import Trainer_GRNA
from method.security.Norm_Direction_Score import Trainer_Norm_Direction_Score
from method.security.MIA import Trainer_MIA
from method.security.TECB import Trainer_TECB

class Trainer_Security:
    def __init__(self, device, model_list, optimizer_list, criterion, train_loader, valid_loader,
                 test_loader, logger, args=None, checkpoint=None):
        init_functions = {'pmc': Trainer_AMC_PMC, 'amc': Trainer_AMC_PMC, 'norm_score': Trainer_Norm_Direction_Score,  # Label-Inference
                 'direction_score': Trainer_Norm_Direction_Score, 'grna': Trainer_GRNA, 'mia': Trainer_MIA,  # Feature Inference
                 'tecb': Trainer_TECB, 'lfba': Trainer_LFBA} # Backdoor Attacks
        self.trainer = init_functions[args.method_name](device, model_list, optimizer_list, criterion, train_loader, valid_loader, test_loader, logger, args)


    def train(self):
        self.trainer.train()
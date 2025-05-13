import copy
from functools import partial

from dataset.utils.utils import prepare_data_vfl
from utils.utils import *
from utils.performance import f1_score, eval_multi_affect
from utils import possible_defenses
from dataset.utils.utils import _process_emo
import torch.optim as optim
import torch.nn.functional as F
import time
import torch.nn as nn
import torch.nn.init as init

from torch.optim.optimizer import Optimizer
import torch
from time import time
from model.MUSTARD_models import*
from model.UCIHAR_models import*
from model.KUHAR_models import*
from model.NUSWIDE_models import*
from model.URFUNNY_models import*
from dataset.utils.utils import _process_none

last_whole_model_params_list = []
new_whole_model_params_list = []
batch_cos_list = []
near_minimum = False

def weights_init_ones(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.ones_(m.weight)

class MaliciousSGD(Optimizer):

    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, gamma_lr_scale_up=1.0, min_grad_to_process=1e-4):

        self.last_parameters_grads = []
        self.gamma_lr_scale_up = gamma_lr_scale_up
        self.min_grad_to_process = min_grad_to_process
        self.min_ratio = 1.0
        self.max_ratio = 5.0

        self.certain_grad_ratios = torch.tensor([])

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MaliciousSGD, self).__init__(params, defaults)
    def __setstate__(self, state):
        super(MaliciousSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        id_group = 0
        if len(self.last_parameters_grads) < len(self.param_groups):
            for i in range(len(self.param_groups)):
                self.last_parameters_grads.append([])

        for group in self.param_groups:

            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            start = time()
            id_parameter = 0

            for p in group['params']:
                if p.grad is None:
                    continue

                if weight_decay != 0:
                    p.grad.data.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(p.grad.data).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, p.grad.data)
                    if nesterov:
                        p.grad.data = p.grad.data.add(momentum, buf)
                    else:
                        p.grad.data = buf

                if not near_minimum:
                    if len(self.last_parameters_grads[id_group]) <= id_parameter:
                        self.last_parameters_grads[id_group].append(p.grad.clone().detach())
                    else:
                        last_parameter_grad = self.last_parameters_grads[id_group][id_parameter]
                        current_parameter_grad = p.grad.clone().detach()
                        ratio_grad_scale_up = 1.0 + self.gamma_lr_scale_up * (current_parameter_grad / (last_parameter_grad + 1e-7))
                        ratio_grad_scale_up = torch.clamp(ratio_grad_scale_up, self.min_ratio, self.max_ratio)
                        p.grad.mul_(ratio_grad_scale_up)
                end = time()
                current_parameter_grad = p.grad.clone().detach()
                self.last_parameters_grads[id_group][id_parameter] = current_parameter_grad

                p.data.add_(-group['lr'], p.grad.data)

                id_parameter += 1
            id_group += 1

        return loss

class BottomModelPlus(nn.Module):
    def __init__(self, args, size_bottom_out, num_classes, num_layer=1, activation_func_type='ReLU', use_bn=True):
        super(BottomModelPlus, self).__init__()
        self.bottom_model = eval('LocalModelFor' + args.dataset + '(args,' + str(args.attack_client) + ')')

        dict_activation_func_type = {'ReLU': F.relu, 'Sigmoid': F.sigmoid, 'None': None}
        self.activation_func = dict_activation_func_type[activation_func_type]
        self.num_layer = num_layer
        self.use_bn = use_bn
        self.args = args
        self.fc_1 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_1 = nn.BatchNorm1d(size_bottom_out)
        self.fc_1.apply(weights_init_ones)

        self.fc_2 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_2 = nn.BatchNorm1d(size_bottom_out)
        self.fc_2.apply(weights_init_ones)

        self.fc_3 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_3 = nn.BatchNorm1d(size_bottom_out)
        self.fc_3.apply(weights_init_ones)

        self.fc_4 = nn.Linear(size_bottom_out, size_bottom_out, bias=True)
        self.bn_4 = nn.BatchNorm1d(size_bottom_out)
        self.fc_4.apply(weights_init_ones)

        self.fc_final = nn.Linear(size_bottom_out, num_classes, bias=True)
        self.bn_final = nn.BatchNorm1d(size_bottom_out)
        self.fc_final.apply(weights_init_ones)

    def forward(self, x):
        x = self.bottom_model(x)
        if self.num_layer >= 2:
            if self.use_bn:
                x = self.bn_1(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_1(x)

        if self.num_layer >= 3:
            if self.use_bn:
                x = self.bn_2(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_2(x)

        if self.num_layer >= 4:
            if self.use_bn:
                x = self.bn_3(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_3(x)

        if self.num_layer >= 5:
            if self.use_bn:
                x = self.bn_4(x)
            if self.activation_func:
                x = self.activation_func(x)
            x = self.fc_4(x)
        if self.use_bn:
            x = self.bn_final(x)
        if self.activation_func:
            x = self.activation_func(x)
        x = self.fc_final(x)

        return x

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


class WeightEMA(object):
    def __init__(self, args, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.type(torch.float)
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param = param.type(torch.float)
            param.mul_(1 - self.wd)


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, args, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch, args.epoch)


def precision_recall(output, target):
    right_samples_num = 0
    TP_samples_num = 0
    TN_samples_num = 0
    FP_samples_num = 0
    FN_samples_num = 0
    wrong_samples_num = 0

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    y_true = np.array(target.clone().detach().cpu())
    y_pred = np.array(pred.clone().detach().cpu()[0])
    if sum(y_pred) == 0:
        y_pred = np.ones_like(y_pred)
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            if y_true[i] == 1.:
                TP_samples_num += 1
            else:
                TN_samples_num += 1
            right_samples_num += 1
        else:
            if y_pred[i] == 1.:
                FP_samples_num += 1
            else:
                FN_samples_num += 1
            wrong_samples_num += 1

    if (TP_samples_num + FP_samples_num) != 0:
        precision = TP_samples_num / (TP_samples_num + FP_samples_num)
    else:
        precision = 0
    if (TP_samples_num + FN_samples_num) != 0:
        recall = TP_samples_num / (TP_samples_num + FN_samples_num)
    else:
        recall = 0

    return precision, recall


def accuracy_label_inference(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    #print(output)
    #print(maxk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Trainer_AMC_PMC:
    def __init__(self, device, model_list, optimizer_list, criterion, train_loader, valid_loader,
                 test_loader, logger, args=None, checkpoint=None):
        self.device = device
        self.model_list = model_list
        self.optimizer_list = optimizer_list
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.logger = logger
        self.args = args
        self.checkpoint = checkpoint
        # adversarial options
        self.defense_ppdl = args.ppdl
        self.defense_gc = args.gc
        self.defense_lap_noise = args.lap_noise
        self.defense_multistep_grad = args.multistep_grad
        self.criterion_local = keep_predict_loss
        # AMC has adaptive optimizer, while PMC uses the default setting
        if self.args.method_name == 'pmc':
            pass
        else:
            self.optimizer_list[self.args.attack_client+1] = MaliciousSGD(
                self.model_list[self.args.attack_client+1].parameters(),
                lr=self.args.lr, momentum=args.momentum,
                weight_decay=args.weight_decay,
                gamma_lr_scale_up = 2.0, nesterov=True)
            

        stone1 = args.stone1
        stone2 = args.stone2

        self.scheduler_list = [torch.optim.lr_scheduler.MultiStepLR(opt,
                                                                    milestones=[stone1, stone2],
                                                                    gamma=args.step_gamma) for opt in
                               self.optimizer_list]
        self.grad_args = [None for _ in range(self.args.client_num)]
    def train(self):
        # Step1: train bottom model
        self.train_model()
        # Step2: train a local classifier
        self.train_attack()

    def train_attack(self):

        def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch,
                  use_cuda, num_classes):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            losses_x = AverageMeter()
            losses_u = AverageMeter()
            ws = AverageMeter()
            end = time()

            labeled_train_iter = iter(labeled_trainloader)
            unlabeled_train_iter = iter(unlabeled_trainloader)

            model.train()

            for batch_idx in range(self.args.val_iteration):
                try:
                    inputs_x, x_length, _, targets_x, _ = next(labeled_train_iter)
                except StopIteration:
                    labeled_train_iter = iter(labeled_trainloader)
                    inputs_x, x_length, _, targets_x, _ = next(labeled_train_iter)
                try:
                    inputs_u, u_length, _, _, _ = next(unlabeled_train_iter)
                except StopIteration:
                    unlabeled_train_iter = iter(unlabeled_trainloader)
                    inputs_u, u_length, _, _, _ = next(unlabeled_train_iter)
                # measure data loading time
                data_time.update(time() - end)

                # in vertical federated learning scenario, attacker only has part of data.
                inputs_x = prepare_data_vfl(inputs_x, self.args)[self.args.attack_client]

                inputs_u = prepare_data_vfl(inputs_u, self.args)[self.args.attack_client]
                inputs_x = inputs_x.type(torch.float)
                inputs_u = inputs_u.type(torch.float)

                batch_size = inputs_x.size(0)

                # Transform label to one-hot
                if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
                    raise Exception('Unsupported dataset. Only for classification task.')
                targets_x = targets_x.view(-1, 1).type(torch.long)
                if self.args.dataset in ['MUSTARD']:
                    targets_x = torch.where(targets_x > 0, 1, 0)
                targets_x = torch.zeros(batch_size, num_classes).scatter_(1, targets_x, 1)

                if use_cuda:
                    inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
                    inputs_u = inputs_u.cuda()

                with torch.no_grad():
                    targets_x.view(-1, 1).type(torch.long)  # compute guessed labels of unlabel samples
                    outputs_u = None
                    if self.args.dataset in ['URFUNNY']: 
                        outputs_u = model([inputs_u, u_length[self.args.attack_client]])
                    else:
                        outputs_u = model(inputs_u)
                    p = torch.softmax(outputs_u, dim=1)
                    pt = p ** (1 / self.args.T)
                    targets_u = pt / pt.sum(dim=1, keepdim=True)
                    targets_u = targets_u.detach()

                # mixup
                try:
                    all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
                    all_targets = torch.cat([targets_x, targets_u], dim=0)
                except RuntimeError:
                    #print("batch_size:{}")
                    print("inputs_x:{}".format(inputs_x.shape))
                    print("inputs_u:{}".format(inputs_u.shape))


                l = np.random.beta(self.args.alpha, self.args.alpha)

                l = max(l, 1 - l)

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]

                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b

                # interleave labeled and unlabeled samples between batches to get correct batch norm calculation
                mixed_input = list(torch.split(mixed_input, batch_size))
                mixed_input = interleave(mixed_input, batch_size)
               
                if self.args.dataset in ['URFUNNY']:
                    x_length_tensor = torch.stack(x_length, dim=0)
                    u_length_tensor = torch.stack(u_length, dim=0)

                    all_length = torch.cat([x_length_tensor, u_length_tensor], dim=1)
                    length_a, length_b = all_length[self.args.attack_client], all_length[self.args.attack_client][idx]
            
                    mixed_length = l * length_a + (1 - l) * length_b
                    mixed_length = list(torch.split(mixed_length, batch_size))
                    mixed_length = interleave(mixed_length, batch_size)
                    logits = [model([mixed_input[0], mixed_length[0]])]
                    #print("ok task1")
                    assert len(mixed_input) == len(mixed_length),"len doesn't match"
                    for i in range(1, len(mixed_input)):
                        #for input in mixed_input[1:]:
                        input = mixed_input[i]
                        length = mixed_length[i]
                        logits.append(model([input, length]))
                else:
                    logits = [model(mixed_input[0])]
                    for input in mixed_input[1:]:
                        logits.append(model(input))

                # put interleaved samples back
                logits = interleave(logits, batch_size)
                logits_x = logits[0]
                logits_u = torch.cat(logits[1:], dim=0)

                Lx, Lu, w = criterion(self.args, logits_x, mixed_target[:logits_x.shape[0]], logits_u, mixed_target[logits_x.shape[0]:],
                                      epoch + batch_idx / self.args.val_iteration)

                loss = Lx + w * Lu

                # record loss
                losses.update(loss.item(), inputs_x.size(0))
                losses_x.update(Lx.item(), inputs_x.size(0))
                losses_u.update(Lu.item(), inputs_x.size(0))
                ws.update(w, inputs_x.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ema_optimizer.step()

                # measure elapsed time
                batch_time.update(time() - end)
                end = time()
                # print('one batch training done')
                if batch_idx % 250 == 0:
                    print("batch_idx:", batch_idx, " loss:", losses.avg)
            return losses.avg, losses_x.avg, losses_u.avg

        def validate(valloader, model, criterion, epoch, use_cuda, mode, num_classes):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            topk = AverageMeter()
            precision = AverageMeter()
            recall = AverageMeter()

            # switch to evaluate mode
            model.eval()

            end = time()
            with torch.no_grad():
                for batch_idx, (inputs, length, _, targets, _) in enumerate(valloader):
                    # in vertical federated learning scenario, attacker(party A) only has part of features, i.e. half of the img
                    inputs = prepare_data_vfl(inputs, self.args)[self.args.attack_client]

                    # measure data loading time
                    data_time.update(time() - end)

                    if use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
                    # compute output
                    inputs = inputs.type(torch.float)
                    outputs = None
                    if self.args.dataset in ['URFUNNY']:
                        outputs = model([inputs, length[0]])
                    else:
                        outputs = model(inputs)

                    if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH', 'UCIHAR',
                                             'KUHAR']:
                        targets = targets.squeeze()
                        if self.args.dataset in ['MUSTARD']:
                            targets = torch.where(targets > 0, 1, 0)
                    if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
                        raise Exception('Unsupported dataset. Only for classification task.')
                    targets = targets.type(torch.long)
                    loss = criterion(outputs, targets)

                    # measure accuracy and record loss
                    prec1, preck = accuracy_label_inference(outputs, targets, topk=(1, self.args.k))
                    if num_classes == 2:
                        prec, rec = precision_recall(outputs, targets)
                        precision.update(prec, inputs.size(0))
                        recall.update(rec, inputs.size(0))

                    losses.update(loss.item(), inputs.size(0))
                    top1.update(prec1.item(), inputs.size(0))
                    topk.update(preck.item(), inputs.size(0))

                    # measure elapsed time
                    batch_time.update(time() - end)
                    end = time()
                    # print('one batch done')
            self.logger.info("Dataset Overall Statistics:")
            if num_classes == 2:
                self.logger.info("  precision:{}".format(precision.avg))
                self.logger.info("  recall:{}".format(recall.avg))
                if (precision.avg + recall.avg) != 0:
                    self.logger.info("  F1:"+str(2 * (precision.avg * recall.avg) / (precision.avg + recall.avg)))
                else:
                    self.logger.info("  F1:0")
            self.logger.info("top 1 accuracy:{}, top {} accuracy:{}".format(top1.avg, self.args.k, topk.avg))
            return losses.avg, top1.avg

        """
            Start training for attack
        """
        self.logger.info("=>Step 2: Training attack model...")
        self.logger.info("=> Prepare attack data...")
        train_data_complete = copy.deepcopy(self.train_loader.dataset)
        train_labels_complete = train_data_complete.targets
        train_labeled_idxs = []
        train_unlabeled_idxs = []
        num_class = len(np.unique(train_labels_complete))
        n_labeled_per_class = self.args.n_labeled_per_class
        if self.args.batch_size > (n_labeled_per_class * num_class):
            raise Exception("args.batch_size must be smaller than args.n_labeled")
        for i in range(num_class):
            idxs = np.where(train_labels_complete == i)[0]
            np.random.shuffle(idxs)
            train_labeled_idxs.extend(idxs[:n_labeled_per_class])
            train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])

        train_labeled_dataset = torch.utils.data.Subset(train_data_complete, train_labeled_idxs)
        train_unlabeled_dataset = torch.utils.data.Subset(train_data_complete, train_unlabeled_idxs)

        process = partial(_process_emo, args=self.args) if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD'] else \
            partial(_process_none, args=self.args)

        self.labeled_trainloader = torch.utils.data.DataLoader(dataset=train_labeled_dataset,
                                                               batch_size=self.args.batch_size, shuffle=True,
                                                               collate_fn=process)

        self.unlabeled_trainloader = torch.utils.data.DataLoader(dataset=train_unlabeled_dataset,
                                                                 batch_size=self.args.batch_size, shuffle=True,
                                                                 collate_fn=process)

        self.train_complete_trainloader = torch.utils.data.DataLoader(dataset=train_data_complete,
                                                                      batch_size=self.args.batch_size, shuffle=True,
                                                                      collate_fn=process)

        def create_model(ema=False, size_bottom_out=10, num_classes=10):
            model = BottomModelPlus(self.args, size_bottom_out, num_classes,
                                    num_layer=1,
                                    activation_func_type='ReLU',
                                    use_bn=True)
            model = model.cuda()

            if ema:
                for param in model.parameters():
                    param.detach_()

            return model
        

        model = create_model(ema=False, size_bottom_out=self.model_list[self.args.attack_client+1].backbone[-2].out_features,
                                num_classes=num_class)
        ema_model = create_model(ema=True, size_bottom_out=self.model_list[self.args.attack_client+1].backbone[-2].out_features,
                                    num_classes=num_class)
        train_criterion = SemiLoss()
        criterion = self.criterion
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        ema_optimizer = WeightEMA(self.args, model, ema_model, alpha=self.args.ema_decay)

        self.logger.info("=> Load bottom model...")
        filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar')
        assert os.path.isfile(filename), 'Error: no checkpoint directory found!'

        checkpoint = torch.load(filename)

        model.bottom_model.load_state_dict(checkpoint['state_dict'][self.args.attack_client + 1])
        ema_model.bottom_model.load_state_dict(checkpoint['state_dict'][self.args.attack_client + 1])

        test_accs = []
        use_cuda = torch.cuda.is_available()
        # Train and test
        best_acc = 0
        best_test_acc = 0
        for epoch in range(self.args.attack_train_epochs_mc):
            self.logger.info('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, self.args.epoch, self.args.lr))

            train_loss, train_loss_x, train_loss_u = train(self.labeled_trainloader, self.unlabeled_trainloader, model,
                                                           optimizer,
                                                           ema_optimizer, train_criterion, epoch, use_cuda, num_class)
            self.logger.info("---Label inference on complete training dataset:")
            _, train_acc = validate(self.train_complete_trainloader, ema_model, criterion, epoch, use_cuda,
                                    mode='Train Stats',
                                    num_classes=num_class)
            if self.test_loader is None:
                self.logger.info("---Label inference on testing dataset:")
                test_loss, test_acc = validate(self.valid_loader, ema_model, criterion, epoch, use_cuda,
                                               mode='Test Stats',
                                               num_classes=num_class)
            else:
                self.logger.info("---Label inference on testing dataset:")
                test_loss, test_acc = validate(self.test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats',
                                               num_classes=num_class)

            # save model
            is_best = train_acc > best_acc
            best_acc = max(train_acc, best_acc)
            best_test_acc = max(test_acc, best_test_acc)
            test_accs.append(test_acc)
            self.logger.info('Test attack top 1 accuracy:{}'.format(test_acc))
        self.logger.info('Best attack top 1 accuracy of training dataset:{}'.format(best_acc))
        self.logger.info('Best attack top 1 accuracy of testing dataset:{}'.format(best_test_acc))

    def train_model(self):
        self.logger.info("=>Step 1: Training Baseline...")
        epoch_loss_list = []
        model_list = self.model_list
        best_acc = 0
        best_epoch = 0
        no_change = 0
        best_mae = float("inf")
        best_eval_results = {}
        best_mse = float("inf")
        valid_f1_macro = 0
        valid_f1_micro = 0
        best_f1_macro = 0
        best_f1_micro = 0

        if self.checkpoint:
            best_acc = self.checkpoint['best_acc']
        # train and update
        for ep in range(self.args.start_epoch, self.args.epoch):
            model_list = [model.train() for model in model_list]
            batch_loss_list = []
            total = 0
            correct = 0
            prede = []
            pred = []
            true = []

            self.logger.info("=> Start Training Epoch {}...".format(ep + 1))

            for step, (x_n, length, index, y, mask) in enumerate(self.train_loader):
                x = x_n
                if isinstance(x, list):
                    for i in range(len(x)):
                        x[i] = x[i].to(self.device).float()
                else:
                    x = x.to(self.device).float()
                y = y.to(self.device).long()
                # split data for vfl
                x_split_list = prepare_data_vfl(x, self.args)

                if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                    y = y.squeeze()
                    if self.args.dataset in ['MUSTARD']:
                        y = torch.where(y > 0, 1, 0)
                if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
                    y = y.float()  # for MSELoss

                # get the local model outputs
                local_output_list = []
                if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                    for i in range(self.args.client_num):
                        local_output_list.append(model_list[i + 1]([x_split_list[i], length[i]]))
                else:
                    for i in range(self.args.client_num):
                        local_output_list.append(model_list[i + 1](x_split_list[i]))
                        

                global_input_list = []
                for i in range(self.args.client_num):
                    global_input_list.append(torch.tensor([], requires_grad=True))
                    global_input_list[i].data = local_output_list[i].data

                global_output = model_list[0](global_input_list)

                # global model backward
                loss_framework = update_top_model_one_batch(optimizer=self.optimizer_list[0],
                                                                 model=self.model_list,
                                                                 output=global_output,
                                                                 batch_target=y,
                                                                 loss_func=self.criterion, args=self.args)
                
                batch_loss_list.append(loss_framework.item())

                # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
                grad_output_list = [input_tensor_top.grad.to(torch.device(f"cuda:{self.args.device}")) for input_tensor_top in global_input_list]

                for i in range(self.args.client_num):
                    if self.grad_args[i] == None:
                        layer_id = 0
                        sum_layer_grad = 0
                        tem_now_id = 0
                        grad_num, num_grad_per_layer = possible_defenses.get_grad_num([grad_output_list[i]])
                        self.grad_args[i] = [(0,0) for _ in range(grad_num)]
                        for onelayer in num_grad_per_layer:
                            for _ in range(onelayer):
                                self.grad_args[i][tem_now_id] = (layer_id, tem_now_id - sum_layer_grad)
                                tem_now_id +=1
                            layer_id += 1
                            sum_layer_grad += onelayer
                # privacy preserving deep learning
                if self.defense_ppdl:
                    for i in range(self.args.client_num):
                        possible_defenses.dp_gc_ppdl(epsilon=1.8, sensitivity=1,
                                                     layer_grad_list=[grad_output_list[i]],
                                                     theta_u=self.args.ppdl_theta_u, gamma=0.001, tau=0.0001, get_grad_layer_id=self.grad_args[i])
                # gradient compression
                if self.defense_gc:
                    tensor_pruner = possible_defenses.TensorPruner(zip_percent=self.args.gc_preserved_percent)
                    for tensor_id in range(len(grad_output_list)):
                        tensor_pruner.update_thresh_hold(grad_output_list[tensor_id])
                        # print("tensor_pruner.thresh_hold:", tensor_pruner.thresh_hold)
                        grad_output_list[tensor_id] = tensor_pruner.prune_tensor(
                            grad_output_list[tensor_id])
                # differential privacy
                if self.defense_lap_noise:
                    dp = possible_defenses.DPLaplacianNoiseApplyer(beta=self.args.noise_scale)
                    for tensor_id in range(len(grad_output_list)):
                        grad_output_list[tensor_id] = dp.laplace_mech(grad_output_list[tensor_id])
                # multistep gradient
                if self.defense_multistep_grad:
                    for tensor_id in range(len(grad_output_list)):
                        grad_output_list[tensor_id] = possible_defenses.multistep_gradient(
                            grad_output_list[tensor_id], bins_num=self.args.multistep_grad_bins,
                            bound_abs=self.args.multistep_grad_bound_abs)

                for i in range(self.args.client_num):
                    update_bottom_model_one_batch(optimizer=self.optimizer_list[i + 1],
                                                       model=self.model_list[i + 1],
                                                       output=local_output_list[i],
                                                       batch_target=grad_output_list[i],
                                                       loss_func=self.criterion_local, args=self.args)

                for i in range(self.args.client_num):
                    self.scheduler_list[i].step()

                if self.args.dataset in ['MOSI', 'MOSEI']:
                    current_loss = sum(batch_loss_list) / len(batch_loss_list)
                    oute = global_output.detach().cpu().numpy().tolist()
                    pred.append(torch.LongTensor(oute))
                    pred_t = torch.cat(pred, 0)
                    true.append(y)
                    true_t = torch.cat(true, 0)

                    eval_results = eval_multi_affect(true_t, pred_t)
                    if step % self.args.print_steps == 0:
                        self.logger.info(
                            'Epoch: {}, {}/{}: train loss: {:.4f}, mean absolute error (mae): {:.4f}, train main task accuracy (f1_score/acc2/acc3/acc5/acc7): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
                                ep + 1,
                                step + 1,
                                len(self.train_loader), current_loss,
                                eval_results['mae'], eval_results['f1_score'],
                                eval_results['acc_2'], eval_results['acc_3'], eval_results['acc_5'],
                                eval_results['acc_7']))

                elif self.args.dataset in ['MUJOCO']:
                    current_loss = sum(batch_loss_list) / len(batch_loss_list)

                    if step % self.args.print_steps == 0:
                        self.logger.info(
                            'Epoch: {}, {}/{}: train loss (MSE): {:.4f}'.format(ep + 1, step + 1,
                                                                                len(self.train_loader),
                                                                                current_loss))

                elif self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                    # calculate the training accuracy
                    _, predicted = global_output.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()

                    # train_acc
                    train_acc = correct / total
                    current_loss = sum(batch_loss_list) / len(batch_loss_list)

                    if step % self.args.print_steps == 0:
                        self.logger.info(
                            'Epoch: {}, {}/{}: train loss: {:.4f}, train main task accuracy: {:.4f}'.format(ep + 1,
                                                                                                            step + 1,
                                                                                                            len(self.train_loader),
                                                                                                            current_loss,
                                                                                                            train_acc))

                elif self.args.dataset in ['MMIMDB', 'PTBXL']:
                    # calculate the training accuracy
                    current_loss = sum(batch_loss_list) / len(batch_loss_list)
                    oute = global_output.detach().cpu()
                    pred.append(torch.sigmoid(oute).round())
                    pred_t = torch.cat(pred, 0)
                    true.append(y)
                    true_t = torch.cat(true, 0).cpu()

                    f1_macro = f1_score(true_t, pred_t, average="macro")
                    f1_micro = f1_score(true_t, pred_t, average="micro")

                    if step % self.args.print_steps == 0:
                        self.logger.info(
                            'Epoch: {}, {}/{}: train loss: {:.4f}, train f1_macro: {:.4f}, train f1_micro: {:.4f}'.format(
                                ep + 1,
                                step + 1,
                                len(self.train_loader),
                                current_loss,
                                f1_macro, f1_micro))

            epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
            epoch_loss_list.append(epoch_loss)
            if self.args.dataset in ['MOSI', 'MOSEI']:
                eval_results = self.valid(ep + 1)
                valid_mae = eval_results['mae']
                if valid_mae < best_mae:
                    # best accuracy
                    best_mae = valid_mae
                    best_eval_results = eval_results
                    no_change = 0
                    best_epoch = ep + 1
                    # save model
                    self.logger.info("=> Save best model...")
                    state = {
                        'epoch': ep + 1,
                        'best_mae': best_mae,
                        'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
                        'optimizer': [self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))],
                    }
                    filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar'.format(ep + 1))
                    torch.save(state, filename)
                else:
                    no_change += 1
                self.logger.info(
                    '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, mean absolute error (mae): {}, '
                    'accuracy (f1_score/acc2/acc3/acc5/acc7): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
                        ep + 1,
                        no_change,
                        best_epoch, best_eval_results['mae'], best_eval_results['f1_score'], best_eval_results['acc_2'],
                        best_eval_results['acc_3'], best_eval_results['acc_5'], best_eval_results['acc_7']))
                if no_change == self.args.early_stop:
                    self.test(ep + 1)
                    return

            elif self.args.dataset in ['MUJOCO']:
                valid_mse = self.valid(ep + 1)
                if valid_mse < best_mse:
                    # best accuracy
                    best_mse = valid_mse
                    no_change = 0
                    best_epoch = ep + 1
                    # save model
                    self.logger.info("=> Save best model...")
                    state = {
                        'epoch': ep + 1,
                        'best_mse': best_mse,
                        'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
                        'optimizer': [self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))],
                    }
                    filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar'.format(ep + 1))
                    torch.save(state, filename)
                else:
                    no_change += 1
                self.logger.info(
                    '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, mean square error (mse): {}'.format(
                        ep + 1,
                        no_change,
                        best_epoch, best_mse))
                if no_change == self.args.early_stop:
                    self.test(ep + 1)
                    return

            elif self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                valid_acc = self.valid(ep + 1)
                # if test_acc > best_acc:
                if valid_acc > best_acc:
                    # best accuracy
                    best_acc = valid_acc
                    no_change = 0
                    best_epoch = ep + 1
                    # save model
                    self.logger.info("=> Save best model...")
                    state = {
                        'epoch': ep + 1,
                        'best_acc': best_acc,
                        'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
                        'optimizer': [self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))],
                    }
                    filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar'.format(ep + 1))
                    torch.save(state, filename)
                else:
                    no_change += 1
                self.logger.info(
                    '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, accuracy: {:.4f}'.format(
                        ep + 1,
                        no_change,
                        best_epoch, best_acc))
                if no_change == self.args.early_stop:
                    self.test(ep + 1)
                    return

            elif self.args.dataset in ['MMIMDB', 'PTBXL']:
                valid_f1_macro, valid_f1_micro = self.valid(ep + 1)
                if valid_f1_macro > best_f1_macro:
                    # best accuracy
                    best_f1_macro = valid_f1_macro
                    best_f1_micro = valid_f1_micro
                    no_change = 0
                    best_epoch = ep + 1
                    # save model
                    self.logger.info("=> Save best model...")
                    state = {
                        'epoch': ep + 1,
                        'best_f1_macro': best_f1_macro,
                        'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
                        'optimizer': [self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))],
                    }
                    filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar'.format(ep + 1))
                    torch.save(state, filename)
                else:
                    no_change += 1
                self.logger.info(
                    '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
                        ep + 1,
                        no_change,
                        best_epoch, best_f1_macro, best_f1_micro))
                if no_change == self.args.early_stop:
                    self.test(ep + 1)
                    return

            if (ep + 1) % 10 == 0:
                self.test(ep + 1)

    def valid(self, ep):
        self.logger.info("=> Validation Accuracy...")
        model_list = self.model_list
        model_list = [model.eval() for model in model_list]
        # main task accuracy
        batch_loss_list = []
        total = 0
        correct = 0
        total_target = 0
        correct_target = 0
        pred = []
        true = []
        for step, (x_n, length, index, y, mask) in enumerate(self.valid_loader):
            x = x_n
            if isinstance(x, list):
                for i in range(len(x)):
                    x[i] = x[i].to(self.device).float()
            else:
                x = x.to(self.device).float()
            y = y.to(self.device).long()
            # split data for vfl
            x_split_list = prepare_data_vfl(x, self.args)
            local_output_list = []
            # get the local model outputs
            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                for i in range(self.args.client_num):
                    local_output_list.append(model_list[i + 1]([x_split_list[i], length[i]]))
            else:
                for i in range(self.args.client_num):
                    local_output_list.append(model_list[i + 1](x_split_list[i]))

            global_output = model_list[0](local_output_list)

            # loss
            if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                y = y.squeeze()
                if self.args.dataset in ['MUSTARD']:
                    y = torch.where(y > 0, 1, 0)
            if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
                y = y.float()  # for MSELoss
            loss = self.criterion(global_output, y)

            if self.args.dataset in ['MUJOCO']:
                batch_loss_list.append(loss.item())
            else:
                batch_loss_list.append(loss.item())
                # calculate the validation accuracy
                if self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                    _, predicted = global_output.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()

                # calculate the validation accuracy
                if self.args.dataset in ['MMIMDB', 'PTBXL']:
                    oute = global_output.detach().cpu()
                    pred.append(torch.sigmoid(oute).round())
                elif self.args.dataset in ['MOSI', 'MOSEI']:
                    oute = global_output.detach().cpu().numpy().tolist()
                    pred.append(torch.LongTensor(oute))
                true.append(y)

        # main task accuracy
        if self.args.dataset in ['MOSI', 'MOSEI']:
            current_loss = sum(batch_loss_list) / len(batch_loss_list)
            pred_t = torch.cat(pred, 0)
            true_t = torch.cat(true, 0)
            eval_results = eval_multi_affect(true_t, pred_t)
            self.logger.info(
                'Valid Epoch: {}, valid samples: {}, valid loss: {}, mean absolute error (mae): {:.4f}, valid main task accuracy (f1_score/acc2/acc3/acc5/acc7): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
                    ep,
                    len(self.valid_loader.dataset), current_loss,
                    eval_results['mae'], eval_results['f1_score'],
                    eval_results['acc_2'], eval_results['acc_3'], eval_results['acc_5'], eval_results['acc_7']))
            return eval_results

        elif self.args.dataset in ['MUJOCO']:
            current_loss = sum(batch_loss_list) / len(batch_loss_list)
            self.logger.info(
                'Valid Epoch: {}, valid samples: {}, mean square error (mse): {:.4f}'.format(
                    ep,
                    len(self.valid_loader.dataset), current_loss))
            return current_loss  # mse value

        elif self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
            # valid_acc
            valid_acc = correct / total
            current_loss = sum(batch_loss_list) / len(batch_loss_list)
            self.logger.info(
                'Valid Epoch: {}, valid samples: {}, valid loss: {:.4f}, valid main task accuracy: {:.4f}'.format(ep,
                                                                                                                  len(self.valid_loader.dataset),
                                                                                                                  current_loss,
                                                                                                                  valid_acc))

            return valid_acc

        elif self.args.dataset in ['MMIMDB', 'PTBXL']:
            current_loss = sum(batch_loss_list) / len(batch_loss_list)
            pred_t = torch.cat(pred, 0)
            true_t = torch.cat(true, 0).cpu()
            f1_micro = f1_score(true_t, pred_t, average="micro")
            f1_macro = f1_score(true_t, pred_t, average="macro")
            self.logger.info(
                'Valid Epoch: {}, valid samples: {}, valid loss: {:.4f}, valid f1_macro: {:.4f}, valid f1_micro: {:.4f}'.format(
                    ep,
                    len(self.valid_loader.dataset),
                    current_loss,
                    f1_macro, f1_micro))

            return f1_macro, f1_micro

    def test(self, ep):
        if self.test_loader is None:
            return
        self.logger.info("=> Test Accuracy...")
        check_path = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar'.format(ep + 1))
        self.logger.info("=> loading best test checkpoint '{}'".format(check_path))
        checkpoint_test = torch.load(check_path, map_location=self.device)
        model_list = self.model_list
        model_list = [model.eval() for model in model_list]
        for i in range(len(model_list)):
            model_list[i].load_state_dict(checkpoint_test['state_dict'][i])
        self.logger.info("=> loaded test checkpoint '{}' (epoch {})"
                         .format(check_path, checkpoint_test['epoch']))
        # test main task accuracy
        batch_loss_list = []
        total = 0
        correct = 0
        total_target = 0
        correct_target = 0
        pred = []
        true = []
        for step, (x_n, length, index, y, mask) in enumerate(self.test_loader):
            x = x_n
            if isinstance(x, list):
                for i in range(len(x)):
                    x[i] = x[i].to(self.device).float()
            else:
                x = x.to(self.device).float()
            y = y.to(self.device).long()
            # split data for vfl
            x_split_list = prepare_data_vfl(x, self.args)
            local_output_list = []
            # get the local model outputs
            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                for i in range(self.args.client_num):
                    local_output_list.append(model_list[i + 1]([x_split_list[i], length[i]]))
            else:
                for i in range(self.args.client_num):
                    local_output_list.append(model_list[i + 1](x_split_list[i]))

            global_output = model_list[0](local_output_list)

            # loss
            if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH']:
                y = y.squeeze()
                if self.args.dataset == 'MUSTARD':
                    y = torch.where(y > 0, 1, 0)
            if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
                y = y.float()  # for MSELoss
            loss = self.criterion(global_output, y)

            if self.args.dataset in ['MUJOCO']:
                batch_loss_list.append(loss.item())
            else:
                batch_loss_list.append(loss.item())
                # calculate the test accuracy
                if self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH']:
                    _, predicted = global_output.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()

                # calculate the test accuracy
                if self.args.dataset in ['MMIMDB', 'PTBXL']:
                    oute = global_output.detach().cpu()
                    pred.append(torch.sigmoid(oute).round())
                elif self.args.dataset in ['MOSI', 'MOSEI']:
                    oute = global_output.detach().cpu().numpy().tolist()
                    pred.append(torch.LongTensor(oute))
                true.append(y)

        # main task accuracy
        if self.args.dataset in ['MOSI', 'MOSEI']:
            current_loss = sum(batch_loss_list) / len(batch_loss_list)
            pred_t = torch.cat(pred, 0)
            true_t = torch.cat(true, 0)
            eval_results = eval_multi_affect(true_t, pred_t)
            self.logger.info(
                'Test Epoch: {}, test samples: {}, test loss: {}, mean absolute error (mae): {:.4f}, test main task accuracy (f1_score/acc2/acc3/acc5/acc7): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
                    ep,
                    len(self.test_loader.dataset), current_loss,
                    eval_results['mae'], eval_results['f1_score'],
                    eval_results['acc_2'], eval_results['acc_3'], eval_results['acc_5'], eval_results['acc_7']))
            return eval_results

        elif self.args.dataset in ['MUJOCO']:
            current_loss = sum(batch_loss_list) / len(batch_loss_list)
            self.logger.info(
                'Test Epoch: {}, test samples: {}, mean square error (mse): {:.4f}'.format(
                    ep,
                    len(self.test_loader.dataset), current_loss))
            return current_loss  # mse value

        elif self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH']:
            # test_acc
            test_acc = correct / total
            current_loss = sum(batch_loss_list) / len(batch_loss_list)
            self.logger.info(
                'Test Epoch: {}, test samples: {}, test loss: {:.4f}, test main task accuracy: {:.4f}'.format(ep,
                                                                                                              len(self.test_loader.dataset),
                                                                                                              current_loss,
                                                                                                              test_acc))

            return test_acc

        elif self.args.dataset in ['MMIMDB', 'PTBXL']:
            current_loss = sum(batch_loss_list) / len(batch_loss_list)
            pred_t = torch.cat(pred, 0)
            true_t = torch.cat(true, 0).cpu()
            f1_micro = f1_score(true_t, pred_t, average="micro")
            f1_macro = f1_score(true_t, pred_t, average="macro")
            self.logger.info(
                'Test Epoch: {}, test samples: {}, test loss: {:.4f}, test f1_macro: {:.4f}, test f1_micro: {:.4f}'.format(
                    ep,
                    len(self.test_loader.dataset),
                    current_loss,
                    f1_macro, f1_micro))

            return f1_macro, f1_micro

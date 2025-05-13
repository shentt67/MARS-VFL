import torch
from dataset.utils.utils import prepare_data_vfl
from utils.utils import *
from utils.performance import f1_score, eval_multi_affect
from dataset.utils.utils import powerset_except_empty
import math
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

def laservfl(trainer, model_list, input, loss_d, correct_d, pred, true, num_batches, num_samples):
    trainer.local_iterations = trainer.args.local_iterations

    # build predictiors
    if len(trainer.model_list) < (2 * trainer.args.client_num):
        for i in range(trainer.args.client_num - 1):
            trainer.model_list.insert(0, eval('GlobalModelFor' + trainer.args.dataset + '(trainer.args)').to(trainer.device))
        if trainer.args.optimizer == 'adam':
            trainer.optimizer_list = [torch.optim.Adam(model.parameters(), lr=trainer.args.lr) for model in
                                      trainer.model_list]
        elif trainer.args.optimizer == 'sgd':
            trainer.optimizer_list = [
                torch.optim.SGD(model.parameters(), lr=trainer.args.lr, momentum=trainer.args.momentum,
                                weight_decay=trainer.args.weight_decay) for
                model in trainer.model_list]

    x = input[0]
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = x[i].to(trainer.device).float()
    else:
        x = x.to(trainer.device).float()

    y = input[-2]
    # split data for vfl
    x_split_list = prepare_data_vfl(x, trainer.args)

    local_output_list = []

    mask = input[-1]

    mask.to(trainer.device)

    if torch.sum(mask).item() == 0:
        num_batches -= 1  # in practice, this batch is not used
        num_samples -= len(x_split_list[0])  # in practice, these samples are not used
        return model_list, loss_d, -1, -1, -1, -1, num_batches, num_samples

    observed_blocks = torch.nonzero(mask).view(-1).tolist()

    # calculate embeddings
    embeddings = {}
    if trainer.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
        for i in observed_blocks:
            embeddings[i] = trainer.model_list[i + trainer.args.client_num]([x_split_list[i], input[-4][i]])
    else:
        for i in observed_blocks:
            embeddings[i] = trainer.model_list[i + trainer.args.client_num](x_split_list[i])

    # calculate outputs
    outputs_list = []
    for i, fusion_model in enumerate(trainer.model_list[0:trainer.args.client_num]):
        if i not in observed_blocks:
            continue
        sets_considered_by_head = [clients_l for clients_l in trainer.powerset if
                                   i in clients_l and set(clients_l).issubset(set(observed_blocks))]
        head_output = {}
        for num_clients_in_agg in range(1, len(observed_blocks) + 1):
            set_to_sample = [client_set for client_set in sets_considered_by_head if
                             len(client_set) == num_clients_in_agg]
            [sample] = random.sample(set_to_sample, 1)
            head_output[sample] = fusion_model([embeddings[j] for j in sample])
        outputs_list.append(head_output)

    # update with backwards
    total_loss = 0
    eval_results_list = []
    f1_macro_list = []
    f1_micro_list = []
    for outputs_per_task_d in outputs_list:
        head_loss = 0
        for clients_subset, outputs in outputs_per_task_d.items():
            loss = trainer.criterion(outputs, y)

            norm_constant = 1.0
            norm_constant = norm_constant / len(clients_subset)  # account for multiple heads

            # if a given element (client) must be in a set, then we do n-1 choose k-1 instead
            n, k = (len(observed_blocks) - 1, len(clients_subset) - 1)
            norm_constant = norm_constant * math.comb(n, k)

            head_loss += loss * norm_constant

            # We divide the metrics by the number of predictors we have for this task, so that we get averaged metrics
            # (across the heads performing each task). This allows for metrics which are comparable to those of the decoupled approach.
            # calculate eval results
            loss_d[clients_subset] += loss.item() * norm_constant
            if trainer.args.dataset not in ['MMIMDB', 'MOSI', 'MOSEI', 'MUJOCO']:
                predicted = outputs.argmax(1)
                correct_d[clients_subset] += (predicted == y).float().sum().item() * norm_constant

            if trainer.args.dataset in ['MOSI', 'MOSEI']:
                oute = outputs.detach().cpu().numpy().tolist()
                pred.append(torch.LongTensor(oute))
                pred_t = torch.cat(pred, 0)
                true.append(y)
                true_t = torch.cat(true, 0)
                eval_results = eval_multi_affect(true_t, pred_t)
                eval_results_list.append(eval_results)

            elif trainer.args.dataset in ['MUJOCO']:
                pass
            elif trainer.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH', 'UCIHAR',
                                       'KUHAR']:
                pass
            elif trainer.args.dataset in ['MMIMDB', 'PTBXL']:
                oute = outputs.detach().cpu()
                pred.append(torch.sigmoid(oute).round())
                pred_t = torch.cat(pred, 0)
                true.append(y)
                true_t = torch.cat(true, 0).cpu()

                f1_macro = f1_score(true_t, pred_t, average="macro")
                f1_micro = f1_score(true_t, pred_t, average="micro")

                f1_macro_list.append(f1_macro)
                f1_micro_list.append(f1_micro)

        total_loss += head_loss

    for opt in trainer.optimizer_list:
        opt.zero_grad()

    total_loss.backward()

    if trainer.args.dataset in ['KUHAR']:
        for model in trainer.model_list:
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.args.clip_grad_t)

    for opt in trainer.optimizer_list:
        opt.step()

    return trainer.model_list, loss_d, correct_d, eval_results_list, f1_macro_list, f1_micro_list, num_batches, num_samples

def valid_laservfl(self, ep):
    self.logger.info("=> Validation Accuracy...")
    for model in self.model_list:
        model.eval()
    # main task accuracy
    batch_loss_list = []
    total = 0
    correct = 0
    total_target = 0
    correct_target = 0
    pred = []
    true = []

    eval_results_list = []
    f1_macro_list = []
    f1_micro_list = []

    num_samples = len(self.valid_loader.dataset)
    num_batches = len(self.valid_loader)

    loss_d = {clients_l: 0.0 for clients_l in self.powerset}
    correct_d = {clients_l: 0.0 for clients_l in self.powerset}

    for step, (x_n, x_aug_1, x_aug_2, length, index, y, mask) in enumerate(self.valid_loader):
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

        if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
            y = y.squeeze()
            if self.args.dataset in ['MUSTARD']:
                y = torch.where(y > 0, 1, 0)
        if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
            y = y.float()  # for MSELoss

        mask.to(self.device)

        if torch.sum(mask).item() == 0:
            num_batches -= 1  # in practice, this batch is not used
            num_samples -= len(x_split_list[0])  # in practice, these samples are not used
            continue

        # observed_blocks = list(range(self.args.client_num))
        observed_blocks = torch.nonzero(mask).view(-1).tolist()

        # get the local model outputs
        # calculate embeddings
        embeddings = {}
        if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
            for i in observed_blocks:
                embeddings[i] = self.model_list[i + self.args.client_num]([x_split_list[i], length[i]])
        else:
            for i in observed_blocks:
                embeddings[i] = self.model_list[i + self.args.client_num](x_split_list[i])

        outputs_list = []
        for i, fusion_model in enumerate(self.model_list[0:self.args.client_num]):
            if i not in observed_blocks:
                continue
            sets_considered_by_head = [clients_l for clients_l in self.powerset if
                                       i in clients_l and set(clients_l).issubset(set(observed_blocks))]
            head_output = {}
            for num_clients_in_agg in range(1, len(observed_blocks) + 1):
                set_to_sample = [client_set for client_set in sets_considered_by_head if
                                 len(client_set) == num_clients_in_agg]
                for item in set_to_sample:
                    head_output[item] = fusion_model([embeddings[j] for j in item])
                # [sample] = random.sample(set_to_sample, 1)
                # head_output[sample] = fusion_model([embeddings[j] for j in sample])
            outputs_list.append(head_output)
        # outputs_list = [{clients_l: fusion_model([embeddings[j] for j in clients_l]) for clients_l in self.powerset if
        #             i in clients_l} for i, fusion_model in enumerate(self.model_list[0:self.args.client_num])]

        # calculate metrics
        total_loss = 0
        for outputs_per_task_d in outputs_list:
            head_loss = 0
            for clients_subset, outputs in outputs_per_task_d.items():
                loss = self.criterion(outputs, y)

                norm_constant = len(clients_subset)  # account for multiple heads

                # We divide the metrics by the number of predictors we have for this task, so that we get averaged metrics
                # (across the heads performing each task). This allows for metrics which are comparable to those of the decoupled approach.
                # calculate eval results
                loss_d[clients_subset] += (loss.item() / norm_constant)
                if self.args.dataset not in ['MMIMDB', 'MOSI', 'MOSEI', 'MUJOCO']:
                    predicted = outputs.argmax(1)
                    correct_d[clients_subset] += ((predicted == y).float().sum().item() / norm_constant)


                if self.args.dataset in ['MOSI', 'MOSEI']:
                    oute = outputs.detach().cpu().numpy().tolist()
                    pred.append(torch.LongTensor(oute))
                    pred_t = torch.cat(pred, 0)
                    true.append(y)
                    true_t = torch.cat(true, 0)
                    eval_results = eval_multi_affect(true_t, pred_t)
                    eval_results_list.append(eval_results)

                elif self.args.dataset in ['MUJOCO']:
                    pass
                elif self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH', 'UCIHAR',
                                           'KUHAR']:
                    pass
                elif self.args.dataset in ['MMIMDB', 'PTBXL']:
                    oute = outputs.detach().cpu()
                    pred.append(torch.sigmoid(oute).round())
                    pred_t = torch.cat(pred, 0)
                    true.append(y)
                    true_t = torch.cat(true, 0).cpu()

                    f1_macro = f1_score(true_t, pred_t, average="macro")
                    f1_micro = f1_score(true_t, pred_t, average="micro")

                    f1_macro_list.append(f1_macro)
                    f1_micro_list.append(f1_micro)

            total_loss += head_loss

    # main task accuracy
    if self.args.dataset in ['MOSI', 'MOSEI']:
        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
        current_loss_t = [loss / num_batches for loss in loss_l]
        current_loss = torch.stack(current_loss_t, dim=0).mean(dim=0)
        keys = eval_results_list[0].keys()
        eval_results = {key: round(torch.mean([d[key] for d in eval_results_list]), 4) for key in keys}
        self.logger.info(
            'Valid Epoch: {}, valid samples: {}, valid loss: {}, mean absolute error (mae): {:.4f}, valid main task accuracy (f1_score/acc2/acc3/acc5/acc7): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
                ep,
                len(self.valid_loader.dataset), current_loss,
                eval_results['mae'], eval_results['f1_score'],
                eval_results['acc_2'], eval_results['acc_3'], eval_results['acc_5'], eval_results['acc_7']))
        return eval_results

    elif self.args.dataset in ['MUJOCO']:
        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
        current_loss_t = [torch.tensor(loss) / num_batches for loss in loss_l]

        current_loss = torch.stack(current_loss_t, dim=0).mean(dim=0)
        self.logger.info(
            'Valid Epoch: {}, valid samples: {}, mean square error (mse): {:.4f}'.format(
                ep,
                len(self.valid_loader.dataset), current_loss))
        return current_loss  # mse value

    elif self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
        if num_batches == 0:
            return 0
        # valid_acc
        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
        correct_l = [correct_d[clients_l] for clients_l in self.powerset]

        current_loss_t = [loss / num_batches for loss in loss_l]
        valid_acc_t = [correct / num_samples for correct in correct_l]

        # get the mean acc for all tasks
        # current_loss = np.stack(current_loss_t).mean()
        # valid_acc = np.stack(valid_acc_t).mean()

        # get the acc for collaboration task
        current_loss = current_loss_t[-1]
        valid_acc = valid_acc_t[-1]
        self.logger.info(
            'Valid Epoch: {}, valid samples: {}, valid loss: {:.4f}, valid main task accuracy: {:.4f}'.format(ep,
                                                                                                              len(self.valid_loader.dataset),
                                                                                                              current_loss,
                                                                                                              valid_acc))

        return valid_acc

    elif self.args.dataset in ['MMIMDB', 'PTBXL']:
        if num_batches == 0:
            return 0, 0
        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
        current_loss_t = [loss / num_batches for loss in loss_l]
        current_loss = np.stack(current_loss_t).mean()

        f1_macro = np.stack(f1_macro_list).mean()
        f1_micro = np.stack(f1_micro_list).mean()
        self.logger.info(
            'Valid Epoch: {}, valid samples: {}, valid loss: {:.4f}, valid f1_macro: {:.4f}, valid f1_micro: {:.4f}'.format(
                ep,
                len(self.valid_loader.dataset),
                current_loss,
                f1_macro, f1_micro))

        return f1_macro, f1_micro

def test_laservfl(self, ep):
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

    eval_results_list = []
    f1_macro_list = []
    f1_micro_list = []

    num_samples = len(self.test_loader.dataset)
    num_batches = len(self.test_loader)

    loss_d = {clients_l: 0.0 for clients_l in self.powerset}
    correct_d = {clients_l: 0.0 for clients_l in self.powerset}

    for step, (x_n, x_aug_1, x_aug_2, length, index, y, mask) in enumerate(self.test_loader):
        x = x_n
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = x[i].to(self.device).float()
        else:
            x = x.to(self.device).float()
        y = y.to(self.device).long()
        # split data for vfl
        x_split_list = prepare_data_vfl(x, self.args)
        # get the local model outputs
        if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
            y = y.squeeze()
            if self.args.dataset in ['MUSTARD']:
                y = torch.where(y > 0, 1, 0)
        if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
            y = y.float()  # for MSELoss

        mask.to(self.device)

        if torch.sum(mask).item() == 0:
            num_batches -= 1  # in practice, this batch is not used
            num_samples -= len(x[0])  # in practice, these samples are not used
            continue

        observed_blocks = list(range(self.args.client_num))

        # get the local model outputs
        # calculate embeddings
        embeddings = {}
        if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
            for i in observed_blocks:
                embeddings[i] = model_list[i + self.args.client_num]([x_split_list[i], length[i]])
        else:
            for i in observed_blocks:
                embeddings[i] = model_list[i + self.args.client_num](x_split_list[i])

        outputs_list = [{clients_l: fusion_model([embeddings[j] for j in clients_l]) for clients_l in self.powerset if
                         i in clients_l} for i, fusion_model in enumerate(model_list[0:self.args.client_num])]

        # calculate metrics
        total_loss = 0
        for outputs_per_task_d in outputs_list:
            head_loss = 0
            for clients_subset, outputs in outputs_per_task_d.items():
                loss = self.criterion(outputs, y)

                norm_constant = 1.0
                norm_constant = norm_constant / len(clients_subset)  # account for multiple heads

                # if a given element (client) must be in a set, then we do n-1 choose k-1 instead
                # n, k = (len(observed_blocks) - 1, len(clients_subset) - 1)
                # norm_constant = norm_constant * math.comb(n, k)

                head_loss += loss * norm_constant

                # We divide the metrics by the number of predictors we have for this task, so that we get averaged metrics
                # (across the heads performing each task). This allows for metrics which are comparable to those of the decoupled approach.
                # calculate eval results
                loss_d[clients_subset] += loss.item() * norm_constant
                if self.args.dataset not in ['MMIMDB', 'MOSI', 'MOSEI', 'MUJOCO']:
                    predicted = outputs.argmax(1)
                    correct_d[clients_subset] += (predicted == y).int().sum().item() * norm_constant


                if self.args.dataset in ['MOSI', 'MOSEI']:
                    oute = outputs.detach().cpu().numpy().tolist()
                    pred.append(torch.LongTensor(oute))
                    pred_t = torch.cat(pred, 0)
                    true.append(y)
                    true_t = torch.cat(true, 0)
                    eval_results = eval_multi_affect(true_t, pred_t)
                    eval_results_list.append(eval_results)

                elif self.args.dataset in ['MUJOCO']:
                    pass
                elif self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH', 'UCIHAR',
                                           'KUHAR']:
                    pass
                elif self.args.dataset in ['MMIMDB', 'PTBXL']:
                    oute = outputs.detach().cpu()
                    pred.append(torch.sigmoid(oute).round())
                    pred_t = torch.cat(pred, 0)
                    true.append(y)
                    true_t = torch.cat(true, 0).cpu()

                    f1_macro = f1_score(true_t, pred_t, average="macro")
                    f1_micro = f1_score(true_t, pred_t, average="micro")

                    f1_macro_list.append(f1_macro)
                    f1_micro_list.append(f1_micro)

            total_loss += head_loss

    # main task accuracy
    if self.args.dataset in ['MOSI', 'MOSEI']:
        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
        current_loss_t = [loss / num_batches for loss in loss_l]
        current_loss = torch.stack(current_loss_t, dim=0).mean(dim=0)
        keys = eval_results_list[0].keys()
        eval_results = {key: round(torch.mean([d[key] for d in eval_results_list]), 4) for key in keys}
        self.logger.info(
            'Test Epoch: {}, test samples: {}, test loss: {}, mean absolute error (mae): {:.4f}, test main task accuracy (f1_score/acc2/acc3/acc5/acc7): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}'.format(
                ep,
                len(self.test_loader.dataset), current_loss,
                eval_results['mae'], eval_results['f1_score'],
                eval_results['acc_2'], eval_results['acc_3'], eval_results['acc_5'], eval_results['acc_7']))
        return eval_results

    elif self.args.dataset in ['MUJOCO']:
        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
        current_loss_t = [torch.tensor(loss) / num_batches for loss in loss_l]

        current_loss = torch.stack(current_loss_t, dim=0).mean(dim=0)
        self.logger.info(
            'Test Epoch: {}, test samples: {}, mean square error (mse): {:.4f}'.format(
                ep,
                len(self.test_loader.dataset), current_loss))
        return current_loss  # mse value

    elif self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH']:
        # test_acc
        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
        correct_l = [correct_d[clients_l] for clients_l in self.powerset]

        current_loss_t = [loss / num_batches for loss in loss_l]
        test_acc_t = [correct / num_samples for correct in correct_l]

        # get the mean acc for all tasks
        # current_loss = np.stack(current_loss_t).mean()
        # test_acc = np.stack(test_acc_t).mean()

        # get the acc for collaboration task
        current_loss = current_loss_t[-1]
        test_acc = test_acc_t[-1]
        self.logger.info(
            'Test Epoch: {}, test samples: {}, test loss: {:.4f}, test main task accuracy: {:.4f}'.format(ep,
                                                                                                          len(self.test_loader.dataset),
                                                                                                          current_loss,
                                                                                                          test_acc))

        return test_acc

    elif self.args.dataset in ['MMIMDB', 'PTBXL']:
        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
        current_loss_t = [loss / num_batches for loss in loss_l]
        current_loss = np.stack(current_loss_t).mean()

        f1_macro = np.stack(f1_macro_list).mean()
        f1_micro = np.stack(f1_micro_list).mean()
        self.logger.info(
            'Test Epoch: {}, test samples: {}, test loss: {:.4f}, test f1_macro: {:.4f}, test f1_micro: {:.4f}'.format(
                ep,
                len(self.test_loader.dataset),
                current_loss,
                f1_macro, f1_micro))

        return f1_macro, f1_micro
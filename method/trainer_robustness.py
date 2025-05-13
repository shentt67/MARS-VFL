from dataset.utils.utils import prepare_data_vfl, powerset_except_empty
from method.robustness.Base_Missing import base_missing, valid_missing, test_missing
from utils.utils import *
from utils.performance import f1_score, eval_multi_affect
import torch.nn.functional as F
from method.robustness.RVFLAug import rvflaug
from method.Base import base
from method.robustness.LEEFVFL import leefvfl, valid_leefvfl, test_leefvfl
from method.robustness.RVFLAlign import rvflalign, valid_rvflalign, test_rvflalign
from method.robustness.LASERVFL import laservfl, valid_laservfl, test_laservfl
from method.Base import base


class Trainer_Robustness:
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

        self.powerset = powerset_except_empty(self.args.client_num)

    def train(self):
        self.logger.info("=> Start Training Baseline...")
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
            for model in self.model_list:
                model.train()
            batch_loss_list = []
            total = 0
            correct = 0
            prede = []
            pred = []
            true = []

            self.logger.info("=> Start Training Epoch {}...".format(ep + 1))

            num_samples = len(self.train_loader.dataset)

            if self.args.method_name == 'laservfl':
                loss_d = {clients_l: 0.0 for clients_l in self.powerset}
                correct_d = {clients_l: 0.0 for clients_l in self.powerset}
                num_samples = len(self.train_loader.dataset)
                num_batches = len(self.train_loader)

            # input: (x, x_aug_1, x_aug_2, length, index, y, mask)
            for step, input in enumerate(self.train_loader):
                y = input[-2].to(self.device).long()
                if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                    y = y.squeeze()
                    if self.args.dataset in ['MUSTARD']:
                        y = torch.where(y > 0, 1, 0)
                if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
                    y = y.float()  # for MSELoss
                input_ = list(input)
                input_[-2] = y

                ########################## methods ############################
                if self.args.perturb_type == 'missing':
                    if self.args.method_name == 'base':
                        self.model_list, global_output, loss, num_samples = base_missing(self, self.model_list,
                                                                                                   input_,
                                                                                                   num_samples=num_samples)
                        if global_output is None:
                            continue
                    if self.args.method_name == 'leefvfl':
                        self.model_list, global_output, loss, num_samples = eval(self.args.method_name)(self, self.model_list, input_,
                                                                                                   num_samples)
                        if global_output is None:
                            continue
                        batch_loss_list.append(loss.item())
                    elif self.args.method_name == 'laservfl':
                        self.model_list, loss_d, _, eval_results_list, f1_macro_list, f1_micro_list, num_batches, num_samples = eval(
                            self.args.method_name)(
                            self, self.model_list, input_, loss_d, correct_d, pred, true, num_batches, num_samples)
                        if eval_results_list == -1:
                            continue
                if self.args.method_name in ['base', 'rvflaug', 'rvflalign']:
                    self.model_list, global_output, loss = eval(self.args.method_name)(self, self.model_list, input_)
                    batch_loss_list.append(loss.item())

                #############################################################

                # save the results in each step
                if self.args.dataset in ['MOSI', 'MOSEI']:
                    if self.args.method_name == 'laservfl':
                        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
                        current_loss_t = [loss / num_batches for loss in loss_l]
                        current_loss = torch.stack(current_loss_t, dim=0).mean(dim=0)
                        keys = eval_results_list[0].keys()
                        eval_results = {key: round(torch.mean([d[key] for d in eval_results_list]), 4) for key in keys}
                    else:
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
                    if self.args.method_name == 'laservfl':
                        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
                        current_loss_t = [torch.tensor(loss) / num_batches for loss in loss_l]

                        current_loss = torch.stack(current_loss_t, dim=0).mean(dim=0)
                    else:
                        current_loss = sum(batch_loss_list) / len(batch_loss_list)

                    if step % self.args.print_steps == 0:
                        self.logger.info(
                            'Epoch: {}, {}/{}: train loss (MSE): {:.4f}'.format(ep + 1, step + 1,
                                                                                len(self.train_loader),
                                                                                current_loss))

                elif self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                    if self.args.method_name == 'laservfl':
                        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
                        correct_l = [correct_d[clients_l] for clients_l in self.powerset]

                        current_loss_t = [loss / num_batches for loss in loss_l]
                        train_acc_t = [correct / num_samples for correct in correct_l]

                        current_loss = np.stack(current_loss_t).mean()
                        train_acc = np.stack(train_acc_t).mean()
                    else:
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
                    if self.args.method_name == 'laservfl':
                        loss_l = [loss_d[clients_l] for clients_l in self.powerset]
                        current_loss_t = [loss / num_batches for loss in loss_l]
                        current_loss = np.stack(current_loss_t).mean()

                        f1_macro = np.stack(f1_macro_list).mean()
                        f1_micro = np.stack(f1_micro_list).mean()
                    else:
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

            if self.args.dataset in ['MOSI', 'MOSEI']:
                eval_results = self.valid(ep + 1)
                valid_acc = eval_results['acc_2']
                if valid_acc > best_acc:
                    # best accuracy
                    best_acc = valid_acc
                    best_eval_results = eval_results
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
                    self.test(ep)
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
                    '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, f1_micro: {:.4f}, f1_macro: {:.4f}'.format(
                        ep + 1,
                        no_change,
                        best_epoch, best_f1_micro, best_f1_macro))
                if no_change == self.args.early_stop:
                    self.test(ep + 1)
                    return

            if (ep + 1) % 10 == 0:
                self.test(ep + 1)

    def valid(self, ep):
        if self.args.method_name == 'base' and self.args.perturb_type == 'missing':
            return valid_missing(self, ep)
        if self.args.method_name == 'laservfl':
            return valid_laservfl(self, ep)
        elif self.args.method_name == 'leefvfl':
            return valid_leefvfl(self, ep)
        elif self.args.method_name == 'rvflalign':
            return valid_rvflalign(self, ep)
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

        if self.args.method_name == 'laservfl':
            loss_d = {clients_l: 0.0 for clients_l in self.powerset}
            correct_d = {clients_l: 0.0 for clients_l in self.powerset}
            num_samples = len(self.train_loader.dataset)
            num_batches = len(self.train_loader)

        for step, (x, x_aug_1, x_aug_2, length, index, y, mask) in enumerate(self.valid_loader):
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

            # global model backward
            if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                y = y.squeeze()
                if self.args.dataset in ['MUSTARD']:
                    y = torch.where(y > 0, 1, 0)
            if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
                y = y.float()  # for MSELoss
            loss = self.criterion(global_output, y)
            if self.args.method_name not in ['laservfl']:
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
                'Valid Epoch: {}, valid samples: {}, valid loss: {:.4f}, valid f1_micro: {:.4f}, valid f1_macro: {:.4f}'.format(
                    ep,
                    len(self.valid_loader.dataset),
                    current_loss,
                    f1_micro, f1_macro))

            return f1_micro, f1_macro

    def test(self, ep):
        if self.test_loader is None:
            return
        if self.args.method_name == 'base' and self.args.perturb_type == 'missing':
            return test_missing(self, ep)
        if self.args.method_name == 'laservfl':
            return test_laservfl(self, ep)
        if self.args.method_name == 'leefvfl':
            return test_leefvfl(self, ep)
        if self.args.method_name == 'rvflalign':
            return test_rvflalign(self, ep)
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
        for step, (x, x_aug_1, x_aug_2, length, index, y, mask) in enumerate(self.test_loader):
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

            # global model backward
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
                'Test Epoch: {}, test samples: {}, test loss: {:.4f}, test f1_micro: {:.4f}, test f1_macro: {:.4f}'.format(
                    ep,
                    len(self.test_loader.dataset),
                    current_loss,
                    f1_micro, f1_macro))

            return f1_micro, f1_macro

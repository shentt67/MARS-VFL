from dataset.utils.utils import prepare_data_vfl
from utils.utils import *
from utils.performance import f1_score, eval_multi_affect
from method.Base import base
from method.efficiency.FedBCD import fedbcd
from method.efficiency.CELUVFL import celuvfl
from method.efficiency.CVFL import cvfl
from method.efficiency.EFVFL import efvfl
import time

class Trainer_Efficiency:
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
        self.communication_cost = 0

        # save results
        self.metric_per_ep = []
        self.metric_test_per_ep = []
        self.communication_cost_per_ep = []
        self.execution_time_per_ep = []

    def train(self):
        self.logger.info("=> Start Training Baseline...")
        epoch_loss_list = []
        model_list = self.model_list
        best_acc = 0
        best_epoch = 0
        no_change = 0
        valid_acc = 0
        valid_mae = 0
        best_mae = float("inf")
        best_eval_results = {}
        best_mse = float("inf")
        eval_results = {}
        valid_f1_macro = 0
        valid_f1_micro = 0
        best_f1_macro = 0
        best_f1_micro = 0

        if self.checkpoint:
            best_acc = self.checkpoint['best_acc']

        # train and update
        self.num_total_comms = 0
        cache = []
        previous_batches = []
        for i in range(self.args.client_num):
            cache.append({})
            previous_batches.append([])
        cache_global = {}

        self.state_list = [None for i in range(self.args.client_num)]

        # train and update
        for ep in range(self.args.start_epoch, self.args.epoch):
            start_time = time.time()
            model_list = [model.train() for model in model_list]
            batch_loss_list = []
            total = 0
            correct = 0
            prede = []
            pred = []
            true = []

            self.logger.info("=> Start Training Epoch {}...".format(ep + 1))

            # input: (x, length, index, y, mask)
            for step, input in enumerate(self.train_loader):
                y = input[-2].to(self.device).long()
                if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                    y = y.squeeze()
                    if self.args.dataset in ['MUSTARD']:
                        y = torch.where(y > 0, 1, 0)
                if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
                    y = y.float()
                input_ = list(input)
                input_[-2] = y

                # methods
                if self.args.method_name in ['base', 'fedbcd', 'cvfl']:
                    model_list, global_output, loss, self.communication_cost = eval(self.args.method_name)(self, model_list, input_, self.communication_cost)
                elif self.args.method_name in ['celuvfl']:
                    model_list, global_output, loss, self.communication_cost = eval(self.args.method_name)(self, model_list, input_, cache, previous_batches, cache_global, self.communication_cost)
                elif self.args.method_name in ['efvfl']:
                    model_list, global_output, loss, self.communication_cost, self.state_list = eval(self.args.method_name)(self, model_list, input_, self.state_list, ep, self.communication_cost)

                batch_loss_list.append(loss.item())

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
                            'Epoch: {}, {}/{}: train loss: {:.4f}, mean absolute error (mae): {:.4f}, train main task accuracy (f1_score/acc2/acc3/acc5/acc7): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}, communication cost (MB): {}'.format(
                                ep + 1,
                                step + 1,
                                len(self.train_loader), current_loss,
                                eval_results['mae'], eval_results['f1_score'],
                                eval_results['acc_2'], eval_results['acc_3'], eval_results['acc_5'],
                                eval_results['acc_7'], self.communication_cost))

                elif self.args.dataset in ['MUJOCO']:
                    current_loss = sum(batch_loss_list) / len(batch_loss_list)

                    if step % self.args.print_steps == 0:
                        self.logger.info(
                            'Epoch: {}, {}/{}: train loss (MSE): {:.4f}, communication cost (MB): {}'.format(ep + 1, step + 1,
                                                                                len(self.train_loader),
                                                                                current_loss, self.communication_cost))

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
                            'Epoch: {}, {}/{}: train loss: {:.4f}, train main task accuracy: {:.4f}, communication cost (MB): {}'.format(ep + 1,
                                                                                                            step + 1,
                                                                                                            len(self.train_loader),
                                                                                                            current_loss,
                                                                                                            train_acc, self.communication_cost))

                elif self.args.dataset in ['MMIMDB', 'PTBXL']:
                    # calculate the training accuracy
                    current_loss = sum(batch_loss_list) / len(batch_loss_list)
                    oute = global_output.detach().cpu()
                    pred.append(torch.sigmoid(oute).round())
                    pred_t = torch.cat(pred, 0)
                    true.append(y)
                    true_t = torch.cat(true, 0).cpu()

                    f1_micro = f1_score(true_t, pred_t, average="micro")
                    f1_macro = f1_score(true_t, pred_t, average="macro")

                    if step % self.args.print_steps == 0:
                        self.logger.info(
                            'Epoch: {}, {}/{}: train loss: {:.4f}, train f1_micro: {:.4f}, train f1_macro: {:.4f}, communication cost (MB): {}'.format(
                                ep + 1,
                                step + 1,
                                len(self.train_loader),
                                current_loss,
                                f1_micro, f1_macro, self.communication_cost))

            epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
            epoch_loss_list.append(epoch_loss)
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
                    filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar')
                    torch.save(state, filename)
                else:
                    no_change += 1
                self.logger.info(
                    '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, mean absolute error (mae): {}, '
                    'accuracy (f1_score/acc2/acc3/acc5/acc7): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}, total communication cost (MB): {}'.format(
                        ep + 1,
                        no_change,
                        best_epoch, best_eval_results['mae'], best_eval_results['f1_score'], best_eval_results['acc_2'],
                        best_eval_results['acc_3'], best_eval_results['acc_5'], best_eval_results['acc_7'], self.communication_cost))
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
                    filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar')
                    torch.save(state, filename)
                else:
                    no_change += 1
                self.logger.info(
                    '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, mean square error (mse): {}, total communication cost (MB): {}'.format(
                        ep + 1,
                        no_change,
                        best_epoch, best_mse, self.communication_cost))
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
                    filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar')
                    torch.save(state, filename)
                else:
                    no_change += 1
                self.logger.info(
                    '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, accuracy: {:.4f}, total communication cost (MB): {}'.format(
                        ep + 1,
                        no_change,
                        best_epoch, best_acc, self.communication_cost))
                if no_change == self.args.early_stop:
                    self.test(ep + 1)
                    return

            elif self.args.dataset in ['MMIMDB', 'PTBXL']:
                valid_f1_micro, valid_f1_macro = self.valid(ep + 1)
                if valid_f1_micro > best_f1_micro:
                    # best accuracy
                    best_f1_micro = valid_f1_micro
                    best_f1_macro = valid_f1_macro
                    no_change = 0
                    best_epoch = ep + 1
                    # save model
                    self.logger.info("=> Save best model...")
                    state = {
                        'epoch': ep + 1,
                        'best_f1_micro': best_f1_micro,
                        'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
                        'optimizer': [self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))],
                    }
                    filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar')
                    torch.save(state, filename)
                else:
                    no_change += 1
                self.logger.info(
                    '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, f1_micro: {:.4f}, f1_macro: {:.4f},  total communication cost (MB): {}'.format(
                        ep + 1,
                        no_change,
                        best_epoch, best_f1_micro, best_f1_macro,  self.communication_cost))
                if no_change == self.args.early_stop:
                    self.test(ep + 1)
                    return

            # test_metric = self.test(ep + 1)
            metric_ep = 0
            # if test_metric is not None:
            #     if isinstance(test_metric, dict):
            #         self.metric_test_per_ep.append(test_metric['f1_score'])
            #     elif isinstance(test_metric, float):
            #         self.metric_test_per_ep.append(test_metric)
            #     elif isinstance(test_metric, tuple):
            #         self.metric_test_per_ep.append(test_metric[0])
            if valid_mae != 0:
                metric_ep = valid_mae
            elif valid_acc != 0:
                metric_ep = valid_acc
            elif valid_f1_micro != 0:
                metric_ep = valid_f1_micro

            end_time = time.time()
            execution_time = end_time - start_time
            self.metric_per_ep.append(metric_ep)
            self.communication_cost_per_ep.append(self.communication_cost)
            self.execution_time_per_ep.append(execution_time)

        test_metric = self.test(ep + 1)
        if test_metric is not None:
            if isinstance(test_metric, dict):
                self.metric_test_per_ep.append(test_metric['f1_score'])
            elif isinstance(test_metric, float):
                self.metric_test_per_ep.append(test_metric)
            elif isinstance(test_metric, tuple):
                self.metric_test_per_ep.append(test_metric[0])


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
                'Valid Epoch: {}, valid samples: {}, valid loss: {:.4f}, valid f1_micro: {:.4f}, valid f1_macro: {:.4f}'.format(
                    ep,
                    len(self.valid_loader.dataset),
                    current_loss,
                    f1_micro, f1_macro))

            return f1_micro, f1_macro

    def test(self, ep):
        if self.test_loader is None:
            return None
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
                'Test Epoch: {}, test samples: {}, test loss: {:.4f}, test f1_micro: {:.4f}, test f1_macro: {:.4f}'.format(
                    ep,
                    len(self.test_loader.dataset),
                    current_loss,
                    f1_micro, f1_macro))

            return f1_micro, f1_macro

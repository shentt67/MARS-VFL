from dataset.utils.utils import prepare_data_vfl
import torch

from utils.performance import eval_multi_affect, f1_score
from utils.utils import *

def base_missing(trainer, model_list, input, num_samples=None):
    global_output = None
    loss = None
    if torch.sum(input[-1]).item() < trainer.args.client_num:
        num_samples -= len(input[0])  # in practice, these samples are not used
        return model_list, global_output, loss, num_samples
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
    # get the local model outputs
    if trainer.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
        for i in range(trainer.args.client_num):
            if not input[-1][i]:
                continue
            local_output_list.append(model_list[i + 1]([x_split_list[i], input[-4][i]]))
    else:
        for i in range(trainer.args.client_num):
            if not input[-1][i]:
                continue
            local_output_list.append(model_list[i + 1](x_split_list[i]))

    global_output = model_list[0](local_output_list)

    loss = trainer.criterion(global_output, y)
    for opt in trainer.optimizer_list:
        opt.zero_grad()

    loss.backward()

    if trainer.args.dataset in ['KUHAR']:
        for model in model_list:
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.args.clip_grad_t)

    for opt in trainer.optimizer_list:
        opt.step()

    return model_list, global_output, loss, num_samples

def valid_missing(self, ep):
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

    for step, (x, x_aug_1, x_aug_2, length, index, y, mask) in enumerate(self.valid_loader):
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = x[i].to(self.device).float()
        else:
            x = x.to(self.device).float()
        y = y.to(self.device).long()
        # split data for vfl
        x_split_list = prepare_data_vfl(x, self.args)
        if torch.sum(mask).item() < self.args.client_num:
            if self.args.dataset in ['MOSI', 'MOSEI']:
                global_output = torch.tensor(np.random.uniform(-5, 5, (
                x_split_list[0].shape[0], 1))).to(self.device)
            elif self.args.dataset in ['MUSTARD', 'URFUNNY']:
                global_output = torch.tensor(np.random.uniform(-5, 5, (
                    x_split_list[0].shape[0], 2))).to(self.device)
            elif self.args.dataset in ['MUJOCO']:
                global_output = torch.tensor(np.random.uniform(-5, 5, (y.shape))).to(self.device)
            elif self.args.dataset in ['PTBXL']:
                global_output = torch.tensor(np.random.uniform(-5, 5, (
                x_split_list[0].shape[0], 5))).to(self.device)
            elif self.args.dataset in ['MIMIC']:
                global_output = torch.tensor(np.random.uniform(-5, 5, (
                x_split_list[0].shape[0], 2))).to(self.device)
            else:
                global_output = torch.tensor(np.random.uniform(-5, 5, (x_split_list[0].shape[0], self.model_list[0].classifier.out_features))).to(self.device)
        else:
            local_output_list = []
            # get the local model outputs
            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                for i in range(self.args.client_num):
                    if not mask[i]:
                        continue
                    local_output_list.append(model_list[i + 1]([x_split_list[i], length[i]]))
            else:
                for i in range(self.args.client_num):
                    if not mask[i]:
                        continue
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

def test_missing(self, ep):
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
        if torch.sum(mask).item() < self.args.client_num:
            if self.args.dataset in ['MOSI', 'MOSEI']:
                global_output = torch.tensor(np.random.uniform(-5, 5, (
                    x_split_list[0].shape[0], 1))).to(self.device)
            elif self.args.dataset in ['MUSTARD', 'URFUNNY']:
                global_output = torch.tensor(np.random.uniform(-5, 5, (
                    x_split_list[0].shape[0], 2))).to(self.device)
            elif self.args.dataset in ['MUJOCO']:
                global_output = torch.tensor(np.random.uniform(-5, 5, (y.shape))).to(self.device)
            elif self.args.dataset in ['PTBXL']:
                global_output = torch.tensor(np.random.uniform(-5, 5, (
                x_split_list[0].shape[0], 5))).to(self.device)
            elif self.args.dataset in ['MIMIC']:
                global_output = torch.tensor(np.random.uniform(-5, 5, (
                x_split_list[0].shape[0], 2))).to(self.device)
            else:
                global_output = torch.tensor(np.random.uniform(-5, 5, (
                x_split_list[0].shape[0], self.model_list[0].classifier.out_features))).to(self.device)
        else:
            local_output_list = []
            # get the local model outputs
            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                for i in range(self.args.client_num):
                    if not mask[i]:
                        continue
                    local_output_list.append(model_list[i + 1]([x_split_list[i], length[i]]))
            else:
                for i in range(self.args.client_num):
                    if not mask[i]:
                        continue
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
import torch.nn as nn

from dataset.utils.utils import prepare_data_vfl
from utils.utils import *
from utils.performance import f1_score, eval_multi_affect
from utils import possible_defenses
from model.common_models import GRU, MLP, Concat
    
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 600),
            nn.LayerNorm(600),
            nn.ReLU(),

            nn.Linear(600, 200),
            nn.LayerNorm(200),
            nn.ReLU(),

            nn.Linear(200, 100),
            nn.LayerNorm(100),
            nn.ReLU(),

            nn.Linear(100, output_dim),
            nn.Sigmoid()
        )
        self.args = args

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = self.backbone(x)
        return x


class Trainer_GRNA:
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

        self.grad_args = [None for _ in range(self.args.client_num)]
    def train(self):
        self.train_model()
        self.train_grna()

    def train_grna(self):
        self.logger.info("=> Start Training Generator...")
        model_list = self.model_list
        for model in model_list:
            model.eval()

        data_iter = iter(self.train_loader)
        x_n, length, index, y, mask = data_iter.next()
        x = x_n
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = x[i].to(self.device).float()
        else:
            x = x.to(self.device).float()
        # split data for vfl
        x_split_list = prepare_data_vfl(x, self.args)
        input_dim = 0
        for i in range(len(self.train_loader.dataset.data.shape) - 1):
            input_dim += self.train_loader.dataset.data.shape[i + 1]
            output_dim = x_split_list[self.args.attack_target_client].shape[-1]
  
        self.netG = Generator(input_dim, output_dim, self.args).to(self.device)
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.args.lr)

        best_validate_loss = torch.inf

        no_change = 0

        for epoch in range(self.args.attack_train_epochs_grna):
            losses = []
            self.netG.train()
            for step, (x_n, length, index, y, mask) in enumerate(self.train_loader):
                x = x_n
                if isinstance(x, list):
                    for i in range(len(x)):
                        x[i] = x[i].to(self.device).float()
                else:
                    x = x.to(self.device).float()
                y = y.to(self.device).long()

                if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                    y = y.squeeze()
                    if self.args.dataset in ['MUSTARD']:
                        y = torch.where(y > 0, 1, 0)
                if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
                    y = y.float()  # for MSELoss

                # split data for vfl
                x_split_list = prepare_data_vfl(x, self.args)

                # use random noise as the target input
                noise = torch.randn(x_split_list[self.args.attack_target_client].size()).to(self.device)
                # generate target feature
                generator_input = x_split_list
                generator_input[self.args.attack_target_client] = noise
                generated_data_target = self.netG(torch.cat(generator_input, dim=-1))

                # var loss, keep stable
                unknown_var_loss = 0.0
                for i in range(generated_data_target.size(0)):
                    unknown_var_loss = unknown_var_loss + (generated_data_target[i].var())

                local_output_list = []
                # get the local model outputs
                if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                    for i in range(self.args.client_num):
                        local_output_list.append(model_list[i + 1]([x_split_list[i], length[i]]))
                else:
                    for i in range(self.args.client_num):
                        local_output_list.append(model_list[i + 1](x_split_list[i]))

                # the ground truth of global output
                global_output_truth = model_list[0](local_output_list)

                # the global output with generated features
                global_input_list = local_output_list
                if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                    global_input_list[self.args.attack_target_client] = model_list[self.args.attack_target_client + 1]([generated_data_target, length[self.args.attack_target_client]])
                else:
                    global_input_list[self.args.attack_target_client] = model_list[self.args.attack_target_client + 1](generated_data_target)
                global_output_generated = model_list[0](global_input_list)

                loss = ((global_output_generated - global_output_truth.detach()) ** 2).sum() + self.args.UnknownVarLambda * unknown_var_loss

                self.optimizerG.zero_grad()
                loss.backward()
                losses.append(loss.detach())
                self.optimizerG.step()

                if step % self.args.print_steps == 0:
                    self.logger.info(
                        'Epoch: {}, {}/{}: train loss: {:.4f}, L2 norm of generated features: {}, L2 norm of original vector: {}'.format(
                            epoch + 1,
                            step + 1,
                            len(self.train_loader),
                            sum(losses) / len(losses),
                            (generated_data_target ** 2).sum(), (x_split_list[self.args.attack_target_client] ** 2).sum()))

            # Validate and test the attack performance
            # Validate
            validate_loss = self.validate_attack(epoch)
            if validate_loss < best_validate_loss or epoch == 0:
                # best accuracy
                best_validate_loss = validate_loss
                no_change = 0
                best_epoch = epoch + 1
                # save model
                self.logger.info("=> Save best generator...")
                state = {
                    'epoch': epoch + 1,
                    'best_validate_loss': best_validate_loss,
                    'state_dict': self.netG.state_dict(),
                    'optimizer': self.optimizerG,
                }
                filename = os.path.join(self.args.results_dir, 'best_generator_checkpoint.pth.tar')
                torch.save(state, filename)
            else:
                no_change += 1
            self.logger.info(
                '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, validate mse: {}'.format(
                    epoch + 1,
                    no_change,
                    best_epoch, validate_loss))
            if no_change == self.args.early_stop:
                self.test(epoch + 1)
                return
            # Test
            if (epoch + 1) % 10 == 0:
                self.test_attack(epoch)

    def train_model(self):
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
            model_list = [model.train() for model in model_list]
            batch_loss_list = []
            total = 0
            correct = 0
            prede = []
            pred = []
            true = []

            self.logger.info("=> Start Training Epoch {}...".format(ep + 1))

            for step, (x_n, length, index, y, mask) in enumerate(self.train_loader):
                # for j in self.train_loader:
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


                global_input_list = []
                for i in range(self.args.client_num):
                    global_input_list.append(torch.tensor([], requires_grad=True))
                    global_input_list[i].data = local_output_list[i].data
                global_output = model_list[0](global_input_list)

                # global model backward
                if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                    y = y.squeeze()
                    if self.args.dataset in ['MUSTARD']:
                        y = torch.where(y > 0, 1, 0)
                if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
                    y = y.float()  # for MSELoss
                #loss = self.criterion(global_output, y)


                loss_framework = update_top_model_one_batch(optimizer=self.optimizer_list[0],
                                                                 model=self.model_list,
                                                                 output=global_output,
                                                                 batch_target=y,
                                                                 loss_func=self.criterion, args=self.args)
                if self.args.dataset in ['KUHAR']:
                    for model in self.model_list:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_grad_t)

                # read grad of: input of top model(also output of bottom models), which will be used as bottom model's target
                grad_output_list = [input_tensor_top.grad for input_tensor_top in global_input_list]
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
                # for opt in self.optimizer_list:
                #     opt.zero_grad()

                # loss.backward()
                for opt in self.optimizer_list:
                    opt.step()
                batch_loss_list.append(loss_framework.item())
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

    def validate_attack(self, epoch):
        model_list = self.model_list
        model_list = [model.eval() for model in model_list]
        self.netG.eval()
        losses_valid = []
        losses_valid_mse = []
        losses_valid_pf = []
        losses_random = []
        losses_random_pf = []
        mse = torch.nn.MSELoss()
        for step, (x_n, length, index, y, mask) in enumerate(self.valid_loader):
            # for j in self.train_loader:
            x = x_n
            if isinstance(x, list):
                for i in range(len(x)):
                    x[i] = x[i].to(self.device).float()
            else:
                x = x.to(self.device).float()

            # split data for vfl
            x_split_list = prepare_data_vfl(x, self.args)

            # use random noise as the target input
            noise = torch.randn(x_split_list[self.args.attack_target_client].size()).to(self.device)
            # generate target feature
            generator_input = x_split_list
            generator_input[self.args.attack_target_client] = noise
            generated_data_target = self.netG(torch.cat(generator_input, dim=-1))

            # var loss, keep stable
            unknown_var_loss = 0.0
            for i in range(generated_data_target.size(0)):
                unknown_var_loss = unknown_var_loss + (generated_data_target[i].var())

            local_output_list = []
            # get the local model outputs
            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                for i in range(self.args.client_num):
                    local_output_list.append(model_list[i + 1]([x_split_list[i], length[i]]))
            else:
                for i in range(self.args.client_num):
                    local_output_list.append(model_list[i + 1](x_split_list[i]))

            # the ground truth of global output
            global_output_truth = model_list[0](local_output_list)

            # the global output with generated features
            global_input_list = local_output_list
            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                global_input_list[self.args.attack_target_client] = model_list[self.args.attack_target_client + 1](
                    [generated_data_target, length[self.args.attack_target_client]])
            else:
                global_input_list[self.args.attack_target_client] = model_list[self.args.attack_target_client + 1](
                    generated_data_target)
            global_output_generated = model_list[0](global_input_list)

            loss = ((global_output_generated - global_output_truth.detach()) ** 2).sum() + self.args.UnknownVarLambda * unknown_var_loss

            losses_valid.append(loss.detach())

            loss_pf = self.lossPerFeature(x_split_list[self.args.attack_target_client], generated_data_target)

            losses_valid_pf.append(loss_pf)

            loss_valid_mse = mse(x_split_list[self.args.attack_target_client], generated_data_target).item()

            losses_valid_mse.append(loss_valid_mse)

            randomguess = torch.rand_like(x_split_list[self.args.attack_target_client])
            loss_random = mse(x_split_list[self.args.attack_target_client], randomguess).item()
            loss_random_pf = self.lossPerFeature(x_split_list[self.args.attack_target_client], randomguess)
            losses_random.append(loss_random)
            losses_random_pf.append(loss_random_pf)

        self.logger.info(
            'Validate Epoch: {}, validate samples: {}, validate loss: {}, validate mse loss: {}, validate mse loss per feature: {}, validate random mse loss: {}, validate random mse loss per feature: {}'.format(
                epoch + 1,
                len(self.valid_loader.dataset),
                sum(losses_valid) / len(losses_valid), sum(losses_valid_mse) / len(losses_valid_mse),
                (sum(losses_valid_pf) / len(losses_valid_pf)).mean(), sum(losses_random) / len(losses_random),
                (sum(losses_random_pf) / len(losses_random_pf)).mean()))

        return (sum(losses_valid_pf) / len(losses_valid_pf)).mean()

    def test_attack(self, epoch):
        if self.test_loader is None:
            return
        self.logger.info("=> Test Accuracy...")
        model_list = self.model_list
        model_list = [model.eval() for model in model_list]
        check_path = os.path.join(self.args.results_dir, 'best_generator_checkpoint.pth.tar')
        self.logger.info("=> loading best test checkpoint '{}'".format(check_path))
        checkpoint_test = torch.load(check_path, map_location=self.device)
        netG = self.netG.eval()
        if self.args.dataset in ['URFUNNY']:
            netG.load_state_dict(checkpoint_test['state_dict'])
        else:   
            netG.load_state_dict(checkpoint_test['state_dict'])
        self.logger.info("=> loaded test checkpoint '{}' (epoch {})"
                         .format(check_path, checkpoint_test['epoch']))
        losses_test = []
        losses_test_mse = []
        losses_test_pf = []
        losses_random = []
        losses_random_pf = []
        mse = torch.nn.MSELoss()
        for step, (x_n, length, index, y, mask) in enumerate(self.test_loader):
            # for j in self.train_loader:
            x = x_n
            if isinstance(x, list):
                for i in range(len(x)):
                    x[i] = x[i].to(self.device).float()
            else:
                x = x.to(self.device).float()

            # split data for vfl
            x_split_list = prepare_data_vfl(x, self.args)

            # use random noise as the target input
            noise = torch.randn(x_split_list[self.args.attack_target_client].size()).to(self.device)
            # generate target feature
            generator_input = x_split_list
            generator_input[self.args.attack_target_client] = noise
            generated_data_target = netG(torch.cat(generator_input, dim=-1))

            # var loss, keep stable
            unknown_var_loss = 0.0
            for i in range(generated_data_target.size(0)):
                unknown_var_loss = unknown_var_loss + (generated_data_target[i].var())

            local_output_list = []
            # get the local model outputs
            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                for i in range(self.args.client_num):
                    local_output_list.append(model_list[i + 1]([x_split_list[i], length[i]]))
            else:
                for i in range(self.args.client_num):
                    local_output_list.append(model_list[i + 1](x_split_list[i]))

            # the ground truth of global output
            global_output_truth = model_list[0](local_output_list)

            # the global output with generated features
            global_input_list = local_output_list
            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                global_input_list[self.args.attack_target_client] = model_list[self.args.attack_target_client + 1](
                    [generated_data_target, length[self.args.attack_target_client]])
            else:
                global_input_list[self.args.attack_target_client] = model_list[self.args.attack_target_client + 1](
                    generated_data_target)
            global_output_generated = model_list[0](global_input_list)

            loss = ((global_output_generated - global_output_truth.detach()) ** 2).sum() + self.args.UnknownVarLambda * unknown_var_loss

            losses_test.append(loss.detach())

            loss_pf = self.lossPerFeature(x_split_list[self.args.attack_target_client], generated_data_target)

            losses_test_pf.append(loss_pf)

            loss_test_mse = mse(x_split_list[self.args.attack_target_client], generated_data_target).item()

            losses_test_mse.append(loss_test_mse)

            randomguess = torch.rand_like(x_split_list[self.args.attack_target_client])
            loss_random = mse(x_split_list[self.args.attack_target_client], randomguess).item()
            loss_random_pf = self.lossPerFeature(x_split_list[self.args.attack_target_client], randomguess)
            losses_random.append(loss_random)
            losses_random_pf.append(loss_random_pf)

        self.logger.info(
            'Test Epoch: {}, test samples: {}, test loss: {}, test mse loss: {}, test mse loss per feature: {}, test random mse loss: {}, test random mse loss per feature: {}'.format(
                epoch + 1,
                len(self.test_loader.dataset),
                sum(losses_test) / len(losses_test), sum(losses_test_mse) / len(losses_test_mse), (sum(losses_test_pf) / len(losses_test_pf)).mean(), sum(losses_random) / len(losses_random), sum(losses_random_pf) / len(losses_random_pf)))

    def lossPerFeature(self, input, target):
        res = []
        for i in range(input.size(1)):
            loss = ((input[:, i] - target[:, i]) ** 2).mean().item()
            res.append(loss)
        return np.array(res)
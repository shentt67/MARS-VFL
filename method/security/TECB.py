from dataset.utils.utils import prepare_data_vfl
from utils.utils import *
from utils.performance import f1_score, eval_multi_affect

from utils.utils import keep_predict_loss
from utils import possible_defenses

class Trainer_TECB:
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
        stone1 = self.args.stone1  # 50 int(args.epochs * 0.5)
        stone2 = self.args.stone2  # 85 int(args.epochs * 0.8)

        self.scheduler_list = [torch.optim.lr_scheduler.MultiStepLR(opt,
                                                                    milestones=[stone1, stone2],
                                                                    gamma=self.args.step_gamma) for opt in
                               self.optimizer_list]

        self.defense_ppdl = args.ppdl
        self.defense_gc = args.gc
        self.defense_lap_noise = args.lap_noise
        self.defense_multistep_grad = args.multistep_grad
        self.criterion_local = keep_predict_loss
        self.grad_args = [None for _ in range(self.args.client_num)]
    def train(self):
        self.train_narcissus()
        self.train_model()
        self.train_poisoning()

    def train_poisoning(self):
        model_list = self.model_list
        model_list = [model.train() for model in model_list]

        best_trade_off = 0
        best_acc = 0
        best_asr = 0
        best_epoch = 0
        no_change = 0

        # create the disturbs of backdoor
        data_iter = iter(self.train_loader)
        x_n, length, index, y, mask = next(data_iter)#.next()
        x = x_n
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = x[i].to(self.device).float()
        else:
            x = x.to(self.device).float()

        bottom_criterion = keep_predict_loss

        for epoch in range(self.args.poison_epochs, self.args.epoch):
            total = 0
            correct = 0
            batch_loss = []
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
                mask = torch.from_numpy(np.isin(index, self.selected_indices)).reshape(-1)
                num_masked = mask.sum().item()

                # select the samples that are not the target class
                non_mask_indices = torch.where(~mask)[0]
                random_indices = torch.randperm(non_mask_indices.size(0))[
                                 :num_masked]  # select the same number of the auxiliary samples
                x_split_list[self.args.attack_client][non_mask_indices[random_indices]] += self.delta.detach().clone()

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
                    if self.args.client_num == self.args.attack_client:
                        random_grad = torch.randn(num_masked, x_split_list[self.args.attack_client].shape[1]).to(
                            self.device)
                        global_input_list[i].data = random_grad
                    else:
                        global_input_list[i].data = local_output_list[i].data

                global_output = model_list[0](global_input_list)

                loss_framework = update_top_model_one_batch(optimizer=self.optimizer_list[0],
                                                            model=self.model_list[0],
                                                            output=global_output,
                                                            batch_target=y,
                                                            loss_func=self.criterion)

                batch_loss.append(loss_framework.item())

                grad_output_list = [input_tensor_top.grad for input_tensor_top in global_input_list]

                # Target grad replacement, replace grad of the poison samples with the target class samples
                grad_output_list[self.args.attack_client][non_mask_indices[random_indices]] = self.args.corruption_amp * \
                                                                                              grad_output_list[
                                                                                                  self.args.attack_client][
                                                                                                  mask]
                 # privacy preserving deep learning
                if self.defense_ppdl:
                    for i in range(self.args.client_num):
                        possible_defenses.dp_gc_ppdl(epsilon=1.8, sensitivity=1,
                                                     layer_grad_list=[grad_output_list[i]],
                                                     theta_u=self.args.ppdl_theta_u, gamma=0.001, tau=0.0001,get_grad_layer_id=self.grad_args[i])
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


                # for i in range(self.args.client_num):
                #     update_bottom_model_one_batch(optimizer=self.optimizer_list[i + 1],
                #                                   model=self.model_list[i + 1],
                #                                   output=local_output_list[i],
                #                                   batch_target=grad_output_list[i],
                #                                   loss_func=bottom_criterion)

                for i in range(self.args.client_num):
                    self.scheduler_list[i].step()

                # calculate training accuracy including the backdoor samples
                _, predicted = global_output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                train_acc = correct / total
                # print the batch info
                if step % self.args.print_steps == 0:
                    self.logger.info(
                        'Epoch: {}, {}/{}: train loss: {:.4f}, train accuracy (including backdoor samples): {:.4f}'.format(
                            epoch + 1,
                            step + 1,
                            len(self.train_loader),
                            sum(batch_loss) / len(batch_loss), train_acc))

            valid_acc, valid_asr = self.valid(epoch + 1)
            valid_trade_off = (valid_acc + valid_asr) / 2
            if valid_trade_off > best_trade_off:
                # best accuracy
                best_trade_off = valid_trade_off
                best_asr = valid_asr
                best_acc = valid_acc
                no_change = 0
                best_epoch = epoch + 1
                # save model
                self.logger.info("=> Save best model...")
                state = {
                    'epoch': epoch + 1,
                    'best_trade_off': best_trade_off,
                    'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
                    'optimizer': [self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))],
                }
                filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar')
                torch.save(state, filename)
            else:
                no_change += 1
            self.logger.info(
                '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, best trade off: {:.4f}, attack success rate of best epoch: {:.4f}, the main task accuracy: {:.4f}'.format(
                    epoch + 1,
                    no_change,
                    best_epoch, best_trade_off, best_asr, best_acc))
            if no_change == self.args.early_stop:
                self.test(epoch + 1)
                return

            if (epoch + 1) % 10 == 0:
                self.test(epoch + 1)

    def train_narcissus(self):
        model_list = self.model_list
        model_list = [model.train() for model in model_list]

        best_trade_off = 0
        best_acc = 0
        best_asr = 0
        best_epoch = 0
        no_change = 0

        # create the disturbs of backdoor
        data_iter = iter(self.train_loader)
        x_n, length, index, y, mask = next(data_iter)#.next()
        x = x_n
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = x[i].to(self.device).float()
        else:
            x = x.to(self.device).float()
        # split data for vfl
        x_split_list = prepare_data_vfl(x, self.args)

        delta = torch.zeros((1, x_split_list[self.args.attack_client].shape[-1]), device=self.device)
        delta.requires_grad_(True)

        target_label = self.args.target_label

        target_indices = np.where(np.array(self.train_loader.dataset.targets) == target_label)[0]

        self.selected_indices = np.random.choice(target_indices, self.args.poison_num, replace=False)

        bottom_criterion = keep_predict_loss

        for epoch in range(self.args.backdoor):
            batch_loss = []
            total = 0
            correct = 0
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

                # the disturbs to be optimized
                batch_delta = torch.zeros_like(x_split_list[self.args.attack_client]).to(self.device)
                mask = torch.from_numpy(np.isin(index, self.selected_indices)).reshape(-1)
                batch_delta[mask] = delta.clone().detach()
                batch_delta.requires_grad_()

                # get the local model outputs
                local_output_list = []
                if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                    for i in range(self.args.client_num):
                        local_output_list.append(model_list[i + 1]([x_split_list[i], length[i]]))
                else:
                    for i in range(self.args.client_num):
                        local_output_list.append(model_list[i + 1](x_split_list[i]))

                # the global input with delta
                if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                    local_output_list[self.args.attack_client] = model_list[self.args.attack_client + 1](
                        [x_split_list[self.args.attack_client] + batch_delta, length[self.args.attack_client]])
                else:
                    local_output_list[self.args.attack_client] = model_list[self.args.attack_client + 1](
                        x_split_list[self.args.attack_client] + batch_delta)

                global_input_list = []
                for i in range(self.args.client_num):
                    global_input_list.append(torch.tensor([], requires_grad=True))
                    global_input_list[i].data = local_output_list[i].data

                global_output = model_list[0](global_input_list)

                loss_framework = update_top_model_one_batch(optimizer=self.optimizer_list[0],
                                                            model=self.model_list[0],
                                                            output=global_output,
                                                            batch_target=y,
                                                            loss_func=self.criterion)

                batch_loss.append(loss_framework.item())

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
                        #pass
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
                                                  loss_func=bottom_criterion)

                for i in range(self.args.client_num):
                    self.scheduler_list[i].step()

                # update delta
                batch_delta_grad = batch_delta.grad[mask]
                grad_sign = batch_delta_grad.detach().mean(dim=0).sign()
                delta = delta - grad_sign * self.args.alpha_delta
                delta = torch.clamp(delta, -self.args.eps, self.args.eps)

                self.delta = delta

                # calculate training accuracy including the backdoor samples
                _, predicted = global_output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                train_acc = correct / total
                # print the batch info
                if step % self.args.print_steps == 0:
                    self.logger.info(
                        'Epoch: {}, {}/{}: train loss: {:.4f}, train accuracy (including backdoor samples): {:.4f}'.format(
                            epoch + 1,
                            step + 1,
                            len(self.train_loader),
                            sum(batch_loss) / len(batch_loss), train_acc))

            valid_acc, valid_asr = self.valid(epoch + 1)
            # if test_acc > best_acc:
            valid_trade_off = (valid_acc + valid_asr) / 2
            if valid_trade_off > best_trade_off:
                # best accuracy
                best_trade_off = valid_trade_off
                best_asr = valid_asr
                best_acc = valid_acc
                no_change = 0
                best_epoch = epoch + 1
                # save model
                self.logger.info("=> Save best model...")
                state = {
                    'epoch': epoch + 1,
                    'best_trade_off': best_trade_off,
                    'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
                    'optimizer': [self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))],
                }
                filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar'.format(epoch + 1))
                torch.save(state, filename)
            else:
                no_change += 1
            self.logger.info(
                '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, best trade off: {:.4f}, attack success rate of best epoch: {:.4f}, the main task accuracy: {:.4f}'.format(
                    epoch + 1,
                    no_change,
                    best_epoch, best_trade_off, best_asr, best_acc))
            if no_change == self.args.early_stop:
                self.test(epoch + 1)
                return

            if (epoch + 1) % 10 == 0:
                self.test(epoch + 1)

    def train_model(self):
        self.logger.info("=> Start Training Baseline...")
        epoch_loss_list = []
        model_list = self.model_list
        best_acc = 0
        best_trade_off = 0
        best_asr = 0
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
        for ep in range(self.args.backdoor, self.args.poison_epochs):
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
                # loss = self.criterion(global_output, y)
                # for opt in self.optimizer_list:
                #     opt.zero_grad()

                # loss.backward()

                # for opt in self.optimizer_list:
                #     opt.step()

                loss_framework = update_top_model_one_batch(optimizer=self.optimizer_list[0],
                                                                 model=self.model_list,
                                                                 output=global_output,
                                                                 batch_target=y,
                                                                 loss_func=self.criterion, args=self.args)

                grad_output_list = [input_tensor_top.grad for input_tensor_top in global_input_list]
                # privacy preserving deep learning
                if self.defense_ppdl:
                    for i in range(self.args.client_num):
                        possible_defenses.dp_gc_ppdl(epsilon=1.8, sensitivity=1,
                                                     layer_grad_list=[grad_output_list[i]],
                                                     theta_u=self.args.ppdl_theta_u, gamma=0.001, tau=0.0001,get_grad_layer_id=self.grad_args[i])
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
                valid_acc, valid_asr = self.valid(ep + 1)
                # if test_acc > best_acc:
                valid_trade_off = (valid_acc + valid_asr) / 2
                if valid_trade_off > best_trade_off:
                    # best accuracy
                    best_trade_off = valid_trade_off
                    best_asr = valid_asr
                    best_acc = valid_acc
                    no_change = 0
                    best_epoch = ep + 1
                    # save model
                    self.logger.info("=> Save best model...")
                    state = {
                        'epoch': ep + 1,
                        'best_trade_off': best_trade_off,
                        'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
                        'optimizer': [self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))],
                    }
                    filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar'.format(ep + 1))
                    torch.save(state, filename)
                else:
                    no_change += 1
                self.logger.info(
                    '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, best trade off: {:.4f}, attack success rate of best epoch: {:.4f}, the main task accuracy: {:.4f}'.format(
                        ep + 1,
                        no_change,
                        best_epoch, best_trade_off, best_asr, best_acc))
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
        total_asr = 0
        correct_asr = 0
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

            ######################## calculate the attack performance
            batch_delta = torch.zeros_like(x_split_list[self.args.attack_client]).to(self.device)
            batch_delta[:] = self.delta.clone().detach()
            local_output_list_backdoor = []
            # the global input with delta
            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                for i in range(self.args.client_num):
                    local_output_list_backdoor.append(model_list[i + 1]([x_split_list[i], length[i]]))
            else:
                for i in range(self.args.client_num):
                    local_output_list_backdoor.append(model_list[i + 1](x_split_list[i]))

            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                local_output_list_backdoor[self.args.attack_client] = model_list[self.args.attack_client + 1](
                    [x_split_list[self.args.attack_client] + batch_delta,
                     length[self.args.attack_client]])
            else:
                local_output_list_backdoor[self.args.attack_client] = model_list[self.args.attack_client + 1](
                    x_split_list[self.args.attack_client] + batch_delta)

            global_output_backdoor = model_list[0](local_output_list_backdoor)
            ##################################

            if self.args.dataset in ['MUJOCO']:
                batch_loss_list.append(loss.item())
            else:
                batch_loss_list.append(loss.item())
                # calculate the validation accuracy
                if self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH', 'UCIHAR', 'KUHAR']:
                    _, predicted = global_output.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()

                    # calculate attack performance
                    _, predicted_backdoor = global_output_backdoor.max(1)
                    total_asr += (y != self.args.target_label).float().sum()
                    correct_asr += (
                                predicted_backdoor[y != self.args.target_label] == self.args.target_label).float().sum()

                    # calculate target class accuracy
                    total_target += (y == self.args.target_label).float().sum()
                    correct_target += predicted.eq(y)[y == self.args.target_label].float().sum().item()

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

            valid_asr = correct_asr / total_asr
            valid_target = correct_target / total_target
            self.logger.info(
                'Valid Epoch: {}, valid samples: {}, valid loss: {:.4f}, valid target class accuracy: {:.4f}, valid trade off: {:.4f}, valid attack success rate: {:.4f}, valid main task accuracy: {:.4f}'.format(
                    ep,
                    len(self.valid_loader.dataset),
                    current_loss,
                    valid_target, (valid_asr + valid_acc) / 2, valid_asr, valid_acc))

            return valid_acc, valid_asr

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
        check_path = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar')
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
        total_asr = 0
        correct_asr = 0
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

            ######################## calculate the attack performance
            batch_delta = torch.zeros_like(x_split_list[self.args.attack_client]).to(self.device)
            batch_delta[index] = self.delta.detach().clone()
            local_output_list_backdoor = []
            # the global input with delta
            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                for i in range(self.args.client_num):
                    #length[i].to(x_split_list[i].device())
                    local_output_list_backdoor.append(model_list[i + 1]([x_split_list[i], length[i]]))
            else:
                for i in range(self.args.client_num):
                    local_output_list_backdoor.append(model_list[i + 1](x_split_list[i]))

            if self.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                local_output_list_backdoor[self.args.attack_client] = model_list[self.args.attack_client + 1](
                    [x_split_list[self.args.attack_client] + batch_delta,
                     length[self.args.attack_client]])
            else:
                local_output_list_backdoor[self.args.attack_client] = model_list[self.args.attack_client + 1](
                    x_split_list[self.args.attack_client] + batch_delta)

            global_output_backdoor = model_list[0](local_output_list_backdoor)
            ##################################

            if self.args.dataset in ['MUJOCO']:
                batch_loss_list.append(loss.item())
            else:
                batch_loss_list.append(loss.item())
                # calculate the test accuracy
                if self.args.dataset in ['MIMIC', 'NUSWIDE', 'URFUNNY', 'MUSTARD', 'VISIONTOUCH']:
                    _, predicted = global_output.max(1)
                    total += y.size(0)
                    correct += predicted.eq(y).sum().item()

                    # calculate attack performance
                    _, predicted_backdoor = global_output_backdoor.max(1)
                    total_asr += (y != self.args.target_label).float().sum()
                    correct_asr += (
                            predicted_backdoor[y != self.args.target_label] == self.args.target_label).float().sum()

                    # calculate target class accuracy
                    total_target += (y == self.args.target_label).float().sum()
                    correct_target += predicted.eq(y)[y == self.args.target_label].float().sum().item()

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

            test_asr = correct_asr / total_asr
            test_target = correct_target / total_target
            self.logger.info(
                'Test Epoch: {}, test samples: {}, test loss: {:.4f}, test target class accuracy: {:.4f}, test trade off: {:.4f}, test attack success rate: {:.4f}, test main task accuracy: {:.4f}'.format(
                    ep,
                    len(self.test_loader.dataset),
                    current_loss,
                    test_target, (test_asr + test_acc) / 2, test_asr, test_acc))

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

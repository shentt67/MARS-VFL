from dataset.utils.utils import prepare_data_vfl
from utils.utils import *
from utils.performance import f1_score, eval_multi_affect
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from utils import possible_defenses

class Trainer_Norm_Direction_Score:
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
        self.criterion_local = keep_predict_loss
        # adversarial options
        self.defense_ppdl = args.ppdl
        self.defense_gc = args.gc
        self.defense_lap_noise = args.lap_noise
        self.defense_multistep_grad = args.multistep_grad
        self.grad_args = [None for _ in range(self.args.client_num)]

    def update_all_norm_leak_auc(self, norm_leak_auc_dict, grad_list, y):
        for (key, grad) in zip(norm_leak_auc_dict.keys(), grad_list):
            # flatten each example's grad to one-dimensional
            grad = torch.reshape(grad, shape=(grad.shape[0], -1))

            ###### auc #######

            predicted_value = torch.norm(grad, p=2, dim=-1, keepdim=False)
            predicted_value = predicted_value.view(-1)

            if torch.sum(y) == 0:  # no positive examples in this batch
                return 0, 0

            val_max = torch.max(predicted_value)
            val_min = torch.min(predicted_value)
            predicted_value = (predicted_value - val_min + 1e-16) / (val_max - val_min + 1e-16)
            predicted_value = predicted_value.cpu().numpy()
            y = y.cpu().numpy()
            auc = roc_auc_score(y_true=y, y_score=predicted_value)

            ###### acc #######
            predicted_label = np.where(predicted_value > 0.5, 1, 0).reshape(-1)
            # print('predicted_label:',predicted_label[:5])
            # print('y:',y[:5])
            acc = (predicted_label == y).sum() / len(y)

            return acc, auc

    def update_all_cosine_leak_auc(self, cosine_leak_auc_dict, grad_list, pos_grad_list, y):

        def cosine_similarity(A, B):
            # Normalize each row (dim=1)
            row_normalized_A = F.normalize(A, p=2, dim=1)
            row_normalized_B = F.normalize(B, p=2, dim=1)
            # Compute cosine similarity (matrix multiplication)
            cosine_matrix = torch.matmul(row_normalized_A, row_normalized_B.T)  # transpose second matrix
            return cosine_matrix

        # If no positive examples in the batch
        if torch.sum(y) == 0:
            return 0, 0

        for (key, grad, pos_grad) in zip(cosine_leak_auc_dict.keys(), grad_list, pos_grad_list):
            # Flatten each example's grad to one-dimensional (reshape to (N, -1))
            grad = grad.view(grad.shape[0], -1)
            pos_grad = pos_grad.view(pos_grad.shape[0], -1)

            # Compute cosine similarity between grad and pos_grad
            predicted_value = cosine_similarity(grad,
                                                pos_grad).cpu().numpy()  # Convert to NumPy array after computation

            # Convert predicted values to binary labels based on threshold
            predicted_label = (predicted_value > 0).astype(int).reshape(-1)

            # Convert y to NumPy array
            _y = y.cpu().numpy()

            # Compute accuracy
            acc = (predicted_label == _y).sum() / len(_y)

            # Reshape predicted value (flatten to 1D)
            predicted_value = predicted_value.reshape(-1)

            # Min-Max normalization of predicted values
            val_max = np.max(predicted_value)
            val_min = np.min(predicted_value)
            pred = (predicted_value - val_min + 1e-16) / (val_max - val_min + 1e-16)

            # Compute AUC
            auc = roc_auc_score(y_true=_y, y_score=pred)  # Convert to NumPy array before passing

            return acc, auc

    def train(self):
        if self.args.dataset not in ['URFUNNY', 'MUSTARD', 'MIMIC', 'VISIONTOUCH']:
            raise Exception('Unsupported dataset. Just for binary classification.')
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
            norm_leak_acc_list = []
            norm_leak_auc_list = []
            cosine_leak_acc_list = []
            cosine_leak_auc_list = []

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

                # global model backward
                if self.args.dataset in ['URFUNNY', 'MUSTARD', 'NUSWIDE', 'MIMIC', 'VISIONTOUCH', 'UCIHAR',
                                         'KUHAR']:
                    y = y.squeeze()
                    if self.args.dataset in ['MUSTARD']:
                        y = torch.where(y > 0, 1, 0)
                if self.args.dataset in ['MUJOCO', 'MMIMDB', 'PTBXL']:
                    y = y.float()  # for MSELoss

                global_input_list = []
                for i in range(self.args.client_num):
                    global_input_list.append(torch.tensor([], requires_grad=True))
                    global_input_list[i].data = local_output_list[i].data

                global_output = model_list[0](global_input_list)

                # global model backward
                loss = update_top_model_one_batch(optimizer=self.optimizer_list[0],
                                                  model=self.model_list[0],
                                                  output=global_output,
                                                  batch_target=y,
                                                  loss_func=self.criterion)

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


                grad_output_list_attacker = grad_output_list[self.args.attack_client]
                tf_pred_a_gradients_clone = torch.tensor(grad_output_list_attacker.cpu().numpy())
                tf_true_label = y

                # --------Norm scoring--------

                norm_leak_acc, norm_leak_auc = self.update_all_norm_leak_auc(
                    norm_leak_auc_dict={'only': ''},
                    grad_list=[tf_pred_a_gradients_clone],  #
                    y=tf_true_label)

                # --------Direction scoring--------
                # find a positive gradient
                # pos_idx = np.random.randint(len(y))
                pos_idx = np.random.choice(np.where(y.cpu().numpy() == 1)[0], size=1)[0]
                # while y[pos_idx] != torch.tensor(1):
                #     pos_idx += 1
                #     if pos_idx >= len(y):
                #         break

                cosine_leak_acc, cosine_leak_auc = self.update_all_cosine_leak_auc(
                    cosine_leak_auc_dict={'only': ''},
                    grad_list=[tf_pred_a_gradients_clone],
                    pos_grad_list=[tf_pred_a_gradients_clone[pos_idx:pos_idx + 1]], y=tf_true_label)

                for i in range(self.args.client_num):
                    update_bottom_model_one_batch(optimizer=self.optimizer_list[i + 1],
                                                  model=self.model_list[i + 1],
                                                  output=local_output_list[i],
                                                  batch_target=grad_output_list[i],
                                                  loss_func=self.criterion_local)

                batch_loss_list.append(loss.item())

                # print attack results
                attack_results = [norm_leak_acc, norm_leak_auc, cosine_leak_acc, cosine_leak_auc]
                norm_leak_acc_list.append(norm_leak_acc)
                norm_leak_auc_list.append(norm_leak_auc)
                cosine_leak_acc_list.append(cosine_leak_acc)
                cosine_leak_auc_list.append(cosine_leak_auc)
                # print(attack_results)

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
                            'Epoch: {}, {}/{}: train loss: {:.4f}, mean absolute error (mae): {:.4f}, train main task accuracy (f1_score/acc2/acc3/acc5/acc7): {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}, norm leak acc/norm leak auc/direction leak acc/direction leak auc: {}'.format(
                                ep + 1,
                                step + 1,
                                len(self.train_loader), current_loss,
                                eval_results['mae'], eval_results['f1_score'],
                                eval_results['acc_2'], eval_results['acc_3'], eval_results['acc_5'],
                                eval_results['acc_7'], attack_results))

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
                            'Epoch: {}, {}/{}: train loss: {:.4f}, train main task accuracy: {:.4f}, norm leak acc/norm leak auc/direction leak acc/direction leak auc: {}'.format(
                                ep + 1,
                                step + 1,
                                len(self.train_loader),
                                current_loss,
                                train_acc, attack_results))

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
                            'Epoch: {}, {}/{}: train loss: {:.4f}, train f1_macro: {:.4f}, train f1_micro: {:.4f}, norm leak acc/norm leak auc/direction leak acc/direction leak auc: {}'.format(
                                ep + 1,
                                step + 1,
                                len(self.train_loader),
                                current_loss,
                                f1_macro, f1_micro, attack_results))

            ep_norm_acc = np.mean(norm_leak_acc_list)
            ep_norm_auc = np.mean(norm_leak_auc_list)
            ep_cosine_acc = np.mean(cosine_leak_acc_list)
            ep_cosine_auc = np.mean(cosine_leak_auc_list)

            self.logger.info(
                'Epoch: {}, {}/{}: mean training norm leak acc/norm leak auc/direction leak acc/direction leak auc: {}'.format(
                    ep + 1,
                    len(self.train_loader),
                    len(self.train_loader), [ep_norm_acc, ep_norm_auc, ep_cosine_acc, ep_cosine_auc]))


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

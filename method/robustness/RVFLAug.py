from dataset.utils.utils import prepare_data_vfl
import torch.nn.functional as F
import torch

def rvflaug(trainer, model_list, input):
    x = input[0]
    x_aug_1 = input[1]
    x_aug_2 = input[2]
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = x[i].to(trainer.device).float()
            x_aug_1[i] = x_aug_1[i].to(trainer.device).float()
            x_aug_2[i] = x_aug_2[i].to(trainer.device).float()
    else:
        x = x.to(trainer.device).float()
        x_aug_1 = x_aug_1.to(trainer.device).float()
        x_aug_2 = x_aug_2.to(trainer.device).float()

    y = input[-2]
    # split data for vfl
    x_split_list = prepare_data_vfl(x, trainer.args)
    x_aug_1_split_list = prepare_data_vfl(x_aug_1, trainer.args)
    x_aug_2_split_list = prepare_data_vfl(x_aug_2, trainer.args)

    local_output_list = []
    local_output_aug_1_list = []
    local_output_aug_2_list = []
    # get the local model outputs
    if trainer.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
        for i in range(trainer.args.client_num):
            local_output_list.append(model_list[i + 1]([x_split_list[i], input[-4][i]]))
            local_output_aug_1_list.append(model_list[i + 1]([x_aug_1_split_list[i], input[-4][i]]))
            local_output_aug_2_list.append(model_list[i + 1]([x_aug_2_split_list[i], input[-4][i]]))
    else:
        for i in range(trainer.args.client_num):
            local_output_list.append(model_list[i + 1](x_split_list[i]))
            local_output_aug_1_list.append(model_list[i + 1](x_aug_1_split_list[i]))
            local_output_aug_2_list.append(model_list[i + 1](x_aug_2_split_list[i]))

    global_output = model_list[0](local_output_list)
    global_output_aug_1 = model_list[0](local_output_aug_1_list)
    global_output_aug_2 = model_list[0](local_output_aug_2_list)

    loss = trainer.criterion(global_output, y)

    p_clean, p_aug1, p_aug2 = F.softmax(
        global_output, dim=1), F.softmax(
        global_output_aug_1, dim=1), F.softmax(
        global_output_aug_2, dim=1)
    # Clamp mixture distribution to avoid exploding KL divergence
    p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
    loss += trainer.args.balance * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                                 F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                                 F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

    for opt in trainer.optimizer_list:
        opt.zero_grad()

    loss.backward()

    if trainer.args.dataset in ['KUHAR']:
        for model in model_list:
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.args.clip_grad_t)

    for opt in trainer.optimizer_list:
        opt.step()

    return model_list, global_output, loss
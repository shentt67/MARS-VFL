from dataset.utils.utils import prepare_data_vfl
import torch
from utils.utils import *

def base(trainer, model_list, input, communication_cost=None):
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
            local_output_list.append(model_list[i + 1]([x_split_list[i], input[-4][i]]))
            if communication_cost is not None:
                # send embeddings to active client in each epoch
                communication_cost += get_tensor_size(local_output_list[i])
                # retain gradients for calculation
                local_output_list[i].requires_grad_(True)
                local_output_list[i].retain_grad()
    else:
        for i in range(trainer.args.client_num):
            local_output_list.append(model_list[i + 1](x_split_list[i]))
            if communication_cost is not None:
                # send embeddings to active client in each epoch
                communication_cost += get_tensor_size(local_output_list[i])

    if communication_cost is not None:
        # except the active client
        communication_cost -= get_tensor_size(local_output_list[trainer.args.active_client])

    global_output = model_list[0](local_output_list)

    loss = trainer.criterion(global_output, y)
    for opt in trainer.optimizer_list:
        opt.zero_grad()

    loss.backward()

    if trainer.args.dataset in ['KUHAR']:
        for model in trainer.model_list:
            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.args.clip_grad_t)

    # send gradients to passive clients
    for i in range(trainer.args.client_num):
        if i != 0 and communication_cost is not None:
            # the gradient is the same size as the embeddings
            communication_cost += get_tensor_size(local_output_list[i])

    for opt in trainer.optimizer_list:
        opt.step()

    if communication_cost is not None:
        return model_list, global_output, loss, communication_cost
    return model_list, global_output, loss
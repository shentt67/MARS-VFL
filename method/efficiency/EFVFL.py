from dataset.utils.utils import prepare_data_vfl
import torch
import math
from utils.utils import *

def ef_compression(state, x, indices, epoch, trainer):
    indices = np.array(indices.reshape(-1))
    if isinstance(x, tuple):
        x = x[0]
        x_size = np.array(x.shape)
        x_size[0] = len(trainer.train_loader.dataset.data)
        if state is None:
            state = torch.zeros(tuple(x_size), requires_grad=False, device=x.device)

        state_detached = state.detach()
        updated_state = state_detached.clone()
        if epoch == 0:
            updated_state[indices] = compress(x, trainer.args.compression_ratio)
        else:
            updated_state[indices] = state_detached[indices] + compress(x - state_detached[indices],
                                                                        trainer.args.compression_ratio)
        state = updated_state.detach()
        return state, (updated_state[indices], [])
    x_size = np.array(x.shape)
    x_size[0] = len(trainer.train_loader.dataset.data)
    if state is None:
        state = torch.zeros(tuple(x_size), requires_grad=False, device=x.device)

    state_detached = state.detach()
    updated_state = state_detached.clone()
    if epoch == 0:
        updated_state[indices] = compress(x, trainer.args.compression_ratio)
    else:
        updated_state[indices] = state_detached[indices] + compress(x - state_detached[indices], trainer.args.compression_ratio)
    state = updated_state.detach()

    return state, updated_state[indices]

def efvfl(trainer, model_list, input, state_list, ep, communication_cost):
    trainer.local_iterations = trainer.args.local_iterations
    x = input[0]
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = x[i].to(trainer.device).float()
    else:
        x = x.to(trainer.device).float()
    y = input[-2]
    # split data for vfl
    x_split_list = prepare_data_vfl(x, trainer.args)

    for q in range(trainer.local_iterations):
        if q == 0:
            # update all clients
            local_output_list = []
            # get the local model outputs
            if trainer.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                for i in range(trainer.args.client_num):
                    st, cr = ef_compression(state_list[i], model_list[i + 1]([x_split_list[i], input[-4][i]]), input[-3], ep, trainer)
                    state_list[i] = st
                    local_output_list.append(cr)
                    # send embeddings to active client in each epoch
                    communication_cost += get_tensor_size(local_output_list[i], trainer.args.compression_ratio)
            else:
                for i in range(trainer.args.client_num):
                    st, cr = ef_compression(state_list[i], model_list[i + 1](x_split_list[i]), input[-3], ep, trainer)
                    state_list[i] = st
                    local_output_list.append(cr)
                    # send embeddings to active client in each epoch
                    communication_cost += get_tensor_size(local_output_list[i], trainer.args.compression_ratio)

            # except the active client
            communication_cost -= get_tensor_size(local_output_list[trainer.args.active_client], trainer.args.compression_ratio)

            global_output = model_list[0](local_output_list)
            loss = trainer.criterion(global_output, y)
            for opt in trainer.optimizer_list:
                opt.zero_grad()

            # loss.backward()

            pred_gradients_list = []
            pred_gradients_list_clone = []
            for i in range(trainer.args.client_num):
                if isinstance(local_output_list[i], tuple):
                    pred_gradients_list.append(
                        torch.autograd.grad(loss, local_output_list[i][0], retain_graph=True, create_graph=True))
                else:
                    pred_gradients_list.append(
                        torch.autograd.grad(loss, local_output_list[i], retain_graph=True, create_graph=True))
                pred_gradients_list_clone.append(pred_gradients_list[i][0].detach().clone())

                # send gradients to passive clients in each epoch
                if i != 0:
                    communication_cost += get_tensor_size(pred_gradients_list_clone[i], trainer.args.compression_ratio)

                if isinstance(local_output_list[i], tuple):
                    weights_grad_a = torch.autograd.grad(
                        local_output_list[i][0],
                        trainer.model_list[i + 1].parameters(),
                        grad_outputs=pred_gradients_list_clone[i],
                        retain_graph=True
                    )
                    for w, g in zip(trainer.model_list[i + 1].parameters(), weights_grad_a):
                        if w.requires_grad:
                            w.grad = g.detach()
                else:
                    weights_grad_a = torch.autograd.grad(
                        local_output_list[i],
                        trainer.model_list[i + 1].parameters(),
                        grad_outputs=pred_gradients_list_clone[i],
                        retain_graph=True
                    )
                    for w, g in zip(trainer.model_list[i + 1].parameters(), weights_grad_a):
                        if w.requires_grad:
                            w.grad = g.detach()

            _gradients = torch.autograd.grad(loss, global_output, retain_graph=True)
            _gradients_clone = _gradients[0].detach().clone()
            torch.autograd.set_detect_anomaly(True)
            weights_grad_a = torch.autograd.grad(
                global_output,
                trainer.model_list[0].parameters(),
                grad_outputs=_gradients_clone,
                retain_graph=True
            )
            for w, g in zip(trainer.model_list[0].parameters(), weights_grad_a):
                if w.requires_grad:
                    w.grad = g.detach()

            for model in trainer.model_list:
                torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.args.clip_grad_t)

            for opt in trainer.optimizer_list:
                opt.step()
        else: # local updates
            for opt in trainer.optimizer_list:
                opt.zero_grad()
            local_output_list = []
            if trainer.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                for i in range(trainer.args.client_num):
                    t = model_list[i + 1]([x_split_list[i], input[-4][i]])
                    # do not compress in local updates
                    local_output_list.append(t)
            else:
                for i in range(trainer.args.client_num):
                    t = model_list[i + 1](x_split_list[i])
                    # do not compress in local updates
                    local_output_list.append(t)
            for i in range(trainer.args.client_num):
                if isinstance(local_output_list[i], tuple):
                    weights_grad_a = torch.autograd.grad(
                        local_output_list[i][0],
                        trainer.model_list[i + 1].parameters(),
                        grad_outputs=pred_gradients_list_clone[i],
                        retain_graph=True
                    )
                    for w, g in zip(trainer.model_list[i + 1].parameters(), weights_grad_a):
                        if w.requires_grad:
                            w.grad = g.detach()
                else:
                    weights_grad_a = torch.autograd.grad(
                        local_output_list[i],
                        trainer.model_list[i + 1].parameters(),
                        grad_outputs=pred_gradients_list_clone[i],
                        retain_graph=True
                    )
                    for w, g in zip(trainer.model_list[i + 1].parameters(), weights_grad_a):
                        if w.requires_grad:
                            w.grad = g.detach()

            global_output = model_list[0](local_output_list)
            loss = trainer.criterion(global_output, y)
            _gradients = torch.autograd.grad(loss, global_output, retain_graph=True)
            _gradients_clone = _gradients[0].detach().clone()
            torch.autograd.set_detect_anomaly(True)
            weights_grad_a = torch.autograd.grad(
                global_output,
                trainer.model_list[0].parameters(),
                grad_outputs=_gradients_clone,
                retain_graph=True
            )
            for w, g in zip(trainer.model_list[0].parameters(), weights_grad_a):
                if w.requires_grad:
                    w.grad = g.detach()

            for model in trainer.model_list:
                torch.nn.utils.clip_grad_norm_(model.parameters(), trainer.args.clip_grad_t)

            for opt in trainer.optimizer_list:
                opt.step()

    return model_list, global_output, loss, communication_cost, state_list
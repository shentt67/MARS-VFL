from dataset.utils.utils import prepare_data_vfl
import torch

from utils.utils import get_tensor_size


def fedbcd(trainer, model_list, input, communication_cost):
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
                    local_output_list.append(model_list[i + 1]([x_split_list[i], input[-4][i]]))
                    # send embeddings to active client in each epoch
                    communication_cost += get_tensor_size(local_output_list[i])
            else:
                for i in range(trainer.args.client_num):
                    local_output_list.append(model_list[i + 1](x_split_list[i]))
                    # send embeddings to active client in each epoch
                    communication_cost += get_tensor_size(local_output_list[i])

            # except the active client
            communication_cost -= get_tensor_size(local_output_list[trainer.args.active_client])

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
                    communication_cost += get_tensor_size(pred_gradients_list_clone[i])

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
        else:  # FedBCD: additional local iterations without communication
            for opt in trainer.optimizer_list:
                opt.zero_grad()
            local_output_list = []
            if trainer.args.dataset in ['MOSI', 'MOSEI', 'URFUNNY', 'MUSTARD']:
                for i in range(trainer.args.client_num):
                    local_output_list.append(model_list[i + 1]([x_split_list[i], input[-4][i]]))
            else:
                for i in range(trainer.args.client_num):
                    local_output_list.append(model_list[i + 1](x_split_list[i]))
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



    return model_list, global_output, loss, communication_cost
import math
import os

import torch
import wandb

from model.criterion_lgcnlstmattn import get_criterion 
from data_loader.dataloader_lgcnlstmattn import get_GES_loaders

from model.metric_lgcnlstmattn import get_metric
from model.optimizer_lgcnlstmattn import get_optimizer
from model.scheduler_lgcnlstmattn import get_scheduler
from datetime import datetime

from model import model_lgcnlstmattn #GESLSTMATTN

def get_model(adj_matrix, **args):

    model = model_lgcnlstmattn.GESLSTMATTN(adj_matrix, **args)


    return model
            
            
def run_with_vaild_loss(args, train_data, valid_data, model):
    train_loader, valid_loader = get_GES_loaders(args['data_loader']['args'], train_data, valid_data)
    
    # only when using warmup scheduler
    args['scheduler']['total_steps'] = int(math.ceil(len(train_loader.dataset) / args['trainer']['batch_size'])) * (
        args['trainer']['n_epochs']
    )
    args['scheduler']['warmup_steps'] = args['scheduler']['total_steps'] // 10

    optimizer = get_optimizer(model, **args['optimizer'])
    scheduler = get_scheduler(optimizer, **args['scheduler'])

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args['trainer']['n_epochs']):

        print(f"Start Training: Epoch {epoch + 1}")

        ### TRAIN
        train_auc, train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, args
        )

        ### VALID
        auc, acc, loss = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        wandb.log(
            {
                "train_loss_epoch": train_loss,
                "valid_loss_epoch": loss,
                "train_auc_epoch": train_auc,
                "valid_auc_epoch": auc,
                "train_acc_epoch": train_acc,
                "valid_acc_epoch": acc,
            }
        )
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                },
                os.path.join(args['trainer']['save_dir'], args['name']),
                "model.pt"
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args['trainer']['patience']:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args['patience']}"
                )
                break

        # scheduler
        if args['scheduler']['name'] == "plateau":
            scheduler.step(best_auc)


def train(train_loader, model, optimizer, scheduler, args):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        input = list(map(lambda t: t.to(args['model']['device']), process_batch(batch)))
        preds = model(input)
        targets = input[3]  # correct

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, scheduler, args)

        if step % args['trainer']['log_step'] == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f"TRAIN AUC : {auc} ACC : {acc}")
    return auc, acc, loss_avg


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(valid_loader):
        input = list(map(lambda t: t.to(args['model']['device']), process_batch(batch)))

        preds = model(input)
        targets = input[3]  # correct
        
        loss = compute_loss(preds, targets)

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f"VALID AUC : {auc} ACC : {acc}\n")
    loss_avg = sum(losses) / len(losses)
    return auc, acc, loss_avg


def validate_with_loss(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        input = list(map(lambda t: t.to(args['model']['device']), process_batch(batch)))

        preds = model(input)
        targets = input[3]  # correct

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f"VALID AUC : {auc} ACC : {acc}\n")

    return auc, acc


def inference(args, test_data, model):

    model.eval()
    _, test_loader = get_GES_loaders(args['data_loader']['args'], None, test_data)

    total_preds = []

    for step, batch in enumerate(test_loader):
        input = list(map(lambda t: t.to(args['model']['device']), process_batch(batch)))

        preds = model(input)

        # predictions
        preds = preds[:, -1]
        preds = torch.nn.Sigmoid()(preds)
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    time = datetime.now().strftime('%Y%m%d%H%M%S')
    model_name = args['model']['name']
    write_path = os.path.join(args['test']['submission_dir'], time + "_" + model_name + ".csv")
    if not os.path.exists(args['test']['submission_dir']):
        os.makedirs(args['test']['submission_dir'])
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))


# 배치 전처리
def process_batch(batch):

    test, question, tag, correct, mask, user_mean, user_acc, elap_time, recent3_elap_time, elo_prob, assess_ans_mean, prefix = batch

    # change to float
    mask = mask.float()
    correct = correct.float()

    # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

    #  test_id, question_id, tag
    test = ((test + 1) * mask).int()
    question = ((question + 1) * mask).int()
    tag = ((tag + 1) * mask).int()

    return (test, question, tag, correct, mask, interaction, user_mean, user_acc, elap_time, recent3_elap_time, elo_prob, assess_ans_mean, prefix)



# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, scheduler, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args['trainer']['clip_grad'])
    if args['scheduler']['name'] == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print("saving model ...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args, adj_matrix):

    model_path = args['test']['model_dir']
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(adj_matrix, **args['model'])

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
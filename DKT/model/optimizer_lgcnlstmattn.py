from torch.optim import Adam, AdamW


def get_optimizer(model, **args):
    if args['name'] == "adam":
        optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    if args['name'] == "adamW":
        optimizer = AdamW(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    # 모든 parameter들의 grad값을 0으로 초기화
    optimizer.zero_grad()

    return optimizer
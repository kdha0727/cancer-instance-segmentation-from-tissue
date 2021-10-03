import torch
from tqdm.notebook import tqdm


def train_one_epoch(model, criterion, optimizer, scheduler, train_loader, val_loader, device, epoch):

    model.train()

    if epoch == 0:  # warm_up_scheduler
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)

        def f(x):
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, f)

    loss_f = None
    inner_tq = tqdm(train_loader, total=len(train_loader), leave=False, desc=f'Iteration {epoch} train')

    for images, masks in inner_tq:

        images = images.to(device).float()
        masks = masks.clamp(0., 1.).to(device).float()  # IMPORTANT: clamp

        prediction = model(images)
        # prediction = prediction.softmax(dim=-3)
        loss = criterion(prediction, masks)
        loss.backward()

        if torch.cuda.device_count() > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        optimizer.step()
        optimizer.zero_grad()

        if epoch == 0:
            scheduler.step()

        if loss_f is not None:
            loss_f = 0.98 * loss_f + 0.02 * loss.item()
        else:
            loss_f = loss.item()
        inner_tq.set_postfix(loss=loss_f)

        print(f"\r{loss.item():.4f}   ", end='')

    if epoch != 0 and scheduler is not None:
        scheduler.step()

    print(f"\rIteration {epoch} train loss: {loss_f:.4f}")

    if val_loader is not None:
        model.eval()
        inner_tq = tqdm(val_loader, total=len(val_loader), leave=False, desc=f'Iteration {epoch} eval')
        with torch.no_grad():
            loss_f = count = 0
            for images, masks in inner_tq:
                images = images.to(device).float()
                masks = masks.clamp(0., 1.).to(device).float()  # IMPORTANT: clamp
                prediction = model(images)
                # prediction = prediction.argmax(dim=-3).long()
                # prediction = torch.zeros_like(prediction).scatter(
                #     dim=-3, index=prediction.unsqueeze(dim=-3).long(), src=torch.ones_like(prediction))
                # prediction = prediction.softmax(dim=-3)
                loss_f += criterion(prediction, masks).item()
                count += 1
            loss_f = loss_f / count if count else None
            inner_tq.set_postfix(loss=loss_f)
            print(f"Iteration {epoch} eval loss: {loss_f:.4f}")

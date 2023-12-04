import torch


def set_lr(optimizer, new_lr):
    """
    Sets the optimizer lr to the specified value.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        new_lr (float): the new learning rate to set.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def warm_up_lr(optimizer, cur_iter, warmup_iters, base_lr, warmup_factor):
    """
    Retrieve the learning rate of the current epoch with the option to perform
    warm up in the beginning of the training stage.
    Args:
        optimizer (optim): the optimizer using to optimize the current network.
        cur_iter (int): the number of iter of the current training stage.
        warmup_iters (int): the number of iter to warm up.
        base_lr (float): the base learning rate for network training
        warmup_factor (float): the starting lr factor
    """

    # Perform warm up.
    if cur_iter < warmup_iters:
        lr_start = warmup_factor * base_lr
        alpha = (base_lr - lr_start) / warmup_iters
        lr = cur_iter * alpha + lr_start

        set_lr(optimizer, lr)
    elif cur_iter == warmup_iters:
        set_lr(optimizer, base_lr)


if __name__ == "__main__":
    a = torch.rand(2, requires_grad=True)
    op = torch.optim.Adam([{'params': a}], lr=1.0)

    sc = torch.optim.lr_scheduler.ReduceLROnPlateau(
        op,
        mode="min",
        factor=0.5,
        patience=10,
        # cooldown=10,
        # min_lr=1e-6,
        verbose=True,)
    import matplotlib.pyplot as plt

    x = range(1, 101)
    warm_iter = 0
    lr_recording = []

    for i in x:
        warm_up_lr(op, i-1, warm_iter, 5.0, 0.1)

        # param.backward()
        assert len(op.param_groups) == 1
        lr_recording.append(op.param_groups[0]['lr'])

        print(i, op.param_groups[0]['lr'])
        if i -1>= warm_iter:
            print('begin')
            sc.step(1.0)
        input()
    print('Step:', max(x))
    print('Lrs: {}len'.format(len(lr_recording)), lr_recording)

    plt.plot(x, lr_recording)
    plt.savefig('lr_text.png')
    plt.close()



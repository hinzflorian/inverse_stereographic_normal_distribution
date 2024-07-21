"""Training function
"""

import time

import torch
import torch.optim.lr_scheduler as lr_scheduler


def train(
    model,
    learning_rate,
    train_loader,
    num_epochs,
    loss_function,
):
    """function to train the model

    Args:
        model : the model to train
        learning_rate : the learning rate in the optimization algorithm
        train_loader : train data loader
        num_epochs : number of epochs to train
        loss_function : the loss function to use
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=4
    )
    total_steps = len(train_loader)
    for epoch in range(num_epochs):
        losses = []
        start_time = time.time()
        for i, torsions in enumerate(train_loader):
            device = next(model.parameters()).device
            torsions = torsions.to(device=device, dtype=torch.float64)
            density_output = model(torsions)
            loss = loss_function(density_output)
            losses.append(loss)
            if torch.isnan(loss):
                print("bug")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(
                "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, i + 1, total_steps, loss.item()
                )
            )
        loss_epoch = torch.mean(torch.stack(losses))
        print("loss_epoch: ", loss_epoch)
        scheduler.step(loss_epoch)
        print("learning rate: ", optimizer.param_groups[0]["lr"])
        end_time = time.time()
        print("time for epoch: ", end_time - start_time)

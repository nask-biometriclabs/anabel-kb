import torch
import torch.nn as nn
from validation import evaluate


def train(model, device, train_dataloader, val_dataloader, n_epochs, optimizer, lr_scheduler, best_model_file,
          tb_writer=None, print_freq=10, save_model=True, max_iterations_per_epoch=None, warmup=3):
    criterion = nn.CrossEntropyLoss()
    best_ap = 0
    lr0 = optimizer.param_groups[0]['lr']

    for epoch in range(1, n_epochs + 1):
        if warmup > 0:
            if epoch < warmup:
                optimizer.param_groups[0]['lr'] = 0.01
            elif epoch == warmup:
                optimizer.param_groups[0]['lr'] = lr0

        ns = 0
        running_loss = 0
        running_corrects = 0
        model.train()
        for b, batch in enumerate(train_dataloader):

            if max_iterations_per_epoch is not None and b > max_iterations_per_epoch:
                break

            x = batch['data'].to(device)
            keys = x[:, :, :1]
            x = x[:, :, 2:]
            targets = batch['user_index'].to(device)

            orig_seq_len = batch['orig_seq_len'].to(device)
            outputs = model(keys, x, orig_seq_len=orig_seq_len)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, targets)
            corrects = torch.sum(preds == targets.data).item()
            running_corrects += corrects
            running_loss += loss.item() * x.size(0)
            ns += len(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (b % print_freq) == 0:
                print('\rEpoch: [{}][{}/{}]\t'
                      'Loss {:.3f} (avg={:.3f})'
                      '\tAccuracy {:.3f} (avg={:.3f})'.format(
                    epoch, b + 1, len(train_dataloader), loss.item(), running_loss / ns, float(corrects) / len(x),
                           float(running_corrects) / ns), end='')

        lr_scheduler.step()
        epoch_loss = running_loss / ns
        epoch_acc = float(running_corrects) / ns
        if tb_writer is not None:
            tb_writer.add_scalar('Training/Loss', epoch_loss, epoch)
            tb_writer.add_scalar('Training/Accuracy', epoch_acc, epoch)
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            tb_writer.add_scalar('Training/Lr', lr, epoch)

        val_results = evaluate(model, device, val_dataloader, epoch, tb_writer)

        ap = epoch_acc + val_results['ROC-AUC']
        if save_model and ap > best_ap:
            best_ap = ap
            print("Saving model")
            model.save(best_model_file, epoch)
    return model

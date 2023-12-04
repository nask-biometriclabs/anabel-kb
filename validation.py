import torch
from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.metrics import roc_auc_score


def get_distance_metrics(x, y):
    D = pairwise_distances(x, metric="cosine")
    L = np.equal(np.tile(y, (len(y), 1)), np.tile(np.array([y]).T, (1, len(y))))
    pos_samples = D[np.where((L == True) & (np.tril(np.ones_like(D), -1) > 0))]
    neg_samples = D[np.where((L == False) & (np.tril(np.ones_like(D), -1) > 0))]
    mean_pos = np.average(pos_samples)
    mean_neg = np.average(neg_samples)
    tf = np.concatenate((np.zeros_like(pos_samples), np.ones_like(neg_samples)), axis=0)
    scores = np.concatenate((pos_samples, neg_samples), axis=0)
    roc_auc = roc_auc_score(tf, scores)
    metrics = dict()
    metrics['mean_pos'] = mean_pos
    metrics['mean_neg'] = mean_neg
    metrics['ROC-AUC'] = roc_auc
    return metrics


def evaluate(model, device, val_dataloader, epoch, tb_writer=None):
    with torch.no_grad():
        model.eval()
        running_corrects = 0
        nb = 0
        ns = 0
        embeddings = []
        user_ids = []

        for b, batch in enumerate(val_dataloader):
            x = batch['data'].to(device)
            keys = x[:, :, :1]
            x = x[:, :, 2:]
            targets = batch['user_index'].to(device)
            orig_seq_len = batch['orig_seq_len'].to(device)
            outputs = model.enroll(keys, x, orig_seq_len=orig_seq_len)
            embeddings.extend(outputs.cpu().numpy().tolist())
            user_ids.extend([uid for uid in batch['user_index']])
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == targets)
            nb += 1
            ns += len(targets)
            print("\rBatch {}/{}: {} samples".format(b + 1, len(val_dataloader), len(batch['data'])), end='')
        epoch_acc = running_corrects.double() / ns
        result = get_distance_metrics(embeddings, user_ids)
        roc_auc = result['ROC-AUC']
        result['Accuracy'] = epoch_acc
        result['n'] = ns
        print('\rVALIDATION Epoch {} ROC-AUC={:.3f}'.format(epoch, roc_auc))

        if tb_writer is not None:
            tb_writer.add_scalar('Validation/ROC-AUC', roc_auc, epoch)
            tb_writer.add_scalar('Validation/Accuracy', epoch_acc, epoch)

    return result

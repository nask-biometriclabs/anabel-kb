import os
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import transforms
from model import create_model
from transforms import FeatureSet
import torch
from torch.utils.data import DataLoader
from dataset import KVCDataset
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anabel-KA trainer.')
    parser.add_argument('-d', "--data_dir", type=str, default="data",
                        help="The location of the directory where KVC data is stored.")
    parser.add_argument('-t', "--task", choices=["mobile", "desktop"], type=str, default="desktop",
                        help="Scenario type ('mobile' or 'desktop')")
    args = parser.parse_args()

    current_directory = os.getcwd()
    folder_path = os.path.join(current_directory, args.data_dir)
    data_folder_path = os.path.join(folder_path, args.task)
    data_path = os.path.join(data_folder_path, f"{args.task}_dev_set.npy")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified data file '{args.data_dir}' does not exist.")

    train_fold_file = os.path.join(folder_path, "folds", args.task, "train_fold{}.txt")
    val_fold_file = os.path.join(folder_path, "folds", args.task, "val_fold{}.txt")

    logdir = os.path.join(current_directory, "logs", args.task)

    # params
    feature_set = FeatureSet.KEYS_HOLD_AND_INTERKEY
    seq_len: int = 66
    n_kernels: int = 32
    representation_size: int = 256
    folds: int = 10  # 10-fold cross validation
    lr: float = 0.01
    batch_size: int = 128
    n_epochs: int = 2
    weight_decay: float = 0.0001  # parameter for Adam optimizer
    step_size: int = 10  # parameter for StepLR
    gamma: float = 0.1  # parameter for StepLR
    warmup: int = 3

    for fold in range(folds):
        train_fold = train_fold_file.format(fold)
        val_fold = val_fold_file.format(fold)

        train_transforms = [transforms.RandomSubstring(),
                            transforms.KVCTransform(seq_len=seq_len),
                            transforms.SelectFeatures(feature_set),
                            transforms.NumpyToTensor()]

        val_transforms = [transforms.KVCTransform(seq_len=seq_len),
                          transforms.SelectFeatures(feature_set),
                          transforms.NumpyToTensor()]

        train_dataset = KVCDataset(data_path, fold_file=train_fold, transform=train_transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        val_dataset = KVCDataset(data_path, fold_file=val_fold, transform=val_transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        n_classes = train_dataset.get_nclasses()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = create_model(n_classes, feature_set, seq_len, n_kernels, representation_size)
        model.to(device)

        subdir = datetime.today().strftime('AnabelKa_run_%Y-%m-%d-%H%M%S')
        tb_writer = SummaryWriter(os.path.join(logdir, subdir))
        best_model_file = os.path.join(logdir, subdir + "_model_best.pkl")

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        train(model, device, train_dataloader, val_dataloader, n_epochs, optimizer, lr_scheduler, best_model_file,
              tb_writer,
              save_model=True, max_iterations_per_epoch=None, warmup=warmup)

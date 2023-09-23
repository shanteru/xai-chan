
import os

import yaml

import torch

from torch import optim
from torch.utils.tensorboard import SummaryWriter

# internal imports
from utils import dataset, models, ssl_loss, trainer_MPCS, transform, augmentation_strategy

def load_configuration(config_path):
    """Load configuration information from a yaml file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_experiment_config(config, experiment_description):
    result_path = os.path.join(config["results"]["result_base_path"], experiment_description)
    os.makedirs(result_path, exist_ok=True)
    with open(f"{result_path}/experiment_config.yaml", 'w') as file:
        yaml.dump(config, file)


def pretrain_model(config):
    """Set up and pretrain model."""
    # Extract configuration values
    data_path = config["data_path"]
    data_portion = config["data_portion"]
    encoder_config = config["encoder"]
    method_config = config["method"]
    learning_rate_config = config["learning_rate"]
    weight_decay = config["weight_decay"]
    computational_infra = config["computational_infra"]
    batch_size_list = config["batch_size_list"]
    GPU = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize model and optimizer
    model = models.ResNet50_SSL(projector=encoder_config["projector"], supervised_pretrained=encoder_config["pretrained"] == "imagenet").to(GPU)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate_config["lr_only"], weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=learning_rate_config["patience"], min_lr=5e-4)
    criterion = ssl_loss.SimCLR_loss(gpu="cuda:0", temperature=method_config["temperature"])
    
    # Pretraining
    for fold in list(computational_infra["fold_to_gpu_mapping"].keys()):
        fold_root = os.path.join(data_path, fold, data_portion)
        experiment_description = f"_{fold}_{method_config['name']}_{method_config['variant']}_{encoder_config['name']}{encoder_config['version']}_{encoder_config['pretrained']}_"
        
        save_experiment_config(config, experiment_description)
        
        for batch_size in batch_size_list:
            train_loader = dataset.get_BreakHis_trainset_loader(
                train_path=fold_root,
                training_method=method_config["name"],
                transform=transform.resize_transform,
                augmentation_strategy=augmentation_strategy.pretrain_augmentation,
                image_pair=[40, 100, 200, 400],
                pair_sampling_method=method_config["variant"],
                batch_size=batch_size,
                num_workers=computational_infra["workers"]
            )
            trainer = trainer_MPCS.Trainer_MPCS(
                experiment_description=experiment_description,
                pair_sampling_method=method_config["variant"],
                dataloader=train_loader,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=config["epochs"],
                batch_size=batch_size,
                gpu=GPU,
                criterion=criterion,
                result_path=os.path.join(config["results"]["result_base_path"], experiment_description),
                writer=SummaryWriter(log_dir=os.path.join(config["logs"]["tensorboard_base_path"], experiment_description)),
                model_save_epochs_dir=config["pretraining_model_saving_scheme"]
            )
            trainer.train()

if __name__ == "__main__":
    config = load_configuration("pretrain_mpcs_op.yaml")
    pretrain_model(config)

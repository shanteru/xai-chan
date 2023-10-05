import  os, yaml
import torch.nn as nn
import torch

from torch.utils.tensorboard import SummaryWriter

# internal packages
from utils import dataset_test, transform, bc_config, models,train_util, augmentation_strategy as aug

# 1. Configuration Loading
def load_config(config_path):
    """Load configuration information from a yaml file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

    

def train_model(args_dict, fold, magnification):
    #1. Data settings
    data_path = args_dict["data_path"]
    train_data_portion = args_dict["train_data_portion"]
    test_data_portion =  args_dict["test_data_portion"]
    
    #2. Model settings
    encoder = args_dict["encoder"]["name"]
    version = args_dict["encoder"]["version"]
    dropout = args_dict["encoder"]["fc_dropout"]
    
    #3. Pretraining method settings
    pretraining_method_type = args_dict["pretrained_encoder"]["method_type"]
    # pretraining_pair_sampling_method = args_dict["pretrained_encoder"]["variant"]
    # pretraining_initial_weights = args_dict["pretrained_encoder"]["initial_weights"]
    # pretraining_batch_size_list = args_dict["pretrained_encoder"]["batch_size_list"]
    # pretraining_epochs_list = args_dict["pretrained_encoder"]["epochs_list"]
    # pretraining_checkpoint_base_path = args_dict["pretrained_encoder"]["checkpoint_base_path"]

    #4. Training (finetune) settings
    epochs = args_dict["epochs"]
    batch_size = args_dict["batch_size"]
    LR = args_dict["learning_rate"]["lr_only"]
    patience = args_dict["learning_rate"]["patience"]
    early_stopping_patience = args_dict["early_stopping_patience"]
    weight_decay = args_dict["weight_decay"]
    augmentation_level = args_dict["augmentation_level"]
    
    
    #5. Computational infra settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    workers = args_dict["computational_infra"]["workers"]
    
    
    #6. Logs and results settings
    tensorboard_base_path = args_dict["logs"]["tensorboard_base_path"]
    os.makedirs(tensorboard_base_path, exist_ok=True)
    result_base_path = args_dict["results"]["result_base_path"]
    os.makedirs(result_base_path, exist_ok=True)
    # result_stats_path = args_dict["results"]["result_stats_path"]
    # os.makedirs(result_stats_path, exist_ok=True)
    
    
    # -----  Loading the Data with Specified Augmentation Strategy
    
    augmentation_strategy = None
    if "low" == augmentation_level:
        augmentation_strategy = aug.augmentation_03
    elif "moderate" == augmentation_level:
        augmentation_strategy = aug.augmentation_05
    elif "high" == augmentation_level:
        augmentation_strategy = aug.augmentation_08
    else:
        raise ValueError ("wrong input for augmentation level parameter")
    
    
    train_loader = dataset_test.get_breakhis_data_loader(
        dataset_path=os.path.join(data_path, fold, train_data_portion),
        transform=transform.train_transform,
        augmentation_strategy=augmentation_strategy,
        pre_processing=[],
        image_type_list=[magnification],
        num_workers=workers
    )

    val_loader = dataset_test.get_breakhis_data_loader(
        dataset_path=os.path.join(data_path, fold, test_data_portion),
        transform=transform.resize_transform,
        pre_processing=[],
        image_type_list=[magnification],
        num_workers=workers,
        is_test=True
    )
    # -----  Data Loaders Created 
    
    # ----- Experiment Description Config
    if "train_" in train_data_portion or "val_" in train_data_portion:
        DP = int(''.join(filter(str.isdigit, train_data_portion)))
    
    experiment_description = f"_{fold}_{magnification}_BreakHis_FT_{DP}_{encoder}{version}_{pretraining_method_type}_"

    
    # ----- 
    
    
    # ----- Model Loading  
    downstream_task_model = None
    if "resnet" == encoder:
        print(f"Start - loading weights for {fold}_{pretraining_method_type}_{encoder}{version}")
        downstream_task_model = models.ResNet_Model(version=int(version), pretrained=True)
        num_ftrs=downstream_task_model.num_ftrs
        print(f"Stop - loading weights for {fold}_{pretraining_method_type}_{encoder}{version}")
        downstream_task_model.model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_ftrs, 1))
    
    downstream_task_model = downstream_task_model.to(device)
    
    # ----- Model Loaded 
    
    # ----- Training Model here 
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(downstream_task_model.parameters(), lr=LR, weight_decay= weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1 ,patience=patience, min_lr= 5e-3)
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_base_path, experiment_description))
    train_util = train_util.Train_Util(
        experiment_description = experiment_description, 
        epochs = epochs, 
        model=downstream_task_model, 
        device=device, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        batch_size=batch_size,
        scheduler=scheduler, 
        num_classes= len(bc_config.binary_label_list), 
        writer=writer, 
        early_stopping_patience = early_stopping_patience, 
        batch_balancing=False,
        result_folder=result_base_path
        )
    train_util.train_and_evaluate()
    
    # ----- Training Model ends 

if __name__ == "__main__":
    config = load_config("imagenet_run.yaml")
    
    for fold in list(config["computational_infra"]["fold_to_gpu_mapping"].keys()):
        train_model(config, fold,'400X')

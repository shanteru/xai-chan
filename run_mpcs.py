import os
import yaml
import torch.nn as nn
import torch
import csv

from torch.utils.tensorboard import SummaryWriter

# Internal packages
from utils import bc_config, dataset_test, train_utils, transform, models, augmentation_strategy as aug

def load_config(config_path):
    """Load configuration information from a yaml file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_mpcs_model(args_dict, fold, magnification):
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
    pretraining_pair_sampling_method = args_dict["pretrained_encoder"]["variant"]
    pretraining_initial_weights = args_dict["pretrained_encoder"]["initial_weights"]
    pretraining_batch_size_list = args_dict["pretrained_encoder"]["batch_size_list"]
    pretraining_epochs_list = args_dict["pretrained_encoder"]["epochs_list"]
    pretraining_checkpoint_base_path = args_dict["pretrained_encoder"]["checkpoint_base_path"]

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
    result_stats_path = args_dict["results"]["result_stats_path"]
    os.makedirs(result_stats_path, exist_ok=True)
    
    
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

    # ----- Experiment Description 
    if "train_" in train_data_portion or "val_" in train_data_portion:
        DP = int(''.join(filter(str.isdigit, train_data_portion)))
    
    threshold = 0.2
    
    # ----- Get the Correct Directory's Weight
    for pretrained_encoder_dir in os.listdir(pretraining_checkpoint_base_path):
            
        if (fold in pretrained_encoder_dir) and (pretraining_method_type in pretrained_encoder_dir) and (pretraining_pair_sampling_method in pretrained_encoder_dir) and (pretraining_initial_weights in pretrained_encoder_dir) and (encoder in pretrained_encoder_dir and str(version) in pretrained_encoder_dir):
            print(fold)
            for pretrained_batch_size in os.listdir(os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir)):
                if os.path.isdir(os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir,pretrained_batch_size)):# and pretrained_batch_size in list(pretraining_batch_size_list):
                    _pretrained_batch_size = int(pretrained_batch_size)
                    if _pretrained_batch_size in list(pretraining_batch_size_list):
                        for pretrained_epoch in list(os.listdir(os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir, str(pretrained_batch_size)))):
                            _pretrained_epoch = int(pretrained_epoch)
                            if _pretrained_epoch in list(pretraining_epochs_list):
                                #Experiment description
                                experiment_description = f"_{fold}_{magnification}_BreakHis_FT_{DP}_{encoder}{version}_{pretraining_method_type}_{pretraining_pair_sampling_method}_BS{pretrained_batch_size}_epoch{pretrained_epoch}_{pretraining_initial_weights}_"
                                print ('experiment_description ', experiment_description)
                                downstream_task_model = None
                                if "resnet" == encoder:
                                    print('inside resnet encoder block')
                                    downstream_task_model = models.ResNet_Model(version=int(version), pretrained=False)
                                    num_ftrs=downstream_task_model.num_ftrs
                                    print(f"Start - loading weights for {fold}_{pretraining_method_type}_{pretraining_pair_sampling_method}_{encoder}{version}_{pretraining_initial_weights}_{pretrained_batch_size}_{pretrained_epoch}")
                                    lst = os.listdir(os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir, pretrained_batch_size, pretrained_epoch))
                                    lst.sort()
                                    model_path = os.path.join(pretraining_checkpoint_base_path, pretrained_encoder_dir, pretrained_batch_size, pretrained_epoch, lst[0])
                                    print ("model path - ", model_path)
                                    print ("GPU to be used - ", device)
                                    downstream_task_model.model.load_state_dict(torch.load(model_path), strict = False)
                                    print(f"Stop - loading weights for {fold}_{pretraining_method_type}_{pretraining_pair_sampling_method}_{encoder}{version}_{pretraining_initial_weights}_{pretrained_batch_size}_{pretrained_epoch}")
                                    downstream_task_model.model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_ftrs, 1))

                                downstream_task_model = downstream_task_model.to(device)
                                criterion = nn.BCELoss()
                                optimizer = torch.optim.Adam(downstream_task_model.parameters(), lr=LR, weight_decay= weight_decay)
                                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1 ,patience=patience, min_lr= 5e-3)
                                writer = SummaryWriter(log_dir=os.path.join(tensorboard_base_path, experiment_description))
                                train_util = train_utils.Train_Util(
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
                                    # threshold=threshold, 
                                    result_folder=result_base_path
                                    )
                                best_acc, best_patient_level_acc, best_f1, best_classwise_precision, best_classwise_recall, best_classwise_f1 = train_util.train_and_evaluate()
                                filepath = os.path.join(result_stats_path, f"{magnification}_{encoder}{version}_FT_{DP}_{pretraining_method_type}_{pretraining_pair_sampling_method}_{pretrained_batch_size}_{pretrained_epoch}.csv")
                                file_exists = os.path.isfile(filepath)
                                with open(filepath, 'a') as f:
                                    writer = csv.writer(f)
                                    if not file_exists:
                                        writer.writerow(['test_name','patient_level_accuracy', 'image_level_accuracy', 'weighted_f1', 'classwise_precision_B', 'classwise_precision_M','classwise_recall_B','classwise_recall_M','classwise_f1_B','classwise_f1_M'])
                                    writer.writerow([f"{magnification}_{fold}_{encoder}{version}_FT_{DP}_{pretraining_method_type}_{pretraining_pair_sampling_method}_{pretrained_batch_size}_{pretrained_epoch}_{threshold}_aug_{augmentation_level}",best_patient_level_acc, best_acc, best_f1, best_classwise_precision[0], best_classwise_precision[1],best_classwise_recall[0], best_classwise_recall[1],best_classwise_f1[0],best_classwise_f1[1]])

        
    
if __name__ == "__main__":
    bc_config = load_config("run_mpcs.yaml")
    
    for fold in list(bc_config["computational_infra"]["fold_to_gpu_mapping"].keys()):
        print("current fold", fold)
        train_mpcs_model(bc_config, fold,'40X')

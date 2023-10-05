
import argparse
import  os, yaml, csv
import torch
import torch.nn as nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter
from datasets_test import get_breakhis_data_loader
from supervised.apply.transform import train_transform, resize_transform
from supervised.apply.augmentation_strategy import augmentation_03,augmentation_05, augmentation_08
from supervised.core.models import ResNet_Model

from supervised.core.train_util import Train_Util
import bc_config


# 1. Configuration Loading
def load_config(config_path):
    """Load configuration information from a yaml file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 2. Data Loading
def load_data(data_path, fold, portion, magnification, transform, augmentation=None, is_test=False, workers=1):
    return get_breakhis_data_loader(
        dataset_path=os.path.join(data_path, fold, portion),
        transform=transform,
        augmentation_strategy=augmentation,
        pre_processing=[],
        image_type_list=[magnification],
        num_workers=workers,
        is_test=is_test
    )


def train_model(args_dict, train_loader, val_loader, model, experiment_description, device):
    
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
    #print(pretraining_batch_size_list)
    pretraining_epochs_list = args_dict["pretrained_encoder"]["epochs_list"]
    pretraining_checkpoint_base_path = args_dict["pretrained_encoder"]["checkpoint_base_path"]

    #4. Training (finetune) settings
    epochs = args_dict["epochs"]
    batch_size = args_dict["batch_size"]
    # threshold = args_dict["threshold"]
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

    augmentation_strategy = None
    if "low" == augmentation_level:
        augmentation_strategy = augmentation_03
    elif "moderate" == augmentation_level:
        augmentation_strategy = augmentation_05
    elif "high" == augmentation_level:
        augmentation_strategy = augmentation_08
    else:
        raise ValueError ("wrong input for augmentation level parameter")



    #Experiment description
    DP = 0
    if "val_20" == train_data_portion:
        DP = 20
    elif "train_100" == train_data_portion:
        DP = 100
    elif "train_80" == train_data_portion:
        DP = 80
    elif "train_60" == train_data_portion:
        DP = 60
    elif "train_40" == train_data_portion:
        DP = 40
    elif "train_20" == train_data_portion:
        DP = 20
        
    def initialize_resnet_model(encoder, version, pretrained=False, model_path=None):
        if "resnet" != encoder:
            return None

        print(f"Start - loading weights for ResNet version {version} (pretrained: {pretrained})")
        model = ResNet_Model(version=int(version), pretrained=pretrained)
        
        if model_path:
            print(f"Loading from: {model_path}")
            model.model.load_state_dict(torch.load(model_path), strict=False)
        
        num_ftrs = model.num_ftrs
        model.model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(num_ftrs, 1))
        print(f"Stop - loading weights for ResNet version {version}")

        return model

    def get_training_util(description, model, device, train_loader, val_loader, batch_size):
        model = model.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1 ,patience=patience, min_lr= 5e-3)
        writer = SummaryWriter(log_dir=os.path.join(tensorboard_base_path, description))
        train_util = Train_Util(
            experiment_description = description, 
            epochs = epochs, 
            model=model, 
            device=device, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            optimizer=optimizer, 
            criterion=criterion, 
            batch_size=batch_size,
            scheduler=scheduler, 
            num_classes=len(bc_config.binary_label_list), 
            writer=writer, 
            early_stopping_patience=early_stopping_patience, 
            batch_balancing=False,
            result_folder=result_base_path
        )
        return train_util
    
    def get_pretrained_model_path(base_path, fold, method_type, sampling_method, initial_weights, encoder, version, batch_size_list, epochs_list):
        for pretrained_encoder_dir in os.listdir(base_path):
            if (fold in pretrained_encoder_dir) and (method_type in pretrained_encoder_dir) and (sampling_method in pretrained_encoder_dir) and (initial_weights in pretrained_encoder_dir) and (encoder in pretrained_encoder_dir and str(version) in pretrained_encoder_dir):
                for pretrained_batch_size in os.listdir(os.path.join(base_path, pretrained_encoder_dir)):
                    if os.path.isdir(os.path.join(base_path, pretrained_encoder_dir, pretrained_batch_size)):
                        _pretrained_batch_size = int(pretrained_batch_size)
                        if _pretrained_batch_size in batch_size_list:
                            for pretrained_epoch in os.listdir(os.path.join(base_path, pretrained_encoder_dir, str(_pretrained_batch_size))):
                                _pretrained_epoch = int(pretrained_epoch)
                                if _pretrained_epoch in epochs_list:
                                    lst = os.listdir(os.path.join(base_path, pretrained_encoder_dir, pretrained_batch_size, pretrained_epoch))
                                    lst.sort()
                                    return os.path.join(base_path, pretrained_encoder_dir, pretrained_batch_size, pretrained_epoch, lst[0])
        return None

    def train_and_log(train_util, result_stats_path, magnification, encoder, version, DP, method_type, sampling_method, pretrained_batch_size, pretrained_epoch, augmentation_level):
        best_acc, best_patient_level_acc, best_f1, best_classwise_precision, best_classwise_recall, best_classwise_f1 = train_util.train_and_evaluate()
        filepath = os.path.join(result_stats_path, f"{magnification}_{encoder}{version}_FT_{DP}_{method_type}_{sampling_method}_{pretrained_batch_size}_{pretrained_epoch}.csv")
        file_exists = os.path.isfile(filepath)
        with open(filepath, 'a') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['test_name','patient_level_accuracy', 'image_level_accuracy', 'weighted_f1', 'classwise_precision_B', 'classwise_precision_M','classwise_recall_B','classwise_recall_M','classwise_f1_B','classwise_f1_M'])
            writer.writerow([f"{magnification}_{fold}_{encoder}{version}_FT_{DP}_{method_type}_{sampling_method}_{pretrained_batch_size}_{pretrained_epoch}_aug_{augmentation_level}", best_patient_level_acc, best_acc, best_f1, best_classwise_precision[0], best_classwise_precision[1], best_classwise_recall[0], best_classwise_recall[1], best_classwise_f1[0], best_classwise_f1[1]])


    if "imagenet" == pretraining_method_type:
        experiment_description = f"_{fold}_{magnification}_BreakHis_FT_{DP}_{encoder}{version}_{pretraining_method_type}_"
        model = initialize_resnet_model(encoder, version, pretrained=True)
        train_util = get_training_util(experiment_description, model, device, train_loader, val_loader, batch_size)
        train_util.train_and_evaluate()

    
    else:
        model_path = get_pretrained_model_path(pretraining_checkpoint_base_path, fold, pretraining_method_type, pretraining_pair_sampling_method, pretraining_initial_weights, encoder, version, pretraining_batch_size_list, pretraining_epochs_list)
        if model_path:
            experiment_description = f"_{fold}_{magnification}_BreakHis_FT_{DP}_{encoder}{version}_{pretraining_method_type}_{pretraining_pair_sampling_method}_BS{os.path.basename(os.path.dirname(model_path))}_epoch{os.path.basename(os.path.dirname(os.path.dirname(model_path)))}_{pretraining_initial_weights}_"
            model = initialize_resnet_model(encoder, version, pretrained=False, model_path=model_path)
            train_util = get_training_util(experiment_description, model, device, train_loader, val_loader, batch_size)
            train_and_log(train_util, result_stats_path, magnification, encoder, version, DP, pretraining_method_type, pretraining_pair_sampling_method, os.path.basename(os.path.dirname(model_path)), os.path.basename(os.path.dirname(os.path.dirname(model_path))), threshold, augmentation_level)


                

if __name__ == '__main__':

        
    parser = argparse.ArgumentParser(description='Finetuning on BreakHis')
    parser.add_argument('--config', help='Config file for the experiment')
    args = parser.parse_args()

    args_dict = load_config(args.config)
    for fold in list(args_dict["computational_infra"]["fold_to_gpu_mapping"].keys()):
        train_loader = load_data(
            args_dict["data_path"], fold, args_dict["train_data_portion"], '400X', train_transform, augmentation_08, False, args_dict["computational_infra"]["workers"]
        )
        val_loader = load_data(
            args_dict["data_path"], fold, args_dict["test_data_portion"], '400X', resize_transform, None, True, args_dict["computational_infra"]["workers"]
        )
        train_model(args_dict, train_loader, val_loader, experiment_description, torch.device("cuda:0"))


import os
import random
import shutil
import csv
import copy
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

# Constants
RANDOM_STATE = 42
ROOT = 'dataset'
BENIGN_LIST = ['/benign/SOB/adenosis/','/benign/SOB/fibroadenoma/', '/benign/SOB/phyllodes_tumor/','/benign/SOB/tubular_adenoma/']
MALIGNANT_LIST = ['/malignant/SOB/lobular_carcinoma/', '/malignant/SOB/papillary_carcinoma/', '/malignant/SOB/ductal_carcinoma/', '/malignant/SOB/mucinous_carcinoma/']
K_FOLDS = 5

def get_patients_from_directory_list(directory_list):
    patients = []
    for directory in directory_list:
        p_dir_path = os.path.join(ROOT, directory.strip('/'))
        for p_id in os.listdir(p_dir_path):
            patients.append(os.path.join(p_dir_path, p_id))
    return patients

def initialize_stat_dict():
    return {'B': 0, 'M': 0, 'DC': 0, 'LC': 0, 'MC': 0, 'PC': 0, 'PT': 0, 'F': 0, 'TA': 0, 'A': 0}

def move_data_and_update_stats(patient_list, target_folder, stat_dict):
    for patient in patient_list:
        main_class = patient.split('/')[-1].split('_')[1]
        sub_class = patient.split('/')[-1].split('_')[2]
        stat_dict[main_class] += 1
        stat_dict[sub_class] += 1
        shutil.copytree(patient, os.path.join(target_folder, patient.split('/')[-1]))

def write_stats_to_csv(writer, data_name, stat_dict):
    writer.writerow([data_name] + [stat_dict[key] for key in sorted(stat_dict.keys())])

def main():
    benign_patients = get_patients_from_directory_list(BENIGN_LIST)
    malignant_patients = get_patients_from_directory_list(MALIGNANT_LIST)

    all_patients = benign_patients + malignant_patients
    random.Random(RANDOM_STATE).shuffle(all_patients)

    abstract_category_list = [p.split('/')[-1].split('_')[1] for p in all_patients]
    concrete_category_list = [p.split('/')[-1].split('_')[2] for p in all_patients]

    kfold = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(all_patients, concrete_category_list)):
        fold_path = os.path.join(ROOT, f'Fold_{fold}_{K_FOLDS}')
        Path(fold_path).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(ROOT, f'fold{fold}_stat.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['part', 'B', 'M', 'DC', 'LC', 'MC', 'PC', 'PT', 'F', 'TA', 'A'])

            stats = initialize_stat_dict()
            val_stats = initialize_stat_dict()
            test_stats = initialize_stat_dict()

            # Train/Val/Test splits & data movement
            for ratio in [(0.25, '60', '20'), (0.125, '70', '10')]:
                train_data, val_data = train_test_split(
                    [all_patients[index] for index in train_ids], 
                    stratify=[concrete_category_list[index] for index in train_ids], 
                    test_size=ratio[0], 
                    random_state=RANDOM_STATE
                )

                move_data_and_update_stats(train_data, os.path.join(fold_path, f'train_{ratio[1]}'), stats)
                move_data_and_update_stats(val_data, os.path.join(fold_path, f'val_{ratio[2]}'), val_stats)
                write_stats_to_csv(writer, f'train_{ratio[1]}', stats)
                write_stats_to_csv(writer, f'val_{ratio[2]}', val_stats)

                # Resetting stats for the next loop
                stats = initialize_stat_dict()
                val_stats = initialize_stat_dict()

            test_data = [all_patients[index] for index in test_ids]
            move_data_and_update_stats(test_data, os.path.join(fold_path, 'test_20'), test_stats)
            write_stats_to_csv(writer, 'test_20', test_stats)

if __name__ == "__main__":
    main()

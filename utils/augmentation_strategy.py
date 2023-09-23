from albumentations import A
from albumentations.pytorch import ToTensorV2

pretrain_augmentation = A.Compose([
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.3),
    A.HorizontalFlip(p=0.3),
    A.Rotate(limit=30, p=0.3),  #specify the 'limit' which defines the rotation range
    A.Affine(translate_percent=0.05, p=0.2),
    ToTensorV2()
])

pretrain_v2_augmentation = A.Compose([
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.3),
    A.HorizontalFlip(p=0.3),
    A.Rotate(limit=30, p=0.3),
    ToTensorV2()
])

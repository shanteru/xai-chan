import albumentations as A
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


augmentation_03 = A.Compose([
        A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.3),
        A.Flip(p=0.3),
        A.Rotate(p=0.3),
        A.Affine(translate_percent = 0.05, p=0.3),
        A.Resize(height=341, width=341, p=1),
        A.RandomCrop(height=252,width=252,p=1)
        ])

augmentation_05 = A.Compose([
        A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.6),
        A.Flip(p=0.5),
        A.Rotate(p=0.5),
        A.Affine(translate_percent = 0.1, p=0.5),
        A.Resize(height=341, width=341, p=1),
        A.RandomCrop(height=252,width=252,p=1)
        ])

augmentation_08 = A.Compose([
        A.ColorJitter (brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, always_apply=False, p=0.8),
        A.Flip(p=0.9),
        A.Rotate(p=0.9),
        A.Affine(translate_percent = 0.1, p=0.8)
        ])
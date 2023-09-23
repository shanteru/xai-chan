from torchvision import transforms as t

# Dataset input processing - trainset
resize_transform = t.Compose([
    t.ToPILImage(), 
    t.Resize((341, 341)),
    t.ToTensor()
])

# resize_transform_224 = t.Compose([
#     t.ToPILImage(), 
#     t.Resize((224, 224)),
#     t.ToTensor()
# ])

# crop_resize_transform = t.Compose([
#     t.ToPILImage(), 
#     t.Resize((230, 350)),
#     t.RandomCrop((230, 300)),
#     t.ToTensor()
# ])

# target_resize_transform = t.Compose([
#     t.ToPILImage(), 
#     t.Resize((230, 300)),
#     t.ToTensor()
# ])

from torchvision import transforms as t

# Dataset input processing - trainset
resize_transform = t.Compose([
    t.ToPILImage(), 
    t.Resize((341, 341)),
    t.ToTensor()
])
train_transform = t.Compose([
    t.ToPILImage(), 
    t.ToTensor()
])

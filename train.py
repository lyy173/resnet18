import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets import MyDataset
from resnet18 import resnet18
import torch.nn as nn

transforms = T.Compose([T.Resize(224),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

classdict = {"cat":0,"dog":1}

train_dataset = MyDataset("/root/resnet/datasets/data/train",classdict,transforms)
val_dataset = MyDataset("/root/resnet/datasets/data/val",classdict,transforms)

train_loader = DataLoader(train_dataset,batch_size=64,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=64,shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch = 500

model = resnet18().cuda()

eval_step = 1
plot_step = 100
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

for e in range(epoch):
    model.train()
    train_loss = 0
    for i,(img,label) in enumerate(train_loader):
        img,label = img.cuda(),label.cuda()
        output = model(img)
        loss = criterion(output,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % plot_step == 0:
            print(f"epoch:[{e}|{i}/{len(train_loader)}] loss:{loss}")

    if e % eval_step == 0:
        model.eval()
        eval_loss = 0
        eval_acc = 0
        best_acc = 0
        with torch.no_grad():
            for i,(img,label) in enumerate(val_loader):
                img,label = img.cuda(),label.cuda()
                output = model(img)
                loss = nn.CrossEntropyLoss()(output,label)
                eval_loss += loss.item()
                acc = torch.sum(torch.argmax(output,dim=1) == label) / len(label)
                eval_acc += acc.item()
            print("eval_loss:{} eval_acc:{}".format(eval_loss/len(val_loader),eval_acc/len(val_loader)))
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(model.state_dict(),"weights/best.pth")
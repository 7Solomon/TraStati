import torch
from neural_network_stuff.custome_DETR.detr import build
from data_folder.manage_datasets import load_datasets
import math, sys
import time

import torch
from torchvision import transforms
from torch.utils.data import DataLoader


from visualize.visualize_attention_map import attention_map
from neural_network_stuff.custome_DETR import misc_stuff



transform = transforms.Compose([
        transforms.ToTensor(),  # Konvertiert das Bild in einen Tensor
        transforms.Resize((840, 960)),  # Ändert die Größe des Bildes
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisiert die Pixelwerte
    ])


def test():
    model, criterion = build()
    train, val = load_datasets('test')

    batch_size = 1
    num_workers = 2   
    clip_max_norm = 0.1

    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)


    # parameter stuff, ka was genau das ist, aber wichtig. Glaube für bakcbane propagation
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]    
    optimizer = torch.optim.AdamW(param_dicts,lr=1e-4,
                                  weight_decay=1e-4)
    model.load_state_dict(torch.load('neural_network_stuff/models/test'))
    
    sampler_train = torch.utils.data.RandomSampler(train)
    sampler_val = torch.utils.data.SequentialSampler(val)

    # Lade den Dataloader
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)
    data_loader_train = DataLoader(train, batch_sampler=batch_sampler_train,
                                   collate_fn=misc_stuff.collate_fn, num_workers=num_workers)
    data_loader_val = DataLoader(val, batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=misc_stuff.collate_fn, num_workers=num_workers)

    model.train()
    criterion.train()
    for i, (samples, targets) in enumerate(data_loader_train):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)

        loss_dict = criterion(outputs, targets)   # Loss data
        weight_dict = criterion.weight_dict         # Von der Dokumentation
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        #print(losses)
        #with torch.no_grad():
        attention_map(outputs["attentions"])

        optimizer.zero_grad()
        losses.backward()
        #wiat_for_key = input('next? ')
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

    



if __name__ == '__main__':
    test()

import math, sys
import time

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

#import matplotlib.pyplot as plt
from src.neural_network_stuff.custome_dataset import CustomImageDataset
from src.visualize.visualize_loss import generate_loss_plot
from src.visualize.visualize_image import visualize_image

from src.neural_network_stuff.custome_DETR import misc_stuff

import src.configure as configure



def train_net(model, criterion, training_set, val_set, num_epochs: int=120, load_model: str = None, save_as: str='default'):
    batch_size = configure.batch_size
    num_workers = configure.num_workers  
    clip_max_norm = configure.clip_max_norm

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
    optimizer = torch.optim.AdamW(param_dicts,lr=1e-5,
                                  weight_decay=1e-4)


    # For plotting und so
    plot_train_loss = []

    # Load model
    if load_model != None and os.path.exists(load_model):
        model.load_state_dict(torch.load(load_model))
    else:
        print('Model wird nicht geladen, da entweder nicht angegeben oder nicht vorhanden')
    
    sampler_train = torch.utils.data.RandomSampler(training_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    # Lade den Dataloader
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)
    data_loader_train = DataLoader(training_set, batch_sampler=batch_sampler_train,
                                   collate_fn=misc_stuff.collate_fn, num_workers=num_workers)
    data_loader_val = DataLoader(val_set, batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=misc_stuff.collate_fn, num_workers=num_workers)
    
    # Training
    start_time = time.time()
    print('Start Training:')
    for epoch in range(num_epochs):
        # Train one Epoch function
        model.train()
        criterion.train()

        for i, (samples, targets) in enumerate(data_loader_train):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
        
            loss_dict = criterion(outputs, targets)   # Loss data
            weight_dict = criterion.weight_dict         # Von der Dokumentation
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)


            # reduce losses over all GPUs for logging purposes    Glaube brauch ich nicht aber ka 
            #loss_dict_reduced = misc_stuff.reduce_dict(loss_dict)
            #loss_dict_reduced_unscaled = {f'{k}_unscaled': v
            #                            for k, v in loss_dict_reduced.items()}
            #loss_dict_reduced_scaled = {k: v * weight_dict[k]
            #                            for k, v in loss_dict_reduced.items() if k in weight_dict}
            #losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            #loss_value = losses_reduced_scaled.item()
            
            print(f'Loss: {losses}, image[{i}]  in epoch {epoch}/{num_epochs}')


            #if not math.isfinite(loss_value):
            #    print("Loss is {}, stopping training".format(loss_value))
            #    print(loss_dict_reduced)
            #    sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            if clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()

            # Save the loss for plotting
            plot_train_loss.append(losses.item())


        
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {losses}")
        

        # Das Falls getoötet wir es gesaved ist
        torch.save(model.state_dict(), save_as)
    
    # Länge des Trainings
    end_time = time.time()
    print(f'Training finished after {end_time - start_time} seconds')

    # Plotting
    loss_plot_image = generate_loss_plot(plot_train_loss)

    # Save the loss plot if configured
    if configure.save_loss_plot:
        visualize_image(loss_plot_image, 'loss_plot', save_path='assets/loss_plot.jpg')
    else:
        visualize_image(loss_plot_image, 'loss_plot')
        
    return model.state_dict()


transform = transforms.Compose([
        transforms.ToTensor(),  # Konvertiert das Bild in einen Tensor
        transforms.Resize((840, 960)),  # Ändert die Größe des Bildes
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisiert die Pixelwerte
    ])





if __name__ == '__main__':
    pass
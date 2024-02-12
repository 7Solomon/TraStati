from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from neural_network_stuff.custome_DETR.detr import build
from neural_network_stuff.custome_DETR.misc_stuff import nested_tensor_from_tensor_list, collate_fn
from neural_network_stuff.custome_DETR import misc_stuff

from functions import load_datasets

transform = transforms.Compose([
        transforms.ToTensor(),  # Konvertiert das Bild in einen Tensor
        transforms.Resize((840, 960)),  # Ändert die Größe des Bildes
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisiert die Pixelwerte
    ])



def test():
    # Some Parameters
    batch_size = 2
    num_workers = 2   # Aus args
    n_epochs = 1

    clip_max_norm = 0.1


    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model, criterion = build()
    model.to(device)

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-5,
        },
    ]    
    optimizer = torch.optim.AdamW(param_dicts,lr=1e-4,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200)


    # Get datasets
    train, val = load_datasets('base_dataset')
    sampler_train = torch.utils.data.RandomSampler(train)
    sampler_val = torch.utils.data.SequentialSampler(val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, batch_size, drop_last=True)

    data_loader_train = DataLoader(train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=num_workers)
    data_loader_val = DataLoader(val, batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=num_workers)

    #start_time = time.time()     Eigentlich ganz nice

    for epoch in range(n_epochs):
        # Train one Epoch function
        model.train()
        criterion.train()

        for samples, targets in data_loader_train:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            print(outputs['outputs_class'].__sizeof__())
            print(outputs['output_center_degree_points'].__sizeof__())
            #print(targets)
            #loss_dict = criterion(outputs, targets)


        #lr_scheduler.step()




    IMG_URL = 'images/WhatsApp Image 2023-08-12 at 14.00.07(6).jpg'
    with Image.open(IMG_URL) as img:

        # get image
        img_tensor = transform(img)
        nested_img_tensor = nested_tensor_from_tensor_list([img_tensor])      

       

    



if __name__ == '__main__':
    test()

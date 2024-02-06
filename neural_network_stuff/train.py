import torch
from torchvision import transforms
#import matplotlib.pyplot as plt
from neural_network_stuff.custome_dataset import CustomImageDataset
from neural_network_stuff.custome_loss import CustomeComLoss
from neural_network_stuff.test_detr_model import testDetr

def train_net(model,training_set,val_set,num_epochs=120, load_model='neural_network_stuff/models/v_1.pth', save_as='default'):
    #training_set= CustomImageDataset('data_folder/test_dataloader/train/label.txt','data_folder/test_dataloader/train')
    #val_set = CustomImageDataset('data_folder/test_dataloader/val/label.txt','data_folder/test_dataloader/val')

    plot_train_los, plot_val_los = [], []

    if load_model != None:
        model.load_state_dict(torch.load(load_model))

    training_loader = torch.utils.data.DataLoader(training_set, batch_size=12, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=12, shuffle=True)

    criterion = CustomeComLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in training_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1) # Gegen explosion von weights  ka ab 0.1 richtig is 
            optimizer.step()


        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs)
                val_loss = criterion(outputs, targets)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")
        plot_train_los.append(loss.item())
        plot_val_los.append(val_loss.item())


    #plt.
    torch.save(model.state_dict(), save_as)
    return model.state_dict()


transform = transforms.Compose([
        transforms.ToTensor(),  # Konvertiert das Bild in einen Tensor
        transforms.Resize((840, 960)),  # Ändert die Größe des Bildes
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisiert die Pixelwerte
    ])


if __name__ == '__main__':
    model = testDetr()
    train_net(model)
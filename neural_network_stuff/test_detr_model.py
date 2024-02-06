from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torchvision import transforms
import torchvision.models as models


class testDetr(nn.Module):
    def __init__(self, hidden_dim=256, num_classes=5,max_output=20, image_size=None):
        super().__init__()
        self.backbone = models.resnet50()
        #self.positionalEncoder = PositionEmbeddingSine()

        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transfromer = nn.Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_data = nn.Linear(hidden_dim, 3)

        self.query_pos = nn.Parameter(torch.rand(max_output, hidden_dim))
        self.image_size = image_size

        #positional encoding with rand
        self.row_embed = nn.Parameter(torch.rand(256, hidden_dim //2))
        self.col_embed = nn.Parameter(torch.rand(256, hidden_dim //2))

    def forward(self,inputs):
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        h = self.conv(x)
        H,W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H,1,1),
            self.row_embed[:H].unsqueeze(1).repeat(1,W,1),
            ], dim=-1).flatten(0,1).unsqueeze(1)
        
        # For Batch Size pos repeating
        current_batch_size = h.size(0)
        pos = pos.repeat(1, current_batch_size, 1)

        h = self.transfromer(pos + 0.1 * h.flatten(2).permute(2,0,1), self.query_pos.unsqueeze(1).repeat(1, current_batch_size, 1)).transpose(0,1)        
        
        class_data = self.linear_class(h)
        data_data = self.linear_data(h).sigmoid()

        assert self.image_size != None , 'keine Width and height fürs Image defineds'
        scaled_data_data = data_data.clone()
        scaled_data_data[:, 0] = data_data[:, 0] * self.image_size[0]  
        scaled_data_data[:, 1] = data_data[:, 1] * self.image_size[1]  
        scaled_data_data[:, 3] = data_data[:, 3] * 360 

        scaled_data_data = scaled_data_data.to(dtype=torch.int32)
        return {'classes': class_data, 'data': scaled_data_data}


    def save_image_tensor_as_png(self,image):
        to_pil = transforms.ToPILImage()
        img_pil = to_pil(image.squeeze(0).cpu())
        img_pil.save('test_output.jpg')

    def split_image_into_grid(self,image,num_grid_h=4,num_grid_w=4):
        grid_height = image.size(2) // num_grid_h
        grid_width = image.size(3) // num_grid_w

        image_tensor_reshaped = image.view(3, num_grid_h, grid_height, num_grid_w, grid_width)

        image_tensor_reshaped = image_tensor_reshaped.permute(1, 3, 0, 2, 4).contiguous()

        image_tensor_reshaped = image_tensor_reshaped.view(num_grid_h * num_grid_w, 3, grid_height, grid_width)
        return image_tensor_reshaped

    def prepare_x_for_positional_encoding(self,x):
        batch_size, num_channels, height, width = x.size()
        sequence_length = height * width
        input_features = num_channels
        return x.view(batch_size, sequence_length, input_features)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#transform = transforms.Compose([      # Old transform
#    transforms.Resize((224, 224)),  # Adjust the size as needed
#    transforms.ToTensor(),
#])

transform = transforms.Compose([
        transforms.ToTensor(),  # Konvertiert das Bild in einen Tensor
        transforms.Resize((840, 960)),  # Ändert die Größe des Bildes
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalisiert die Pixelwerte
    ])

if __name__ == '__main__':

   print('Just a class')



# Display Mode
display_mode = "cv2"


# Save Options
save_heatmap = True
save_loss_plot = True




### Learning STUFF

# Backbone
lr_backbone = 1e-7
use_frozen_bn = True

# DETR
cD_loss_coef = 5
giou_loss_coef = 2
eos_coef = 0.1
num_classes = 5

# Matcher
cost_class=1
cost_cD=5


# Transformer
d_model=256
dropout=0.01
nhead=2
dim_feedforward=512
num_encoder_layers=2
num_decoder_layers=2
normalize_before=True
return_intermediate_dec=True
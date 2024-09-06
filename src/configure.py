# "plt"
# "cv2"
# "pil"
# "ipy"

# Display Mode
display_mode = "pil"

# Image generation
randomize_images = False
latex_abstand = 20
generated_system_colums = 8
generated_system_rows = 20



# Save Options
save_heatmap = True
save_loss_plot = True


# Data train Options
batch_size = 6
num_workers = 2   
clip_max_norm = 0.1

# Backbone
train_backbone = 1e-7
return_interm_layers = True

# Detr
num_classes = 5 ##
cD_loss_coef = 5
ce_loss_coef = 1
giou_loss_coef = 2
eos_coef = 0.1

# Matcher
cost_class=1
cost_cD=5

# posEmbedding
N_steps = 256 // 2

# Trasnformer
d_model=25
dropout=0.01
nhead=2
dim_feedforward=512
num_encoder_layers=2
num_decoder_layers=2
normalize_before=True
return_intermediate_dec=True

### Erstellung system

# Conenction Map Parameter
img_cut_out = 100
black_pixel_margin = 100
line_margin = 10

# Sandart length
standart_length_margin = 3
standart_direction_margin = 0.005

### Randomize Image
# Cut
cut_image_margin = 200

#Randomize
possible_blurrs = [(1,1),(3,3,),(5,5)]
trapez_kor_grenze_start = -20
trapez_kor_grenze_end = 0

degree_lines_radius = 100



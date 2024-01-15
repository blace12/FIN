epochs = 200
clamp = 2.0

# optimizer
lr = 1e-4
betas = (0.5, 0.999)
gamma = 0.5
weight_decay = 1e-5

noise_flag = True

# input settings
message_weight = 1500
stego_weight = 1
message_length = 64

# Train:
batch_size = 4
cropsize = 128

# Val:
batchsize_val = 4
cropsize_val = 128

# Data Path
TRAIN_PATH = 'data/DIV2K_train_HR'
VAL_PATH = 'data/DIV2K_valid_HR'

format_train = 'png'
format_val = 'png'

# Saving checkpoints:
MODEL_PATH = 'experiments/JPEG'
SAVE_freq = 5

suffix = ''
train_continue = False
diff = False

noises = [

    ['Cropout', 0.1],
    ['Cropout', 0.2],
    ['Cropout', 0.3],
    ['Cropout', 0.4],
    ['Cropout', 0.5],

    ['Dropout', 0.6],
    ['Dropout', 0.5],
    ['Dropout', 0.4],
    ['Dropout', 0.3],
    ['Dropout', 0.2],

]

optional_noises = [

    ['Cropout', 0.1],

    ['Dropout', 0.6],

    ['SPNoist', 0.01],

    ['JpegComp', 50],

    ['GaussNoise', 0.01],

    ['GaussBlur', 0.5],

    ['MedianBlur', 3],
]
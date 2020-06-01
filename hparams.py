from text import symbols

# Text
text_cleaners = ['english_cleaners']

# Mel
n_mel_channels = 80
num_mels = 80

# FastSpeech
vocab_size = 1024
N = 6
Head = 2
d_model = 384
duration_predictor_filter_size = 256
duration_predictor_kernel_size = 3
dropout = 0.1

#### brain
num_channel = 25 ## eeg 25, emg 3, v 1


brain_spectrogram_dim = 64
hz_start = 0
brain_hz_range = 50
brain_dim = (brain_hz_range-hz_start)*num_channel # 64-->1600,  50 --> 1250



word_vec_dim = 384
encoder_n_layer = 6
encoder_head = 2

encoder_conv1d_filter_size = 384
# encoder_conv1d_filter_size = 1536

transcript_encoder_conv1d_filter_size = 1536

max_sep_len = 2048
encoder_output_size = 384
decoder_n_layer = 6
decoder_head = 2
# decoder_conv1d_filter_size = 1536
decoder_conv1d_filter_size = 384

decoder_output_size = 384
fft_conv1d_kernel = 3
fft_conv1d_padding = 1
### Phoneme predictor
phoneme_predictor_filter_size = 256  ### 다른애들은 256
phoneme_predictor_kernel_size = 3
phoneme_predictor_dropout = 0.1
n_phonemes = 81 # korean 80 + silence for brain(@)
# Traind

model_name = 'final_condition5_test'
checkpoint_path = "/hd0/sh_save/BTS_5ch_filtered_model_0526/"+model_name+"/"#"/hd0/fastspeech/model_new"
# checkpoint_path = "/sd0/BTS_FFT/model_save/"+model_name+"/"#"/hd0/fastspeech/model_new"
logger_path = "./logger_"+model_name+"/"
mel_ground_truth = "./mels_"+model_name+"/"

npz_path = "/sd0/BTS_FFT/BTS_train_5ch_filtered_duration/"
val_npz_path = "/sd0/BTS_FFT/BTS_test1_5ch_filtered_duration/"

batch_size = 8
epochs = 1000
n_warm_up_step = 4000


grad_clip_thresh = 1.0


save_step = 5000
log_step = 5
clear_Time = 20

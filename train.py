# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from model import GradTTS
from data import TextMelDataset, TextMelBatchCollate
from utils import plot_tensor, save_plot
from text.symbols import symbols

import sys
sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_filelist_path = params.train_filelist_path
valid_filelist_path = params.valid_filelist_path
cmudict_path = params.cmudict_path
add_blank = params.add_blank

log_dir = params.log_dir
n_epochs = params.n_epochs
batch_size = params.batch_size
out_size = params.out_size
learning_rate = params.learning_rate
random_seed = params.seed

nsymbols = len(symbols) + 1 if add_blank else len(symbols)
n_enc_channels = params.n_enc_channels
filter_channels = params.filter_channels
filter_channels_dp = params.filter_channels_dp
n_enc_layers = params.n_enc_layers
enc_kernel = params.enc_kernel
enc_dropout = params.enc_dropout
n_heads = params.n_heads
window_size = params.window_size

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

dec_dim = params.dec_dim
beta_min = params.beta_min
beta_max = params.beta_max
pe_scale = params.pe_scale
clip_range = params.clip_range

HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'
N_TIMESTEPS = 50
ANGRY_IDX = 2

with open(HIFIGAN_CONFIG) as f:
    h = AttrDict(json.load(f))
vocoder = HiFiGAN(h)
vocoder.to(device)
vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
_ = vocoder.eval()
vocoder.remove_weight_norm()

# Keep track of global reward queries
reward_queries = 0

from emotion_predictor import SuperbPredictor
from scipy.io.wavfile import write
emotion_predictor = SuperbPredictor()

def get_rewards(model, x, x_lengths):
    global reward_queries
    y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=N_TIMESTEPS)
    audio = vocoder.forward(y_dec).squeeze().clamp(-1, 1)
    rewards = emotion_predictor.predict_emotion_batch(audio)
    rewards = rewards[:, ANGRY_IDX].view(-1, 1)
    # Update the global number of reward queries

    reward_queries += rewards.shape[0]
    return rewards.to(device)

if __name__ == "__main__":
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=log_dir)

    print('Initializing data loaders...')
    train_dataset = TextMelDataset(train_filelist_path, cmudict_path, add_blank,
                                   n_fft, n_feats, sample_rate, hop_length,
                                   win_length, f_min, f_max)
    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)
    test_dataset = TextMelDataset(valid_filelist_path, cmudict_path, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max)

    print('Initializing model...')
    model = GradTTS(nsymbols, 1, None, n_enc_channels, filter_channels, filter_channels_dp, 
                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                    n_feats, dec_dim, beta_min, beta_max, pe_scale, learning_rate).cuda()
    checkpoint_path = './checkpts/grad-tts.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=lambda loc, storage: loc))
        
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    #print('Initializing optimizer...')
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    for i, item in enumerate(test_batch):
        mel = item['y']
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=reward_queries, dataformats='HWC')
        save_plot(mel.squeeze(), f'{log_dir}/original_{i}.png')

    print('Preprocessing')
    iteration = 0
    base_model_all_log_probs = []
    base_model_all_rewards = []
    with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
        for batch_idx, batch in enumerate(progress_bar):
            x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
            y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
            _, _, _, base_model_log_probs = model(x, x_lengths, n_timesteps=N_TIMESTEPS, stoc=True, all_log_probs=True)
            base_model_all_log_probs.append(base_model_log_probs)
            model.eval()
            base_model_all_rewards.append(get_rewards(model, x, x_lengths))
            model.train()

    all_rewards = torch.stack(base_model_all_rewards)
    # Save the preprocessing
    torch.save(all_rewards, "all_rewards.pt")
    rewards_mean = all_rewards.mean()
    rewards_std = all_rewards.std()

    all_advantages = (all_rewards - rewards_mean) / (rewards_std + 1e-8)


    # all_rewards should be batch_indicies, size of batch, 1
    
    print('Start training...')
    for epoch in range(1, n_epochs + 1):
        model.train()
        # dur_losses = []
        # prior_losses = []
        # diff_losses = []
        with tqdm(loader, total=len(train_dataset)//batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()
                
                base_log_probs =  base_model_all_log_probs[batch_idx]
                advantages = all_advantages[batch_idx]

                

                info = model.train_ppo(x, x_lengths, N_TIMESTEPS, advantages, clip_range, base_log_probs, stoc=True)
                mean_loss = torch.mean(info['losses'])
                mean_approx_kl = torch.mean(info['approx_kl'])
                mean_clipfrac = torch.mean(info['clipfrac'])

                logger.add_scalar('mean_loss', mean_loss, global_step=reward_queries)
                logger.add_scalar('mean_approx_kl', mean_approx_kl, global_step=reward_queries)
                logger.add_scalar('mean_clipfrac', mean_clipfrac, global_step=reward_queries)

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)

                logger.add_scalar('training/encoder_grad_norm', enc_grad_norm,
                                  global_step=reward_queries)
                logger.add_scalar('training/decoder_grad_norm', dec_grad_norm,
                                  global_step=reward_queries)

                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, Reward queries: {reward_queries}'
                    progress_bar.set_description(msg)
                
                iteration += 1
        # We are not using these any more?
        # log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        # log_msg += '| prior loss = %.3f ' % np.mean(prior_losses)
        # log_msg += '| diffusion loss = %.3f\n' % np.mean(diff_losses)
        # with open(f'{log_dir}/train.log', 'a') as f:
        #     f.write(log_msg)

        if epoch % params.save_every > 0:
            continue

        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            rewards = []
            # How is this test_batch being set, is it the same as the previous grad-tts?
            for i, item in enumerate(test_batch):
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=50)


                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=reward_queries, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=reward_queries, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=reward_queries, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(), 
                          f'{log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(), 
                          f'{log_dir}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(), 
                          f'{log_dir}/alignment_{i}.png')
                audio = (vocoder.forward(y_dec).squeeze().clamp(-1, 1))
                rewards.append(emotion_predictor.predict_emotion(audio))
        
        ## Should we keep track of the loss as well?
        rewards = torch.cat(rewards)
        print("Logged rewards shape: ", rewards.shape)
        mean_reward = torch.mean(rewards, dim=0) 
        logger.add_scalar("neutral_reward", mean_reward[0], global_step=reward_queries)
        logger.add_scalar("happy_reward", mean_reward[1], global_step=reward_queries)
        logger.add_scalar("angry_reward", mean_reward[ANGRY_IDX], global_step=reward_queries)
        logger.add_scalar("sad_reward", mean_reward[3], global_step=reward_queries)
        print("angry reward: ", mean_reward[ANGRY_IDX])
        ckpt = model.state_dict()
        torch.save(ckpt, f=f"{log_dir}/grad_{epoch}.pt")
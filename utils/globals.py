'''
Author: Wang Hui (wh203@zju.edu.cn)
Date: 2024-10-10 15:01:54
LastEditors: silent-lindsay wh203@zju.edu.cn
LastEditTime: 2024-11-21 11:01:14
FilePath: /aiTvae/utils/globals.py
Description: 如果不难，要你干嘛。
'''

DATAPATH_DICT = { 
    'traffic': "dataset/traffic/traffic.csv",
    'electricity': "dataset/electricity/electricity.csv",
    'weather': "dataset/weather/weather.csv",
    'ETTh1':"dataset/ETT_small/ETTh1.csv",
    'ETTh2':"dataset/ETT_small/ETTh2.csv",
    'solar':"dataset/Solar/solar_AL.txt",
    'PEMS':"dataset/PEMS/PEMS03.npz",
    'ETTm1':"dataset/ETT_small/ETTm1.csv",
    'ETTm2':"dataset/ETT_small/ETTm2.csv"

}

CHANNEL_DICT = {
    'traffic': 862,
    'electricity': 321,
    'weather': 21,
    'ETTh1':7,
    'solar':137,
    'PEMS':358,
    'ETTh2':7,
    'ETTm1':7,
    'ETTm2':7,
}

DATAPATH = lambda x: DATAPATH_DICT[x]
SEQ_LEN = 96
PRED_LEN = 96
CHANNEL = lambda x: CHANNEL_DICT[x]
LATENT_DIM = 96
HIDDEN = [96, 64, 32, 16]

RECONSTRUCTION = 'reconstruction'
FORECAST = 'forecast'
MASK4RECONSTRUCTION = 0.2

BETA = 0.001

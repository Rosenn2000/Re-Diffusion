from argparse import ArgumentParser
import time
import numpy as np
import torch
from tqdm import tqdm
from data_provider.data_factory import data_provider
from utils.globals import DATAPATH, FORECAST, MASK4RECONSTRUCTION, RECONSTRUCTION, SEQ_LEN,CHANNEL
from utils.tools import sample, KL_divergence
from iTransformer import ITformer
from utils.metrics import metric
import re
import os
from models import Autoformer, Transformer, TimesNet, DLinear, FEDformer, \
    Informer,Reformer, ETSformer, Pyraformer, PatchTST,Crossformer,iTransformer, \
         TimeMixer, TSMixer,TimeXer
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from testquick import Exp_Long_Term_Forecast
from torch import optim, nn
from torch.nn import functional as F
from utils.tools import EarlyStopping_DF, adjust_learning_rate, visual
import seaborn as sns
import matplotlib.pyplot as plt
# def cal(z):
#     z_squared = z ** 2  # 逐元素平方
#     z_for=z**4
#     var = z_squared.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True).mean(dim=2,keepdim=True)
#     four=z_for.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True).mean(dim=2,keepdim=True)
#     # x_hat = (x - mean) / torch.sqrt(var + eps)
#     out=0.5 *var + 0.1*four

    
#     return out
class VaeIT(nn.Module):

    def __init__(self, configs):
        super(VaeIT, self).__init__()
        self.args=configs
        self.seq_len = configs.seq_len 
        self.pred_len = configs.pred_len
        self.device = self._acquire_device()
        self.enc_embedding = DataEmbedding_inverted(configs.pred_len, configs.vaed_model, configs.vae_embed, configs.freq,
                                                    configs.dropout)
        self.model_dict = {
            'TimesNet': TimesNet,
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'DLinear': DLinear,
            'FEDformer': FEDformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'ETSformer': ETSformer,
            'PatchTST': PatchTST,
            'Pyraformer': Pyraformer,
            'Crossformer': Crossformer,
            'iTransformer': iTransformer,
            'TimeMixer': TimeMixer,
            'TSMixer': TSMixer,
            'TimeXer': TimeXer
        }
        self.model = self._build_model().to(self.device)

        self.encoder =Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.pred_len, configs.n_heads),
                    configs.pred_len,
                    configs.vae_embed,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.pred_len)
        )
        
        # Decoder
        self.decoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.vaed_model, configs.n_heads),
                    configs.vaed_model,
                    configs.vae_embed,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.vaed_model)
        )
        
        self.projector = nn.Linear(configs.vaed_model, configs.pred_len, bias=True)
        self.linear=nn.Linear(configs.pred_len,configs.vaed_model)
        self.to(self.device)
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _acquire_device(self):
        device = torch.device('cuda:{}'.format(self.args.gpu_device))
        print('Use GPU: cuda:{}'.format(self.args.gpu_device))
        return device
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        setting='{}_{}'.format(
                    self.args.model_id,
                    self.args.model)
        path=os.path.join(self.args.checkpoints, setting)
        checkpoint_path=path+'/'+'checkpoint.pth'
        model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

        return model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.parameters(), lr=self.args.vae_learning_rate)
        return model_optim

    def cal(self,z):
        z_squared = z ** 2  # 逐元素平方
        z_for=z**4
        var = z_squared.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True).mean(dim=2,keepdim=True)
        four=z_for.mean(dim=0, keepdim=True).mean(dim=1, keepdim=True).mean(dim=2,keepdim=True)
        # x_hat = (x - mean) / torch.sqrt(var + eps)
        out=0.5 *var + 0.1*four
    
        return out

    def forward(self, x_enc,x_mark_enc,y_enc,x_mark_dec,batch_y):
    
        with torch.no_grad():
            y_pro=self.model(x_enc,x_mark_enc,y_enc,x_mark_dec)
        
        #B L N
        y_re=batch_y-y_pro
        ##B N L
        y_re=y_re.permute(0,2,1)

        ## B N E
        # y_enc=sigmoid_layer(y_enc)
        z, attns = self.encoder(y_re, attn_mask=None)
        z_before=self.linear(z)
         
       
        z, _ = self.decoder(z_before, attn_mask=None)
        
        
        # B N E -> B N S -> B S N 
        dec_out = self.projector(z).permute(0,2,1)[:, :self.pred_len, :] # filter the covariates
        
        dec_out=dec_out+y_pro
        dec_out=dec_out[:, -self.pred_len:, :]
        return dec_out, z_before
    
    def fit(self):
        train_data, train_loader = self._get_data(flag='train')
        # train_steps = len(train_loader)
        opt = self._select_optimizer()
        train_losses,kl_losses=[],[]

        for epoch in range(self.args.vae_epochs):
            total_loss, kl_loss = 0, 0
            self.train()
            for batch_x, batch_y, batch_x_mark, batch_y_mark,in tqdm(train_loader, desc='Training'): # Dataloader of thuml provides mark_x and mark_y, additionally
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

                outputs,z = self.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)
                

                opt.zero_grad()
                # y_hat, z= self.forward(x.float().to(device),y_batch.float().to(device),is_training=True)
                rec, kl = nn.MSELoss()(outputs, batch_y),self.cal(z)
                
                if epoch%2!=0:
                    while kl>1:
                        kl=kl*0.1
                    loss=rec+kl
                else:
                    loss=rec
                # loss = rec #*self.trainable_beta
                total_loss += loss.item()  # 累积每个batch的损失
                kl_loss+=kl.item()
                loss.backward()
                opt.step()
            self.eval()
                
                
            epoch_loss = total_loss / len(train_loader)
            epoch_kl_loss =kl_loss/ len(train_loader)
            train_losses.append(epoch_loss)
            kl_losses.append(epoch_kl_loss)
            print(f"Epoch {epoch+1}, Training Loss: {epoch_loss}, Valid Loss: {epoch_kl_loss}.")
            

           
        return 

    def forecast(self, x_enc,x_mark_enc,y_enc,x_mark_dec,batch_y):
        dec_out, _,= self.forward(x_enc,x_mark_enc,y_enc,x_mark_dec,batch_y)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
    def embedding(self,x):
        z, attns = self.encoder(x, attn_mask=None)
        enc_out=self.linear(z)
        
        return enc_out
    def encode_latent(self,x_bef,x_mark_bef,x_enc,x_mark_enc,y_enc,x_mark_dec,batch_y):
        enc_inp = torch.zeros_like(x_enc[:, -self.args.pred_len:, :]).float()
        enc_inp = torch.cat([x_enc[:, :self.args.label_len, :], enc_inp], dim=1).float().to(self.device)

        with torch.no_grad():
            y_pro=self.model(x_enc,x_mark_enc,y_enc,x_mark_dec)
            x_pro=self.model(x_bef,x_mark_bef,enc_inp,x_mark_enc)
        y_con=y_pro[:,:self.args.seq_len,:]
        x_con=x_pro[:,:self.args.seq_len,:]
        y_re=(batch_y-y_pro).permute(0,2,1)
        x_1=(x_enc-x_con).permute(0,2,1)
        x_2=(x_enc-y_con).permute(0,2,1)
        # x_3=(x_before-x_pro).permute(0,2,1)
        x_4=(x_enc-x_bef).permute(0,2,1)
        # print(x_1.shape)
        # print(x_2.shape)
        # print(x_4.shape)
        
        
        with torch.no_grad():
            z_before=self.embedding(y_re)
        
        
        

        
        ##x-y偏移残差，[B,N,L]
        # x_trans=x_enc-y_pro
        # with torch.no_grad():
        #     z_trans, attns = self.encoder(y_enc, attn_mask=None)
        #     trans=self.linear(z_trans)
        #     ##转移到[B*N,E]
        #     trans=trans.view(-1,E)

        # similarity_1 = F.cosine_similarity(z_enc, z_1, dim=1)
        # mean_1=similarity_1.mean(dim=0)
        # stdev_1 = torch.var(similarity_1, dim=0)
        # print("1","mean",mean_1,"stdev",stdev_1)
        # similarity_2 = F.cosine_similarity(z_enc, z_2, dim=1)
        # mean_2=similarity_2.mean(dim=0)
        # stdev_2 = torch.var(similarity_2, dim=0)
        # print("2","mean",mean_2,"stdev",stdev_2)
        # similarity_1 = F.cosine_similarity(z_enc, z_3, dim=1)
        # mean_1=similarity_1.mean(dim=0)
        # stdev_1 = torch.var(similarity_1, dim=0)
        # print("3","mean",mean_1,"stdev",stdev_1)
        
            

        # z_enc = z_before.detach()
        # means = z_enc.mean(1, keepdim=True).mean(0,keepdim=True).mean(2,keepdim=True).detach()
        # x_enc = z_enc - means
        
        # stdev = torch.sqrt(torch.var(z_enc, dim=(0,1,2), keepdim=True, unbiased=False))
        
        # x_pre=z_enc/stdev
        # x_ab=torch.mean(x_pre**3)
        # y_ab=torch.mean(x_pre**4)
        # print("x_ab",x_ab)
        # print("y_ab",y_ab)
        # 在 K 方向计算均值和标准差
        
        return z_before,y_pro,x_1,x_4,x_2
    def test_mean(self,z_before):
        z_enc = z_before.detach()
        means = z_enc.mean(1, keepdim=True).mean(0,keepdim=True).mean(2,keepdim=True).detach()
        x_enc = z_enc - means
        
        stdev = torch.sqrt(torch.var(z_enc, dim=(0,1,2), keepdim=True, unbiased=False))
        
        x_pre=z_enc/stdev
        x_ab=torch.mean(x_pre**3)
        y_ab=torch.mean(x_pre**4)
        print("x_ab",x_ab)
        print("y_ab",y_ab)
        # 在 K 方向计算均值和标准差
        return
    def decode_latent(self, z_before,y_pro):
        z, _ = self.decoder(z_before, attn_mask=None)
        dec_out = self.projector(z).permute(0,2,1)[:, :self.pred_len, :self.args.c_out] # filter the covariates
        dec_out=dec_out+y_pro
        return dec_out
    def vision(self,data1,data2,path):
        

        # 展平数据并绘制
        plt.figure(figsize=(8, 5))
        sns.histplot(data1.flatten(), kde=True, stat="density", color="skyblue")
        # sns.histplot(data1.flatten(), kde=True,color="skyblue")
        sns.histplot(data2.flatten(),kde=True,stat="density",color="green")
        plt.title("Global Distribution of Flattened Data")
        plt.xlabel("Value")
        plt.ylabel("Density")
        # plt.legend
        plt.savefig(path, bbox_inches='tight')
        return 
    def test(self,setting):
        test_data, test_loader = self._get_data(flag='test')
        self.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, '128vae.pth')))
        folder_path = './results/' + setting + '/'+'vae/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.eval()
        
        y_true, y_pred = [], []
        # print(mask_rate)
        with torch.no_grad():
            for i,(batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader): # Dataloader of thuml provides mark_x and mark_y, additionally
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                with torch.no_grad():
                    y_pro=self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                y_hat,z= self.forward(batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)
                # self.test_mean(z)
                y_pred.append(y_hat.cpu())
                y_true.append(batch_y.float().cpu())
                # pred=(batch_y-y_pro).float.cpu()
                # true=z.permute(0,2,1)
                # print(pred.shape())
                # print(true.shape())
                if i % 200 == 0:
                    pred=(batch_y-y_pro).float().cpu()
                    true=z.permute(0,2,1).cpu()
                    # input = batch_x.detach().cpu().numpy()
                    # if test_data.scale and self.args.inverse:
                    #     shape = input.shape
                    #     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt=pred[0,:,:]
                    pd=true[0,:,:]

                    # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    self.vision(pd,gt,c)
                    # visual(gt, pd, os.path.join(folder_path,str(i) + '.pdf'))
        
        y_true, y_pred = np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        vae_path=path+'/'+f'{self.args.vaed_model}vae.pth'
        torch.save(self.state_dict(), vae_path)
        metrics=metric(y_pred, y_true)
        print(f"Training and Test of vae on {self.args.dataset}, [mae, mse, rmse, mape, mspe] are: [{metrics}], respectively.")
       
        return 


def main(params):
    params.data_path = DATAPATH(params.dataset) # Datapath of each dataset is predefined.
    # assert params.train_type in [FORECAST, RECONSTRUCTION]
    
    
    setting = '{}_{}'.format(
                    params.model_id,
                    params.model)
    
    # for ii in range(params.itr):
    #     exp = Exp_Long_Term_Forecast(params)
    #     # model=model_dict[params.model].Model(params).float().to(device)
    #     print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    #     exp.train(setting)

    #     print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #     exp.test(setting)
    itvae = VaeIT(params)
    
    
    # itvae.fit()
    itvae.test(setting)
    
    
    
    


if __name__ == '__main__':
    parser = ArgumentParser()
    # basic config
    parser.add_argument("--beta_start",type=float,default=0.001)
    parser.add_argument("--beta_end",type=float,default=0.5)
    parser.add_argument("--num_steps",type=int,default=70)
    # parser.add_argument("--con_len",type=int,default=96)
    parser.add_argument("--dit_num",type=int,default=5)
    parser.add_argument("--dif_lr",type=float,default=1e-4)
    parser.add_argument("--dataset", type=str, default='electricity')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Informer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    # data loader
    parser.add_argument('--data', type=str, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='/home/admin/workspace/aop_lab/chiqiang/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='dataset/electricity/electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    #vae
    parser.add_argument('--vaed_model',type=int,default=32)
    parser.add_argument('--vae_embed',type=int,default=32)
    parser.add_argument('--vae_learning_rate',type=float,default=5e-5)
    parser.add_argument('--vae_epochs',type=int,default=8)
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=SEQ_LEN*2, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=SEQ_LEN, help='prediction sequence length')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('-gpu', "--gpu_device", type=int, default=0)
    parser.add_argument('-bs', "--batch_size", type=int, default=32)
    parser.add_argument('-lr', "--learning_rate", type=float, default=5e-4)
    parser.add_argument('-tre', "--train_epochs", type=int, default=30)
    parser.add_argument('-smp', "--save_model_parameters", action='store_true')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--channel_independence', type=bool, default=False, help='whether to use channel_independence mechanism')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')

    args = parser.parse_args()

    main(args)
from argparse import ArgumentParser
import json
import os
import torch, time, re
import numpy as np
from tqdm import tqdm
from torch import optim, nn
from layers.Embed import DataEmbedding_inverted
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from utils.globals import *
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from data_provider.data_factory import data_diff
from utils.tools import plot_loss
from utils.metrics import metric, MSE
from mix_CSDI import mix_csdi
from utils.tools import EarlyStopping_DF, adjust_learning_rate, visual
from testquick import Exp_Long_Term_Forecast
from vae import VaeIT
import numpy as np
class Re_Diffusion(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args=args
        self.emb_size=args.vaed_model
        self.d_model=args.vae_embed
        self.n_heads=4
        self.num_heads=4
        self.num_steps=args.num_steps
        self.seq_len=args.seq_len
        self.pred_len=args.pred_len
        self.device = device = torch.device('cuda:{}'.format(self.args.gpu_device))
        self.embedd=nn.Sequential(
            nn.Linear(4*self.seq_len,self.emb_size),
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.emb_size,
                    nhead=self.n_heads,
                    dim_feedforward=self.d_model,
                    dropout=self.args.dropout,
                    activation='gelu',
                    batch_first=True,
                ),
                num_layers=self.args.e_layers,
                norm=nn.LayerNorm(self.d_model)
            ),
            nn.Linear(self.d_model, self.d_model)
        )
        
        self.timeEncoder=nn.Linear(4,self.args.c_out)
        self.time_press=nn.Linear(self.args.pred_len,self.args.seq_len)
        self.vae = self._build_model().to(self.device)
        self.diffmodel=mix_csdi(args.dit_num, self.emb_size, self.num_heads)

        self.beta = np.linspace(args.beta_start, args.beta_end, args.num_steps)
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1).to(self.device)
        self.to(self.device)
    def timeEnc(self,timey):
        time=self.timeEncoder(timey).permute(0,2,1)
        time=self.time_press(time)
        return time
    def _get_data(self, flag):
        data_set, data_loader = data_diff(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.parameters(), lr=self.args.dif_lr)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion 

    def _build_model(self):
        model = VaeIT(self.args).float()
        setting='{}_{}'.format(
                    self.args.model_id,
                    self.args.model)
        path=os.path.join(self.args.checkpoints, setting)
        checkpoint_path=path+'/'+f'{self.args.vaed_model}vae.pth'
        model.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

        return model
    def forward(self,x_bef,x_mark_bef,x_enc,x_mark_enc,y_enc,x_mark_dec,batch_y): # every batch of x,y are supposed to be [B, L, C]
        
        with torch.no_grad():
            enc_y,y_pro,x_1,x_4,x_2=self.vae.encode_latent(x_bef,x_mark_bef,x_enc,x_mark_enc,y_enc,x_mark_dec,batch_y)
        time_y=x_mark_dec[:, :self.pred_len, :]

        timey=self.timeEnc(time_y)
        codi=torch.cat((x_4,x_1,timey,x_2),dim=2)

        condition=self.embedd(codi)
        B, _, _= condition.shape
        t = torch.randint(0, self.num_steps, [B]).to(self.device) 
        
        
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(enc_y).to(self.device) #[B,K,L]
        # print(enc_y.shape)

        noisy_data = (current_alpha ** 0.5) * enc_y + (1.0 - current_alpha) ** 0.5 * noise #[B K E]
        
        pred_y=self.diffmodel.condition(noisy_data, condition, t)
        

        
        # loss=nn.MSELoss()(pred_y, enc_y)

        return pred_y,enc_y
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.eval()
        with torch.no_grad():
            for batch_2x, batch_y, batch_2x_mark, batch_y_mark, in tqdm(vali_loader,desc='Testing'):
                x_before,batch_x=batch_2x[:,:self.seq_len,:],batch_2x[:,-self.seq_len:,:]
                x_before_mark,batch_x_mark=batch_2x_mark[:,:self.seq_len,:],batch_2x_mark[:,-self.seq_len:,:]
                x_before=x_before.float().to(self.device)
                x_before_mark=x_before_mark.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                batch_y = batch_y[:, -self.pred_len:, :].to(self.device)
                outputs,enc_y = self.forward(x_before,x_before_mark,batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)
                # outputs = outputs[:, -self.pred_len:, :]
                

                pred = outputs.detach().cpu()
                true = enc_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.train()
        return total_loss
                
    def fit(self,setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        train_steps = len(train_loader)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        early_stopping = EarlyStopping_DF(patience=self.args.patience, verbose=True)
        for epoch in range(self.args.train_epochs):
            train_loss=[]
            self.train()
            for batch_2x, batch_y, batch_2x_mark, batch_y_mark,in tqdm(train_loader, desc='Training'): # Dataloader of thuml provides mark_x and mark_y, additionally
                x_before,batch_x=batch_2x[:,:self.seq_len,:],batch_2x[:,-self.seq_len:,:]
                x_before_mark,batch_x_mark=batch_2x_mark[:,:self.seq_len,:],batch_2x_mark[:,-self.seq_len:,:]
                x_before=x_before.float().to(self.device)
                x_before_mark=x_before_mark.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                model_optim.zero_grad()
                
                outputs,enc_y = self.forward(x_before,x_before_mark,batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)
                # outputs = outputs[:, -self.pred_len:, :]
                
                loss = criterion(outputs, enc_y) # (x_hat, x_batch) for Reconstruction
                
                train_loss.append(loss.item())  # 累积每个batch的损失
                loss.backward()
                model_optim.step()
            
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 're_diffusion.pth'
        self.load_state_dict(torch.load(best_model_path))

        return 

    def forecast(self,x_bef,x_mark_bef,x_enc,x_mark_enc,y_enc,x_mark_dec,batch_y):
        

        with torch.no_grad():
            enc_y,y_pro,x_1,x_4,x_2=self.vae.encode_latent(x_bef,x_mark_bef,x_enc,x_mark_enc,y_enc,x_mark_dec,batch_y)
            
            time_y=x_mark_dec[:, -self.pred_len:, :]
            timey=self.timeEnc(time_y)
            
            codi=torch.cat((x_4,x_1,timey,x_2),dim=2)
            
            enc_x = self.embedd(codi)
            
        B, K, E = enc_x.shape
        current_sample = torch.randn_like(enc_x).to(self.device)

        t1 = torch.full((enc_x.size(0,),),self.num_steps).to(self.device)
        pred = self.diffmodel.condition(current_sample, enc_x, t1).to(self.device)

        current_sample = torch.clamp(pred,-1.0,1.0).detach()
        with torch.no_grad():
            dec_out = self.vae.decode_latent(current_sample,y_pro) # .permute(0, 2, 1)[:, :, :K] # filter the covariates
        
        return dec_out,y_pro
    
    def test(self,setting):
        test_data, test_loader = self._get_data(flag='test')
        self.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 're_diffusion.pth')))
        
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        self.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for i,(batch_2x, batch_y, batch_2x_mark, batch_y_mark) in enumerate(test_loader): # Dataloader of thuml provides mark_x and mark_y, additionally
                
                x_before,batch_x=batch_2x[:,:self.seq_len,:],batch_2x[:,-self.seq_len:,:]
                x_before_mark,batch_x_mark=batch_2x_mark[:,:self.seq_len,:],batch_2x_mark[:,-self.seq_len:,:]
                x_before=x_before.float().to(self.device)
                x_before_mark=x_before_mark.float().to(self.device)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs,_ = self.forecast(x_before,x_before_mark,batch_x, batch_x_mark, dec_inp, batch_y_mark,batch_y)
                # outputs = outputs[:, -self.args.pred_len:, :]
                pred = outputs.cpu()
                true = batch_y.float().cpu()
                
                y_pred.append(outputs.cpu())
                y_true.append(batch_y.float().cpu())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    # if test_data.scale and self.args.inverse:
                    #     shape = input.shape
                    #     input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
        y_true, y_pred = np.concatenate(y_true, axis=0), np.concatenate(y_pred, axis=0)
        
        mae,mse,rmse,mape,mspe=metric(y_pred, y_true)

        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting +"Re-diffusion"+"  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        return


def main(params):
    
    # device = torch.device(f"cuda:{params.gpu_device}" if torch.cuda.is_available() else "cpu")
    params.data_path = DATAPATH(params.dataset)
    # _, train_data_loader = data_provider(params, 'train')
    # _, valid_data_loader = data_provider(params, 'val')
    # _, test_data_loader = data_provider(params, 'test')
    # Only parameters of Diffusion further trained
    # opt = optim.AdamW(mtodel.parameters(), lr=params.learning_rate)
    # tls, vls = mtodel.fit(opt, params.num_epoch, train_data_loader, valid_data_loader)
    setting = '{}_{}'.format(
                    args.model_id,
                    args.model)
    for ii in range(params.itr):
        
        
        exp = Exp_Long_Term_Forecast(params)
        # model=model_dict[params.model].Model(params).float().to(device)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
    itvae = VaeIT(params)
    
    
    itvae.fit()
    itvae.test(setting)

    re_di=Re_Diffusion(params)
    print('>>>>>>>start training Re_diffusion>>>>>>>>>>>>>>>>>>>>>>>>>>')
    re_di.fit(setting)
    print('>>>>>>>testing <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    re_di.test(setting)
    # res_1,res_2 = mtodel.test(test_data_loader)
    # print("Results of multi-metrics (mae, mse, rmse, mape, mspe):", res_1)
    # print("Results of multi-metrics transformer (mae, mse, rmse, mape, mspe):",res_2)

    # current_datetime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(time.time())))
    # filestr = ''.join(re.findall(r'\d', current_datetime))
    # torch.save(mtodel.state_dict(), f'ckpts/diffusion/{params.dataset}/{filestr}.pth') # 保存模型参数
    # print(f"Model parameters saved in ckpts/diffusion/{params.dataset}/{filestr}.")
   
if __name__ == "__main__":
    parser = ArgumentParser()    
#     # Arguments for dataloaader of iTransformer
#     parser.add_argument('--data', type=str, default='ETTm2', help="Type of dataloader for different dataset file types.")
#     parser.add_argument('--embed', type=str, default='timeF')
#     parser.add_argument('--freq', type=str, default='h')
#     parser.add_argument('--root_path', type=str, default='/home/admin/workspace/aop_lab/chiqiang/')
#     parser.add_argument('--data_path', type=str, default='dataset/electricity/electricity.csv')
#     parser.add_argument('--seq_len', type=int, default=192)
#     parser.add_argument('--label_len', type=int, default=0, help="If no need label for decoder, set 0 by default") # 
#     parser.add_argument('--pred_len', type=int, default=SEQ_LEN)
#     parser.add_argument('--features', type=str, default='M')
#     parser.add_argument('--target', type=str, default='OT')
#     parser.add_argument('--num_workers', type=int, default=8)
#     parser.add_argument('--checkpoint_vae',type=str,default='/home/admin/workspace/aop_lab/chiqiang/ts/ckpts/ETTm2/vaenew20250205170019.pth')
#     parser.add_argument("--dataset", type=str, default='ETTm2')
#     parser.add_argument('-gpu', "--gpu_device", type=int, default=0)
#     parser.add_argument('-bs', "--batch_size", type=int, default=25)
#     parser.add_argument('-lr', "--learning_rate", type=float, default=1e-4)
#     parser.add_argument('-npc', "--num_epoch", type=int, default=50)
#     parser.add_argument('-smp', "--save_model_parameters", action='store_true')
#     parser.add_argument("--cycle",type=int,default=168)
#     args = parser.parse_args()
    #diffusion
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument("--beta_start",type=float,default=0.001)
    parser.add_argument("--beta_end",type=float,default=0.5)
    parser.add_argument("--num_steps",type=int,default=50)
    # parser.add_argument("--con_len",type=int,default=96)
    parser.add_argument("--dit_num",type=int,default=5)
    parser.add_argument("--dif_lr",type=float,default=1e-4)
    #train
    parser.add_argument("--dataset", type=str, default='electricity')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='Informer',
                        help='model name, options: [iTransformer, iInformer, iReformer, iFlowformer, iFlashformer]')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
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
    parser.add_argument('--vae_epochs',type=int,default=10)
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=SEQ_LEN*2, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=SEQ_LEN, help='prediction sequence length')

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('-gpu', "--gpu_device", type=int, default=0)
    parser.add_argument('-bs', "--batch_size", type=int, default=32)
    parser.add_argument('-lr', "--learning_rate", type=float, default=0.0001)
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

    
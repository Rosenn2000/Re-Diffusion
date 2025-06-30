from torch import nn 
import torch, math
import numpy as np

# class TimestepEmbedder(nn.Module):
#     """
#     Embeds scalar timesteps into vector representations.
#     """
#     def __init__(self, hidden_size, frequency_embedding_size=256):
    #     super().__init__()
    #     self.mlp = nn.Sequential(
    #         nn.Linear(frequency_embedding_size, hidden_size, bias=True),
    #         nn.SiLU(),
    #         nn.Linear(hidden_size, hidden_size, bias=True),
    #     )
    #     self.frequency_embedding_size = frequency_embedding_size

    # @staticmethod
    # def timestep_embedding(t, dim, max_period=10000):
    #     """
    #     Create sinusoidal timestep embeddings.
    #     :param t: a 1-D Tensor of N indices, one per batch element.
    #                       These may be fractional.
    #     :param dim: the dimension of the output.
    #     :param max_period: controls the minimum frequency of the embeddings.
    #     :return: an (N, D) Tensor of positional embeddings.
    #     """
    #     # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    #     half = dim // 2
    #     freqs = torch.exp(
    #         -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    #     ).to(device=t.device)
    #     args = t[:, None].float() * freqs[None]
    #     embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    #     if dim % 2:
    #         embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    #     return embedding

    # def forward(self, t):
    #     t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    #     t_emb = self.mlp(t_freq)
    #     return t_emb

# import math

class TimeEmbedding(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.half_emb_size=emb_size//2 if emb_size%2==0 else emb_size//2+1
        self.half_emb=torch.exp(torch.arange(self.half_emb_size)*(-1*math.log(10000)/(self.half_emb_size-1)))
        

    def forward(self,t):
        t=t.view(t.size(0),1)
        halfemb=self.half_emb.unsqueeze(0).expand(t.size(0),self.half_emb_size).to(t.device)
        half_emb_t=halfemb*t
        embs_t=torch.cat((half_emb_t.sin(),half_emb_t.cos()),dim=-1).to(t.device)
        
        return embs_t[:, :self.emb_size]



class DiTBlock(nn.Module):
    def __init__(self,emb_size,nhead):
        super().__init__()
        
        self.emb_size=emb_size
        self.nhead=nhead
        
        # conditioning
        self.gamma1=nn.Linear(emb_size,emb_size)
        self.beta1=nn.Linear(emb_size,emb_size)        
        self.alpha1=nn.Linear(emb_size,emb_size)
        self.gamma2=nn.Linear(emb_size,emb_size)
        self.beta2=nn.Linear(emb_size,emb_size)
        self.alpha2=nn.Linear(emb_size,emb_size)
        
        # layer norm
        self.ln1=nn.LayerNorm(emb_size)
        self.ln2=nn.LayerNorm(emb_size)
        
        # multi-head self-attention
        self.wq=nn.Linear(emb_size,nhead*emb_size) # (batch,seq_len,nhead*emb_size)
        self.wk=nn.Linear(emb_size,nhead*emb_size) # (batch,seq_len,nhead*emb_size)
        self.wv=nn.Linear(emb_size,nhead*emb_size) # (batch,seq_len,nhead*emb_size)
        self.lv=nn.Linear(nhead*emb_size,emb_size)
        
        # feed-forward
        self.ff=nn.Sequential(
            nn.Linear(emb_size,emb_size*4),
            nn.ReLU(),
            nn.Linear(emb_size*4,emb_size)
        )

    def forward(self,x,cond):   # x:(batch,seq_len,emb_size), cond:(batch,emb_size)
        # conditioning (batch,emb_size)
        gamma1_val=self.gamma1(cond)
        beta1_val=self.beta1(cond)
        alpha1_val=self.alpha1(cond)
        gamma2_val=self.gamma2(cond)
        beta2_val=self.beta2(cond)
        alpha2_val=self.alpha2(cond)
        
        # layer norm
        y=self.ln1(x) # (batch,seq_len,emb_size)
        
        # scale&shift
        y=y*(1+gamma1_val)+beta1_val
        
        # attention
        q=self.wq(y) 
        k=self.wk(y)    # (batch,seq_len,nhead*emb_size)    
        v=self.wv(y)    # (batch,seq_len,nhead*emb_size)
        
        
        q=q.view(q.size(0),q.size(1),self.nhead,self.emb_size).permute(0,2,1,3) # (batch,nhead,seq_len,emb_size)
        k=k.view(k.size(0),k.size(1),self.nhead,self.emb_size).permute(0,2,3,1) # (batch,nhead,emb_size,seq_len)
        v=v.view(v.size(0),v.size(1),self.nhead,self.emb_size).permute(0,2,1,3) # (batch,nhead,seq_len,emb_size)
        
        attn=q@k/math.sqrt(q.size(2))   # (batch,nhead,seq_len,seq_len)
        attn=torch.softmax(attn,dim=-1)   # (batch,nhead,seq_len,seq_len)
        y=attn@v    # (batch,nhead,seq_len,emb_size)
        y=y.permute(0,2,1,3) # (batch,seq_len,nhead,emb_size)
        y=y.reshape(y.size(0),y.size(1),y.size(2)*y.size(3))    # (batch,seq_len,nhead*emb_size)
        y=self.lv(y)    # (batch,seq_len,emb_size)
        
        # scale
        y=y*alpha1_val
        
        # redisual
        y=x+y  
        
        # layer norm
        z=self.ln2(y)
        # scale&shift
        z=z*(1+gamma2_val)+beta2_val
        # feed-forward
        z=self.ff(z)
        # scale 
        z=z*alpha2_val
        # residual
        return y+z

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size,hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size*2, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class mix_csdi(nn.Module):
    def __init__(
        self,
        dit_num:int=3,
        emb_size: int=512,
        head:int=4,
    ):
        super().__init__()
        self.emb_size=emb_size
        self.time_emb=nn.Sequential(
            TimeEmbedding(self.emb_size),
            nn.Linear(self.emb_size,self.emb_size*4),
            nn.ReLU(),
            nn.Linear(self.emb_size*4,self.emb_size)
        )
        #DiT Block
        self.dits=nn.ModuleList()
        for _ in range(dit_num):
            self.dits.append(DiTBlock(self.emb_size,head))
        # layer norm
        self.ln=FinalLayer(self.emb_size)
        
    def condition(self,enc_x,enc_y,t):
        # enc_x=self.emb(enc_x)
        
        #这里面输入是过完vae-encoder的部分
        B,K,E=enc_y.shape
        
        #use_norm
        ##构造时间步长t的embedding
        t=self.time_emb(t).unsqueeze(-1).to(enc_y.device) #[B E 1 ]
        # print("t",t.shape)
        t = t.repeat(1, K, 1) #[B E*K 1]
        # print("t",t.shape)
        
        t=t.reshape(B,K,E) #[B K E]
        cond=enc_y+t#（B K E )
        # dit blocks
        
        for dit in self.dits:
            x=dit(enc_x,cond)
        #layer norm
        x=self.ln(x,cond)
        return x
    def x_only(self,enc_x,t):
        # enc_x=self.emb(enc_x)
        B,K,E=enc_x.shape
        t=self.time_emb(t).unsqueeze(-1).to(enc_x.device) #[B E 1 ]
        # print("t",t.shape)
        t = t.repeat(1, K, 1) #[B E*K 1]
        # print("t",t.shape)
        
        t=t.reshape(B,K,E)
        for dit in self.dits:
            x=dit(enc_x,t)
        x=self.ln(x,t)
        # x=x.permute(0,2,1)
        return x
    def class_free(self,enc_x,cond,t,w=1):
        x_1=self.condition(enc_x,cond,t)
        x_2=self.x_only(enc_x,t)
        x_pred=(1+w)*x_1-w*x_2
        return x_pred

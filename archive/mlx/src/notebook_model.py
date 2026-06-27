import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OP_VOCAB = {"PAD":0,"add":1,"sub":2,"mul":3,"div":4,"eq":5,"const":6,"var":7,"finalanswer":8}
DIGIT_VOCAB = {"PAD":0,"0":1,"1":2,"2":3,"3":4,"4":5,"5":6,"6":7,"7":8,"8":9,"9":10,"NEG":11,"EOS":12}
DIGIT_VOCAB_SIZE = 13
MAX_DIGITS = 8
IDX2DIG = {v: k for k, v in DIGIT_VOCAB.items()}

class DenseGATLayer(nn.Module):
    def __init__(self, in_f, out_f, heads=4, concat=True, drop=0.1):
        super().__init__()
        self.heads=heads; self.out_f=out_f; self.concat=concat
        self.W=nn.Linear(in_f, heads*out_f, bias=False)
        self.as_=nn.Linear(out_f,1,bias=False); self.ad=nn.Linear(out_f,1,bias=False)
        self.drop=nn.Dropout(drop)
    def forward(self, x, adj):
        B,N,_=x.shape
        xp=self.W(x).reshape(B,N,self.heads,self.out_f)
        s=self.as_(xp).squeeze(-1); d=self.ad(xp).squeeze(-1)
        e=F.leaky_relu(s.unsqueeze(2)+d.unsqueeze(1),0.2)
        e=e.masked_fill(adj.unsqueeze(-1)==0,-1e4)
        
        attn=self.drop(F.softmax(e,dim=2))
        h=torch.einsum("bnjh,bjhd->bnhd",attn,xp)
        return h.reshape(B,N,self.heads*self.out_f) if self.concat else h.mean(2)

class GraphAwareBridge(nn.Module):
    def __init__(self, vsz, d, vf=2, gh=128, gl=3, heads=4):
        super().__init__()
        self.emb=nn.Embedding(vsz, d-vf)
        self.vp=nn.Linear(d,d)
        self.gats=nn.ModuleList()
        ind=d
        for i in range(gl):
            out=gh; co=True
            if i==gl-1: out=d; co=False
            self.gats.append(DenseGATLayer(ind,out,heads=heads,concat=co,drop=0.1))
            ind=out*(heads if co else 1)
    def forward(self, nids, nvals, adj):
        x=torch.cat([self.emb(nids), nvals], dim=-1)
        x=self.vp(x)
        pad=(nids==0)&(nvals.abs().sum(-1)==0)
        for layer in self.gats:
            pm=pad.unsqueeze(2)|pad.unsqueeze(1)
            a2=adj.clone(); a2[pm]=0.0
            x=F.elu(layer(x,a2))
        return x

def rms_norm(x, eps=1e-5):
    return x*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+eps).to(x.dtype)

class SwiGLU(nn.Module):
    def __init__(self, d, ex=2.0):
        super().__init__()
        i=int(round(ex*d*2/3)); i=(i+255)//256*256
        self.gu=nn.Linear(d,i*2,bias=False); self.dn=nn.Linear(i,d,bias=False)
    def forward(self,x):
        g,u=self.gu(x).chunk(2,dim=-1); return self.dn(F.silu(g)*u)

class HRMBlock(nn.Module):
    def __init__(self,d,h,ex=2.0):
        super().__init__()
        self.h=h; self.hd=d//h
        self.qkv=nn.Linear(d,3*d,bias=False); self.op=nn.Linear(d,d,bias=False)
        self.mlp=SwiGLU(d,ex)
    def forward(self,x,mask=None):
        B,N,D=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.h,self.hd)
        q,k,v=qkv.unbind(2); q,k,v=q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)
        s=(q@k.transpose(-2,-1))/math.sqrt(self.hd)
        if mask is not None: s=s+mask
        out=(F.softmax(s,dim=-1)@v).transpose(1,2).reshape(B,N,D)
        x=rms_norm(x+self.op(out)); x=rms_norm(x+self.mlp(x)); return x

class HRMModule(nn.Module):
    def __init__(self,nl,d,h,ex=2.0):
        super().__init__()
        self.layers=nn.ModuleList([HRMBlock(d,h,ex) for _ in range(nl)])
    def forward(self,hid,inj,mask=None):
        hid=hid+inj
        for l in self.layers: hid=l(hid,mask)
        return hid

class HRMForMath(nn.Module):
    def __init__(self,vsz=9,d=512,heads=8,Hc=4,Lc=8,Hl=8,Ll=8,ex=2.0,slen=50):
        super().__init__()
        self.Hc=Hc; self.Lc=Lc
        self.bridge=GraphAwareBridge(vsz,d,gh=128,gl=3,heads=4)
        self.pos=nn.Embedding(slen,d)
        self.Hmod=HRMModule(Hl,d,heads,ex); self.Lmod=HRMModule(Ll,d,heads,ex)
        self.Hi=nn.Parameter(torch.randn(d)*0.02)
        self.Li=nn.Parameter(torch.randn(d)*0.02)
        self.dhead=nn.Linear(d,MAX_DIGITS*DIGIT_VOCAB_SIZE)
        self.qhead=nn.Linear(d,2)
        nn.init.zeros_(self.qhead.weight); self.qhead.bias.data.fill_(-5.0)

    def forward(self,batch):
        ni=batch["node_ids"]; nv=batch["node_values"]; am=batch["adj_mask"]
        B,N=ni.shape
        xt=self.bridge(ni,nv,am)
        xt=xt+self.pos(torch.arange(N,device=ni.device).unsqueeze(0))
        pad=(ni==0)&(nv.abs().sum(-1)==0)
        amask=pad.float().unsqueeze(1).unsqueeze(1)*-1e4
        zH=self.Hi.unsqueeze(0).unsqueeze(0).expand(B,N,-1)
        zL=self.Li.unsqueeze(0).unsqueeze(0).expand(B,N,-1)
        with torch.no_grad():
            for h in range(self.Hc):
                for l in range(self.Lc):
                    if h==self.Hc-1 and l==self.Lc-1: continue
                    zL=self.Lmod(zL,zH+xt,amask)
                if h!=self.Hc-1: zH=self.Hmod(zH,zL,amask)
        zL=self.Lmod(zL,zH+xt,amask)
        zH=self.Hmod(zH,zL,amask)
        dl=self.dhead(zH).reshape(B,N,MAX_DIGITS,DIGIT_VOCAB_SIZE)
        ql=self.qhead(zH)
        return dl, ql[:,:,0], ql[:,:,1]

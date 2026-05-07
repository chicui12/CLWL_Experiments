from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

Array=np.ndarray

# ------------------------------------------------------------
# Pure-NumPy MNIST10 row-tilt experiment.
# No PyTorch dependency; uses a linear score model trained by Adam.
# ------------------------------------------------------------

def confusion_pairs() -> Dict[int,int]:
    return {0:6,6:0,1:7,7:1,2:5,5:2,3:8,8:3,4:9,9:4}

def validate_transition(M: Array, atol: float=1e-8):
    if M.min() < -atol:
        raise ValueError(f"negative transition entry {M.min()}")
    if not np.allclose(M.sum(axis=0), 1.0, atol=atol, rtol=0):
        raise ValueError(f"columns do not sum to one: {M.sum(axis=0)}")

def make_mtrue(c:int=10, modes:int=5) -> Array:
    pair=confusion_pairs(); d=c*modes
    correct=np.array([0.39900524108419216,0.10116230951437952,0.06466960063142882,0.04613398237868094,0.04138942377111112])
    paired=np.array([0.06191560447031741,0.04013388830373068,0.07555813012062282,0.03864030228596840,0.02953031318216359])
    M=np.zeros((d,c),float)
    for y in range(c):
        r=pair[y]
        for m in range(modes):
            M[modes*y+m,y]+=correct[m]
            M[modes*r+m,y]+=paired[m]
        rem=1-M[:,y].sum()
        others=[j for j in range(c) if j not in (y,r)]
        targets=[]
        for j in others:
            for m in range(1,modes): targets.append(modes*j+m)
        for idx in targets: M[idx,y]+=rem/len(targets)
    validate_transition(M); return M

def make_H(c:int=10, modes:int=5, scale:float=1.5)->Array:
    pair=confusion_pairs(); d=c*modes
    a=1.4005811812263396; b=1.6291827431030947; cneg=0.06761143669152814; dneg=4.470677536865842
    pos=np.array([a,b,0.8*b,0.5*b,0.3*b])
    neg=np.array([cneg,dneg,0.8*dneg,0.5*dneg,0.3*dneg])
    H=np.zeros((d,c),float)
    for y in range(c):
        r=pair[y]
        for m in range(modes):
            H[modes*r+m,y]+=pos[m]
            H[modes*y+m,y]-=neg[m]
    H=H-H.mean(axis=0,keepdims=True)
    return scale*H

def row_tilt(Mtrue:Array,H:Array,s:float)->Array:
    X=Mtrue*np.exp(s*H)
    M=X/X.sum(axis=0,keepdims=True)
    validate_transition(M); return M

def construct_T(M:Array, safety:float=0.95)->Array:
    N=np.linalg.pinv(M)  # c x d
    q=N.min(axis=0)
    T0=N-np.ones((N.shape[0],1))@q.reshape(1,-1)
    alpha=safety/T0.max() if T0.max()>0 else 1.0
    return alpha*T0

def fit_standard(A:Array)->dict:
    c=A.shape[0]
    X=np.zeros((c*c,1+c)); y=np.zeros(c*c); n=0
    for i in range(c):
        for j in range(c):
            X[n,0]=1.0 if i==j else 0.0; X[n,1+j]=1.0; y[n]=A[i,j]; n+=1
    sol=np.linalg.lstsq(X,y,rcond=None)[0]
    lam=float(sol[0]); v=sol[1:]
    Ahat=lam*np.eye(c)+np.ones((c,1))@v.reshape(1,-1)
    rel=float(np.linalg.norm(A-Ahat,'fro')/max(np.linalg.norm(A,'fro'),1e-12))
    margin=float(min(A[j,j]-np.max(np.delete(A[:,j],j)) for j in range(c)))
    return {"lambda":lam,"relative_residual":rel,"ranking_margin":margin}

def ce_col(p,q): return float(-np.sum(p*np.log(np.clip(q,1e-12,1))))

def forward_vertex_diag(Mtrue:Array,Mhat:Array)->dict:
    pair=confusion_pairs(); c=Mtrue.shape[1]
    acc=0; tp=[]; pp=[]; margins=[]
    for y in range(c):
        ces=np.array([ce_col(Mtrue[:,y],Mhat[:,k]) for k in range(c)])
        pred=int(np.argmin(ces)); acc+=pred==y
        prob=np.exp(-ces-(-ces).max()); prob/=prob.sum()
        tp.append(prob[y]); pp.append(prob[pair[y]]); margins.append(ces[pair[y]]-ces[y])
    return {"forward_vertex_acc":acc/c,"forward_true_prob":float(np.mean(tp)),"forward_pair_prob":float(np.mean(pp)),"forward_ce_margin_mean":float(np.mean(margins)),"forward_ce_margin_min":float(np.min(margins))}

def diagnostics(Mtrue,H,s_grid):
    rows=[]
    for s in s_grid:
        Mhat=row_tilt(Mtrue,H,s); T=construct_T(Mhat); A=T@Mtrue
        rows.append({"s":s,**fit_standard(A),**forward_vertex_diag(Mtrue,Mhat)})
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

def load_data(data_source:str, max_trainval:int|None, max_test:int|None, seed:int):
    rng=np.random.default_rng(seed)
    if data_source=='digits':
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        ds=load_digits(); X=ds.data.astype(np.float64)/16.0; y=ds.target.astype(int)
        Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.30,random_state=seed,stratify=y)
    elif data_source=='openml':
        from sklearn.datasets import fetch_openml
        mnist=fetch_openml('mnist_784',version=1,as_frame=False,parser='auto')
        X=mnist.data.astype(np.float64)/255.0; y=mnist.target.astype(int)
        Xtr,ytr=X[:60000],y[:60000]; Xte,yte=X[60000:],y[60000:]
    else:
        raise ValueError(data_source)
    if max_trainval is not None and len(Xtr)>max_trainval:
        idx=rng.choice(len(Xtr),max_trainval,replace=False); Xtr,ytr=Xtr[idx],ytr[idx]
    if max_test is not None and len(Xte)>max_test:
        idx=rng.choice(len(Xte),max_test,replace=False); Xte,yte=Xte[idx],yte[idx]
    # standardize using train stats
    mu=Xtr.mean(axis=0,keepdims=True); sd=Xtr.std(axis=0,keepdims=True)+1e-6
    return (Xtr-mu)/sd, ytr, (Xte-mu)/sd, yte

def split_train_val(X,y,seed,val_frac=0.2):
    from sklearn.model_selection import train_test_split
    return train_test_split(X,y,test_size=val_frac,random_state=seed,stratify=y)

def sample_z(y,Mtrue,seed):
    rng=np.random.default_rng(seed); z=np.empty_like(y)
    for i,yi in enumerate(y): z[i]=rng.choice(Mtrue.shape[0],p=Mtrue[:,int(yi)])
    return z.astype(int)

# ------------------------------------------------------------
# Pure NumPy linear training
# ------------------------------------------------------------

def softmax(S):
    S=S-S.max(axis=1,keepdims=True); E=np.exp(S); return E/E.sum(axis=1,keepdims=True)

def sigmoid(X): return 1/(1+np.exp(-np.clip(X,-40,40)))

def init_params(p,c,seed):
    rng=np.random.default_rng(seed)
    W=rng.normal(scale=0.01,size=(p,c)); b=np.zeros(c)
    return W,b

def predict(W,b,X): return X@W+b

def acc(W,b,X,y): return float(np.mean(np.argmax(predict(W,b,X),axis=1)==y))

def train_numpy(method,Xtr,ytr,ztr,Xval,yval,M_or_T,epochs=400,lr=0.03,wd=1e-4,seed=0,batch_size=512):
    n,p=Xtr.shape; c=10
    W,b=init_params(p,c,seed)
    mW=np.zeros_like(W); vW=np.zeros_like(W); mb=np.zeros_like(b); vb=np.zeros_like(b)
    beta1=0.9; beta2=0.999; eps=1e-8; t=0
    rng=np.random.default_rng(seed+123)
    best=(0,None,None,0)
    for ep in range(1,epochs+1):
        idx=rng.permutation(n)
        for start in range(0,n,batch_size):
            ii=idx[start:start+batch_size]; X=Xtr[ii]; z=ztr[ii]
            S=X@W+b
            if method=='clwl':
                T=M_or_T
                target=T[:,z].T  # batch x c
                G=(sigmoid(S)-target)/len(ii)
            else:
                M=M_or_T
                P=softmax(S)
                R=np.sum(P*M[z,:],axis=1,keepdims=True)
                G=P*(1-M[z,:]/np.clip(R,1e-12,None))/len(ii)
            gW=X.T@G + wd*W; gb=G.sum(axis=0)
            t+=1
            mW=beta1*mW+(1-beta1)*gW; vW=beta2*vW+(1-beta2)*(gW*gW)
            mb=beta1*mb+(1-beta1)*gb; vb=beta2*vb+(1-beta2)*(gb*gb)
            W-=lr*(mW/(1-beta1**t))/(np.sqrt(vW/(1-beta2**t))+eps)
            b-=lr*(mb/(1-beta1**t))/(np.sqrt(vb/(1-beta2**t))+eps)
        va=acc(W,b,Xval,yval)
        if va>best[0]+1e-9:
            best=(va,W.copy(),b.copy(),ep)
    return best[1],best[2],best[3]

def run(args):
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    s_grid=[float(x) for x in args.s_grid]
    Mtrue=make_mtrue(); H=make_H(scale=args.h_scale)
    np.save(out/'M_true.npy',Mtrue); np.save(out/'H.npy',H)
    diag=diagnostics(Mtrue,H,s_grid); diag.to_csv(out/'diagnostics.csv',index=False)
    print('=== Diagnostics ==='); print(diag)
    rows=[]
    for seed in args.train_seeds:
        Xtv,ytv,Xte,yte=load_data(args.data_source,args.max_trainval_samples,args.max_test_samples,seed)
        Xtr,Xval,ytr,yval=split_train_val(Xtv,ytv,seed)
        ztr=sample_z(ytr,Mtrue,seed+1000)
        # val weak labels unused for monitoring clean val acc
        W,b,ep=train_numpy('forward',Xtr,ytr,ztr,Xval,yval,Mtrue,epochs=args.epochs,lr=args.lr,wd=args.weight_decay,seed=seed+2000,batch_size=args.batch_size)
        oracle=acc(W,b,Xte,yte)
        for s in s_grid:
            Mhat=row_tilt(Mtrue,H,s); T=construct_T(Mhat)
            W,b,epc=train_numpy('clwl',Xtr,ytr,ztr,Xval,yval,T,epochs=args.epochs,lr=args.lr,wd=args.weight_decay,seed=seed+3000,batch_size=args.batch_size)
            clwl=acc(W,b,Xte,yte)
            W,b,epf=train_numpy('forward',Xtr,ytr,ztr,Xval,yval,Mhat,epochs=args.epochs,lr=args.lr,wd=args.weight_decay,seed=seed+4000,batch_size=args.batch_size)
            fwd=acc(W,b,Xte,yte)
            rows += [
                {'method':'CLWL_T_Mhat','seed':seed,'s':s,'test_accuracy':clwl,'best_epoch':epc},
                {'method':'Forward_Mhat','seed':seed,'s':s,'test_accuracy':fwd,'best_epoch':epf},
                {'method':'Forward_oracle_Mtrue','seed':seed,'s':s,'test_accuracy':oracle,'best_epoch':ep},
            ]
    raw=pd.DataFrame(rows); raw.to_csv(out/'raw_results.csv',index=False)
    summ=raw.groupby(['method','s'],as_index=False).agg({'test_accuracy':['mean','std'],'best_epoch':['mean','std']})
    summ.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in summ.columns]
    summ.to_csv(out/'summary_results.csv',index=False)
    print('=== Summary ==='); print(summ)
    plot(summ,diag,out/'results.png',out/'diagnostics.png')

def plot(summary,diag,result_path,diag_path):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for method in ['CLWL_T_Mhat','Forward_Mhat','Forward_oracle_Mtrue']:
        df=summary[summary.method==method].sort_values('s')
        x=df.s.to_numpy(float); y=df.test_accuracy_mean.to_numpy(float); e=np.nan_to_num(df.test_accuracy_std.to_numpy(float),nan=0)
        ax[0].plot(x,y,marker='o',label=method); ax[0].fill_between(x,y-e,y+e,alpha=.15)
    ax[0].set_xlabel('estimate-bias strength s'); ax[0].set_ylabel('test clean accuracy'); ax[0].set_ylim(0,1); ax[0].grid(True,alpha=.3); ax[0].legend()
    c=summary[summary.method=='CLWL_T_Mhat'].sort_values('s'); f=summary[summary.method=='Forward_Mhat'].sort_values('s')
    ax[1].plot(c.s,c.test_accuracy_mean.to_numpy(float)-f.test_accuracy_mean.to_numpy(float),marker='o')
    ax[1].axhline(0,ls='--',alpha=.7); ax[1].set_xlabel('s'); ax[1].set_ylabel('CLWL - Forward'); ax[1].grid(True,alpha=.3)
    fig.savefig(result_path,dpi=200,bbox_inches='tight'); plt.close(fig)
    fig,ax=plt.subplots(1,4,figsize=(17,4),constrained_layout=True)
    x=diag.s.to_numpy(float)
    for a,col,title in zip(ax,['lambda','relative_residual','ranking_margin','forward_true_prob'],['lambda','relative residual','CLWL ranking margin','Forward true/pair prob']):
        a.plot(x,diag[col],marker='o',label=col)
        if col=='forward_true_prob': a.plot(x,diag['forward_pair_prob'],marker='o',label='forward_pair_prob')
        a.grid(True,alpha=.3); a.set_xlabel('s'); a.set_title(title); a.legend()
    fig.savefig(diag_path,dpi=200,bbox_inches='tight'); plt.close(fig)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--data_source',choices=['digits','openml'],default='digits')
    p.add_argument('--out_dir',default='artifacts_mnist10_numpy_early_margin')
    p.add_argument('--train_seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--s_grid',nargs='+',type=float,default=[0,0.1,0.2,0.3,0.4,0.5,0.6])
    p.add_argument('--h_scale',type=float,default=1.5)
    p.add_argument('--epochs',type=int,default=400)
    p.add_argument('--lr',type=float,default=0.03)
    p.add_argument('--weight_decay',type=float,default=1e-4)
    p.add_argument('--batch_size',type=int,default=512)
    p.add_argument('--max_trainval_samples',type=int,default=None)
    p.add_argument('--max_test_samples',type=int,default=None)
    return p.parse_args()
if __name__=='__main__': run(parse_args())

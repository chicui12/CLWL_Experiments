#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, zipfile
from pathlib import Path
import numpy as np, pandas as pd, torch
import importlib.util

spec=importlib.util.spec_from_file_location('exp', str(Path(__file__).with_name('5.4clwl_cour_mnist4_experiment.py')))
exp=importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
torch.set_num_threads(1)

def make_synth(n, seed, strength):
    rng=np.random.default_rng(seed)
    M,Tn,B,sets,eta_star,Pset=exp.make_cour_style_transition(4, eps=1e-4)
    X=rng.normal(size=(n,8)).astype(np.float32)
    W=rng.normal(size=(8,4)); W/=np.maximum(np.linalg.norm(W,axis=0,keepdims=True),1e-12)
    logits=X@W + 0.25*np.sin(X[:,[0]]*np.array([[1.0,2.0,3.0,4.0]]))
    logits=(logits-logits.mean(1,keepdims=True))/(logits.std(1,keepdims=True)+1e-6)
    out=np.log(np.clip(eta_star,1e-8,1))[None,:]+strength*logits
    out-=out.max(1,keepdims=True); eta=np.exp(out); eta/=eta.sum(1,keepdims=True)
    y=exp.sample_y_from_eta(eta,seed+1); z=exp.sample_z_from_y(M,y,seed+2)
    idx=rng.permutation(n); tr=idx[:int(.6*n)]; va=idx[int(.6*n):int(.8*n)]; te=idx[int(.8*n):]
    return X,eta,y,z,tr,va,te,M,B,exp.construct_clwl_T(M,0.99),sets

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out_dir',default='/mnt/data/artifacts_final_cour_synthetic')
    ap.add_argument('--n',type=int,default=1500)
    ap.add_argument('--strength',type=float,default=0.8)
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--hidden_dim',type=int,default=64)
    ap.add_argument('--lr',type=float,default=1e-3)
    ap.add_argument('--batch_size',type=int,default=256)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]; diag=[]
    for seed in args.seeds:
        X,eta,y,z,tr,va,te,M,B,T,sets=make_synth(args.n,seed,args.strength)
        A_clpl=B.T@M; A_clwl=T@M
        diag.append({'seed':seed,'transform':'CLPL_native_T_M',**exp.evaluate_A(A_clpl,eta[te]),**{f'native_{k}':v for k,v in exp.fit_standard_form(A_clpl).items()},'dominance_violation_rate':exp.dominance_violation_rate(M,sets,eta[te])})
        diag.append({'seed':seed,'transform':'CLWL_T_M',**exp.evaluate_A(A_clwl,eta[te]),**{f'clwl_{k}':v for k,v in exp.fit_standard_form(A_clwl).items()},'dominance_violation_rate':exp.dominance_violation_rate(M,sets,eta[te])})
        for name,kind,obj in [('CLPL_native','clpl',B),('CLWL','clwl',T)]:
            model,info=exp.train_weak_model(kind,X[tr],z[tr],X[va],z[va],4,obj,args.hidden_dim,args.epochs,args.batch_size,args.lr,1e-4,'cpu',1000+10*seed+len(name))
            scores=exp.predict_logits(model,X[te],args.batch_size,'cpu')
            rows.append({'seed':seed,'method':name,**exp.evaluate_scores(scores,y[te],eta[te]),**info})
    raw=pd.DataFrame(rows); ddf=pd.DataFrame(diag)
    raw.to_csv(out/'raw_results.csv',index=False); ddf.to_csv(out/'diagnostics_raw.csv',index=False)
    summ=raw.groupby('method').agg(['mean','std']).reset_index(); summ.columns=['_'.join([str(x) for x in c if x]) for c in summ.columns]; summ.to_csv(out/'summary_results.csv',index=False)
    ds=ddf.groupby('transform').agg(['mean','std']).reset_index(); ds.columns=['_'.join([str(x) for x in c if x]) for c in ds.columns]; ds.to_csv(out/'diagnostics_summary.csv',index=False)
    (out/'meta.json').write_text(json.dumps(vars(args),indent=2))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    metrics=[('sampled_clean_accuracy','Sampled clean accuracy'),('eta_argmax_accuracy','Eta-argmax accuracy'),('pairwise_order_rate','Pairwise order rate')]
    for ax,(m,t) in zip(axes,metrics):
        for method in raw.method.unique():
            vals=raw[raw.method==method][m].to_numpy(float)
            ax.scatter(range(len(vals)),vals,label=method); ax.axhline(vals.mean(),ls='--',alpha=.7)
        ax.set_title(t); ax.set_ylim(0,1); ax.grid(True,alpha=.3)
    axes[0].legend(); fig.savefig(out/'synthetic_cour_results.png',dpi=200,bbox_inches='tight'); plt.close(fig)
    zip_path=out.with_suffix('.zip')
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in out.iterdir(): zf.write(p,arcname=p.name)
        zf.write(__file__,arcname=Path(__file__).name)
        zf.write(Path(__file__).with_name('5.4clwl_cour_mnist4_experiment.py'),arcname='5.4clwl_cour_mnist4_experiment.py')
    print('=== diagnostics ==='); print(ds.to_string(index=False))
    print('=== summary ==='); print(summ.to_string(index=False))
    print('Saved',out); print('Zip',zip_path)
if __name__=='__main__': main()

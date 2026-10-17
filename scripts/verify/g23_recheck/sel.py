import os
import pandas as pd, numpy as np, json
REPO='/work/submit/jinboz1/hnl/hnl-displaced'
FR=json.load(open(REPO+'/results/addendum_g_selection_v1.json'))['fr_star']
wp=json.load(open(REPO+'/results/working_points_v1.json'))
print('FR* in selection',FR, '| working_points keys', [k for k in wp if 'fr' in k.lower() or 'fake' in k.lower()][:5])
g=pd.read_csv(REPO+'/results/addendum_g_t3_val_grid_v1.csv',dtype={'params':str})
fam=pd.read_csv(REPO+'/results/addendum_g_t3_val_families_v1.csv')
def pick(t,n):
    e=t.eff_primary.max(); se=np.sqrt(e*(1-e)/n)
    tied=t[t.eff_primary>=e-se-1e-12]
    return tied.sort_values(['fake_rate_hnl','eff_primary'],ascending=[True,False]).iloc[0]
sel=json.load(open(REPO+'/results/addendum_g_selection_v1.json'))
for model,gm in g.groupby('model'):
    per={}
    for (f,s),t in gm.groupby(['family','seed']):
        ok=t[t.fake_rate_hnl<=FR+1e-12]
        if len(ok): per[(f,s)]=pick(ok,ok.n_primary.iloc[0])
    fams=sorted({f for f,_ in per if all((f,s) in per for s in (0,1,2))})
    rows=[]
    for f in fams:
        r=[per[(f,s)] for s in (0,1,2)]
        rows.append(dict(family=f,eff_primary=np.mean([x.eff_primary for x in r]),fake_rate_hnl=np.mean([x.fake_rate_hnl for x in r]),n=sum(x.n_primary for x in r),cfg=[x.params for x in r]))
    T=pd.DataFrame(rows)
    best=pick(T,T.n.iloc[0])
    print(model,'families at FR* in every seed:',len(T),'all families in grid:',gm.family.nunique())
    print(T[['family','eff_primary','fake_rate_hnl','n','cfg']].sort_values('eff_primary',ascending=False).to_string())
    print(' -> chosen',best.family,best.cfg)
    # compare against committed family table and selection file
    F=fam[fam.model==model].set_index('family')
    for _,r in T.iterrows():
        if r.family in F.index:
            a=F.loc[r.family]
            if abs(a.eff_primary-r.eff_primary)>1e-12 or abs(a.fake_rate_hnl-r.fake_rate_hnl)>1e-12: print('  family-table mismatch',r.family)
    print('  family table chosen:',F.index[F.chosen].tolist(),' families listed',len(F))
    m=sel['models'][model]
    print('  selection file:',m['family'],[x['config'] for x in m['seeds']],'guarded_alternative:',m.get('guarded_alternative'))
    print('  guard_arm_i per seed:',[x['prompt_guard']['guard_arm_i'] for x in m['seeds']])
    print('  ckpts:',[os.path.basename(x.get('gnn_ckpt') or '') for x in m['seeds']] if False else [ (x.get('canto_ckpt') or x.get('gnn_ckpt') or '').split('/')[-1] for x in m['seeds']])

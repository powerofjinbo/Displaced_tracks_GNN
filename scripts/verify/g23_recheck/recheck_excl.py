import numpy as np, pandas as pd, json, os, sys
R='/ceph/submit/data/user/j/jinboz1/hnl_lrt'
REPO='/work/submit/jinboz1/hnl/hnl-displaced'
T1=R+'/derived/generated_v1/t1'
def readlist(p):
    return np.array(sorted(int(l) for l in open(p) if l.strip() and not l.startswith('#')),dtype=np.int64)
iso_u=readlist(REPO+'/splits/generic_iso_union_test.txt')
iso_a=readlist(REPO+'/splits/generic_iso_test.txt'); iso_x=readlist(REPO+'/splits/generic_iso_x1_test.txt')
dense=readlist(REPO+'/splits/generic_dense_test.txt')
print('union',len(iso_u),'= t1',len(iso_a),'+ x1',len(iso_x), np.array_equal(iso_u,np.union1d(iso_a,iso_x)), len(np.intersect1d(iso_a,iso_x)),'dense',len(dense))
ti=pd.concat([pd.read_parquet(T1+'/truth_generic_iso.parquet'),pd.read_parquet(T1+'/truth_generic_iso_x1.parquet')],ignore_index=True)
td=pd.read_parquet(T1+'/truth_generic_dense.parquet')
def pop(t,ev):
    t=t[t.event.isin(ev)]
    g=t.gun.fillna(False).astype(bool)
    mine=g & t.is_target.astype(bool) & ((t.d0.abs()>=10)|(t.r_v.fillna(-1)>=100))
    return t, mine.to_numpy()
ti,Pi=pop(ti,iso_u); td,Pd=pop(td,dense)
EX=int(os.environ.get("EXCL","-1")); print("exclude iso truth event",EX,"(event_id",5*10**8+EX,") in union:",EX in iso_u,"P_G targets in it:",int((Pi & (ti.event.to_numpy()==EX)).sum()))
Pi=Pi & (ti.event.to_numpy()!=EX)
print('P_G mine',Pi.sum(),'in_pop col',ti.in_pop.sum(),'agree',(Pi==ti.in_pop.to_numpy(bool)).all())
print('P_dense mine',Pd.sum(),'in_pop col',td.in_pop.sum())
# bootstrap tables: own regeneration
def weights(n,seed):
    rng=np.random.default_rng(seed); idx=rng.integers(0,n,size=(500,n))
    w=np.zeros((500,n)); 
    for k in range(500): w[k]=np.bincount(idx[k],minlength=n)
    return w
Wi=weights(len(iso_u),12345); Wd=weights(len(dense),12346)
z=np.load(R+'/derived/bootstrap/generated_v1/generic_iso_union_test.npz'); print('stored iso table equal mine', np.array_equal(z['units'],iso_u), np.array_equal(z['w'],Wi))
z=np.load(R+'/derived/bootstrap/generated_v1/generic_dense_test.npz'); print('stored dense table equal mine', np.array_equal(z['units'],dense), np.array_equal(z['w'],Wd))
RUNS={'canto_ttbar':R+'/runs/addendum_g_v1/eval/canto_l_ttbar_s{}/test','acorn_g':R+'/runs/addendum_g_t4_v1/eval/acorn_g_s{}/test','canto_g':R+'/runs/addendum_g_t4_v1/eval/canto_g_s{}/test'}
exp_ids=np.sort(np.r_[5*10**8+iso_u, 6*10**8+dense])
res={}
for name,pat in RUNS.items():
    for s in range(3):
        d=pat.format(s)
        pp=pd.read_parquet(d+'/per_particle.parquet',columns=['event_id','particle_id','dm_matched'])
        pe=pd.read_parquet(d+'/per_event.parquet')
        cov=np.array_equal(np.sort(pe.event_id.to_numpy()),exp_ids)
        ndup=pp.duplicated(['event_id','particle_id']).sum()
        out={}
        for comp,t,P,code,ev,W in (('iso',ti,Pi,5,iso_u,Wi),('dense',td,Pd,6,dense,Wd)):
            tt=t[P]
            key=pd.DataFrame({'event_id':code*10**8+tt.event.to_numpy(np.int64),'particle_id':tt.particle_id.to_numpy(np.int64)})
            m=key.merge(pp.drop_duplicates(['event_id','particle_id']),how='left',on=['event_id','particle_id'])
            nmiss=m.dm_matched.isna().sum()
            k=m.dm_matched.fillna(False).astype(float).to_numpy()
            pos=np.searchsorted(ev,tt.event.to_numpy())
            num=np.bincount(pos,weights=k,minlength=len(ev)); den=np.bincount(pos,minlength=len(ev)).astype(float)
            pec=pe[(pe.event_id//10**8==code)&~((code==5)&(pe.event_id==5*10**8+EX))]; epos=np.searchsorted(ev,(pec.event_id%10**8).to_numpy())
            nf=np.bincount(epos,weights=pec.n_fake.to_numpy(float),minlength=len(ev)); nt=np.bincount(epos,weights=pec.n_tracks_ge5.to_numpy(float),minlength=len(ev))
            out[comp]=dict(eff=num.sum()/den.sum(),effrep=(W@num)/(W@den),fr=nf.sum()/nt.sum(),frrep=(W@nf)/(W@nt),n=int(den.sum()),miss=int(nmiss))
        res[(name,s)]=out
        print(name,s,'events',len(pe),'coverage_exact',cov,'dups',ndup,'iso eff %.4f%% fr %.4f%% n %d miss %d | dense eff %.4f%% fr %.4f%% miss %d'%(100*out['iso']['eff'],100*out['iso']['fr'],out['iso']['n'],out['iso']['miss'],100*out['dense']['eff'],100*out['dense']['fr'],out['dense']['miss']))
def model(name,comp,q='eff'):
    pts=np.array([res[(name,s)][comp][q] for s in range(3)]); reps=np.array([res[(name,s)][comp][q+'rep' if q=='eff' else 'frrep'] for s in range(3)])
    return pts,reps
def claim(A,B,label):
    pa,ra=model(A,'iso'); pb,rb=model(B,'iso')
    D=100*(pa.mean()-pb.mean())
    sb=100*np.std(ra.mean(0)-rb.mean(0),ddof=1)
    ss=100*np.sqrt(pa.var(ddof=1)/3+pb.var(ddof=1)/3)
    sig=np.hypot(sb,ss)
    fa=100*np.mean([res[(A,s)]['iso']['fr'] for s in range(3)]); fb=100*np.mean([res[(B,s)]['iso']['fr'] for s in range(3)])
    print(f'{label}: effA {100*pa.mean():.4f} effB {100*pb.mean():.4f} D {D:.4f} sigma {sig:.4f} (boot {sb:.4f} [ddof0 {100*np.std(ra.mean(0)-rb.mean(0)):.4f}], seeds {ss:.4f}) D-2s {D-2*sig:.4f} D/s {D/sig:.3f} frA {fa:.4f} frB {fb:.4f} dfr {fa-fb:.4f}')
    print('  seedsA',np.round(100*pa,3),'seedsB',np.round(100*pb,3))
claim('canto_ttbar','acorn_g','G2'); claim('canto_g','acorn_g','G3')
for n in RUNS:
    p,_=model(n,'dense'); print('dense',n,'%.4f'%(100*p.mean()))

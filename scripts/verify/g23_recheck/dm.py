# Independent double-majority matching from stored tracks.parquet + event hit tables; compares to stored per-particle/per-event tables.
import numpy as np, pandas as pd, glob, os, sys
R='/ceph/submit/data/user/j/jinboz1/hnl_lrt'
EV=R+'/derived/events/v1/hits'
hits={}
def load_hits(sample):
    fs=sorted(glob.glob(f'{EV}/{sample}/shard_*.parquet'))
    h=pd.concat([pd.read_parquet(f,columns=['event','row','pid']) for f in fs],ignore_index=True)
    h['event']=h.event.astype(np.int64); h['row']=h.row.astype(np.int64); h['pid']=h.pid.astype(np.int64)
    return h
H=pd.concat([load_hits('generic_iso').assign(code=5),load_hits('generic_dense').assign(code=6)],ignore_index=True)
H['event_id']=H.code*10**8+H.event
H=H[['event_id','row','pid']]
npart=H[H.pid!=0].groupby(['event_id','pid']).size().rename('nP').reset_index()
runs=sys.argv[1:]
for run in runs:
    bad_pp=bad_pe=tot_pp=tot_pe=0
    for b in sorted(glob.glob(run+'/blocks/*')):
        tr=pd.read_parquet(b+'/tracks.parquet'); tr=tr[tr.track>=0]
        tr=tr.merge(H,on=['event_id','row'],how='left')
        assert tr.pid.notna().all()
        nT=tr.groupby(['event_id','track']).size().rename('nT').reset_index()
        x=tr[tr.pid!=0].groupby(['event_id','track','pid']).size().rename('nTP').reset_index()
        x=x.merge(nT,on=['event_id','track']).merge(npart,on=['event_id','pid'])
        dm=x[(2*x.nTP>x.nT)&(2*x.nTP>x.nP)]
        pp=pd.read_parquet(b+'/per_particle.parquet',columns=['event_id','particle_id','dm_matched'])
        mine=pp[['event_id','particle_id']].merge(dm[['event_id','pid']].drop_duplicates().rename(columns={'pid':'particle_id'}).assign(m=True),how='left',on=['event_id','particle_id'])
        mm=mine.m.fillna(False).to_numpy(bool)
        bad_pp+=(mm!=pp.dm_matched.to_numpy(bool)).sum(); tot_pp+=len(pp)
        t5=nT[nT.nT>=5].copy()
        matched=dm[['event_id','track']].drop_duplicates().assign(ok=True)
        t5=t5.merge(matched,how='left',on=['event_id','track']); t5['fake']=t5.ok.isna()
        g=t5.groupby('event_id').agg(nt=('nT','size'),nf=('fake','sum')).reset_index()
        pe=pd.read_parquet(b+'/per_event.parquet').merge(g,how='left',on='event_id').fillna({'nt':0,'nf':0})
        bad_pe+=((pe.nt!=pe.n_tracks_ge5)|(pe.nf!=pe.n_fake)).sum(); tot_pe+=len(pe)
    print(run,'per-particle rows',tot_pp,'mismatched',bad_pp,'| per-event rows',tot_pe,'mismatched',bad_pe,flush=True)

import tarfile, json, glob, os
B='/ceph/submit/data/user/j/jinboz1/hnl_lrt/runs/baseline_g1_ts'
tot=0
for s in (0,1,2):
    for sp in ('train','val'):
        tp=f'{B}/armg_s{s}/ltf_union_graphs/{sp}set.tar'; tot+=os.path.getsize(tp)
        names=[m.name for m in tarfile.open(tp)]
        base=[os.path.basename(n) for n in names]
        blocks=sorted(glob.glob(f'{B}/armg_s{s}/ltf_union_graphs/blocks/{sp}_*'))
        prov=[json.load(open(b+'/provenance.json')) for b in blocks]
        nb=sum(p['n_events'] for p in prov)
        ok_fl=all(p['first'] in base and p['last'] in base for p in prov)
        pr=all(os.path.exists(b+'/pruned') for b in blocks); leftover=sum(len(glob.glob(b+'/*.tar')) for b in blocks)
        print(s,sp,'members',len(names),'unique',len(set(base)),'blocks',len(blocks),'sum n_events',nb,'first/last present',ok_fl,'all pruned',pr,'leftover block tars',leftover, 'n_blocks field',set(p['n_blocks'] for p in prov))
print('gathered total bytes',tot)

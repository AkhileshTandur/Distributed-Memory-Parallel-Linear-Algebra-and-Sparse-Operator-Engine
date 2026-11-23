"""Run actual measurements; fixed threads/rank, baseline is first rank count."""
import argparse,subprocess,os,json
p=argparse.ArgumentParser();p.add_argument('--exe',default='./build/spmv');p.add_argument('--n',type=int,default=1000000);p.add_argument('--threads',type=int,default=1);p.add_argument('--ranks',type=int,nargs='+',default=[1,2,4]);a=p.parse_args()
if a.threads<1 or any(r<1 for r in a.ranks) or a.ranks[0]!=1: p.error('positive threads/ranks and first rank count must be 1')
base=None
for r in a.ranks:
    out=subprocess.check_output(['mpiexec','-n',str(r),a.exe,str(a.n),'100'],env={**os.environ,'OMP_NUM_THREADS':str(a.threads)},text=True)
    item=json.loads(out.strip());base=base or item['seconds'];item['speedup']=base/item['seconds'];item['parallel_efficiency']=item['speedup']/r
    print(json.dumps(item),flush=True)

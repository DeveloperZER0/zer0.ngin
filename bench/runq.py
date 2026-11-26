# runq.py NAME... : serialized runs of llama-server variants through bench.py (one GPU job at a time, Windows)
import subprocess, sys, json, time, os, socket, urllib.request
HERE=os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
OLD=os.path.expandvars(r'%LOCALAPPDATA%\Programs\llama.cpp-Inkling'); NEW=r'C:\src\lcpp-new\x'
M=OLD+r'\models\Inkling-2-Mini-UD-Q4_K_M.gguf'; PORT='18081'
com=['-m',M,'--alias','inkling-2-mini','--device','CUDA0','--host','127.0.0.1','--port',PORT,'-c','32768','-np','1','--jinja','-fa','on','-t','12','-tb','12','--no-mmap'] 
ot=['-ctk','q8_0','-ctv','q8_0','--fit','off','-ngl','99','--override-tensor',r'blk\.(0|1|2|3|4|5|1[0-9]|2[0-3])\.ffn_(up|down|gate_up|gate)_(ch|)exps=CPU,blk\.9\.ffn_down.*=CPU']
mm=['--mmproj',OLD+r'\models\mmproj-BF16.gguf']
V={
 'ctl':        (OLD,com+mm+ot+['-b','512','-ub','256']),                       # the current 32k profile
 'ctl-nomm':   (OLD,com+ot+['-b','512','-ub','256']),
 'cache-cpu0': (OLD,com+['-ctk','q8_0','-ctv','q8_0','--cpu-moe','-ngl','99','--moe-cache-mib','3500','-b','2048','-ub','512']),
 'cache-cpu50':(OLD,com+['-ctk','q8_0','-ctv','q8_0','--cpu-moe','-ngl','99','--moe-cache-mib','3500','--moe-cache-cpu-share','0.5','-b','2048','-ub','512']),
 'cache-cpu100':(OLD,com+['-ctk','q8_0','-ctv','q8_0','--cpu-moe','-ngl','99','--moe-cache-mib','3500','--moe-cache-cpu-share','1.0','-b','2048','-ub','512']),
}
V['ctl-nomm-ub512']=(OLD,com+ot+['-b','2048','-ub','512'])
for k in list(V): V['new-'+k]=(NEW,V[k][1])
def main(names, reps, pl):
    for n in names:
        base=n.split('@')[0]
        if base=='ink': d,args=os.path.abspath('../bin'),['-m',M,'--port',PORT]
        else: d,args=V[base]; args=[a for a in args]
        if '--no-mmap' in args: args.remove('--no-mmap'); args+=['--load-mode','none']
        lf=open(f'logs/{n.replace("@","_")}.log','wb')
        p=subprocess.Popen([(d+r'\zer0.ngin.exe') if base=='ink' else (d+r'\llama-server.exe')]+args,cwd=d,stdout=lf,stderr=lf)
        try:
            ok=False
            for _ in range(180):
                try:
                    urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=2).read(); ok=True; break
                except Exception:
                    if p.poll() is not None: break
                    time.sleep(1)
            if not ok: print(n,'FAILED TO START'); continue
            cfg={'name':n,'base_url':f'http://127.0.0.1:{PORT}','model':'inkling-2-mini','concurrency':1,'prefill_lengths':pl,'temperature':0.6,'extra_body':{'reasoning_effort':'low','seed':1}}
            json.dump(cfg,open(f'configs/{n.replace("@","_")}.json','w'),indent=1)
            print('vram',subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),flush=True)
            subprocess.run([sys.executable,'bench.py','run',f'configs/{n.replace("@","_")}.json','--skip-cN','--reps',str(reps)]+LONG)
        finally:
            p.terminate(); p.wait(); time.sleep(4)
LONG=['--long','24000']
if __name__=='__main__':
    main(sys.argv[3:],int(sys.argv[1]),json.loads(sys.argv[2]))

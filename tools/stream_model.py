"""Cycle-level Python model of systolic_array_ws_4x4 (same register timing as the RTL).
Checks: weight swap via 'first' token, row skew, column deskew, latency, B>=7 rule."""
import numpy as np, sys
N=4
def run(tiles_W, A_tiles, B, gap_after_last=12):
    # tiles_W[t][k][c], A_tiles[t][i][k]; returns rows emitted (cycle, row[4]) and expected sum
    T=len(tiles_W)
    # state
    w=np.zeros((N,N),int)
    act=np.zeros((N,N+1),int); val=np.zeros((N,N+1),int); fst=np.zeros((N,N+1),int)
    ps=np.zeros((N+1,N),int)         # ps[k][c] = psum_out of PE(k-1,c) ; ps[0]=0
    # PE output regs
    act_o=np.zeros((N,N),int); val_o=np.zeros((N,N),int); fst_o=np.zeros((N,N),int); ps_o=np.zeros((N,N),int)
    # input skew regs: delay k for row k (list of fifos)
    sk_a=[[0]*k for k in range(N)]; sk_v=[[0]*k for k in range(N)]; sk_f=[[0]*k for k in range(N)]
    dsk=[[0]*(N-1-c) for c in range(N)]
    stream=[(t,i) for t in range(T) for i in range(B)]
    rows=[]; cyc=0
    total=len(stream)+gap_after_last+20
    for cyc in range(total):
        # presented vector
        if cyc<len(stream):
            t,i=stream[cyc]; v=1; f=1 if i==0 else 0
            araw=[A_tiles[t][i][k] for k in range(N)]; bus=tiles_W[t]
        else:
            v=0;f=0;araw=[0]*N; bus=tiles_W[T-1]   # bus holds last tile
        # skew outputs (visible this cycle)
        ain=[];vin=[];fin=[]
        for k in range(N):
            if k==0: ain.append(araw[0]);vin.append(v);fin.append(f)
            else: ain.append(sk_a[k][-1]);vin.append(sk_v[k][-1]);fin.append(sk_f[k][-1])
        # combinational PE inputs
        nxt_act_o=act_o.copy();nxt_val_o=val_o.copy();nxt_fst_o=fst_o.copy();nxt_ps_o=ps_o.copy();nw=w.copy()
        for k in range(N):
            for c in range(N):
                a_in = ain[k] if c==0 else act_o[k][c-1]
                v_in = vin[k] if c==0 else val_o[k][c-1]
                f_in = fin[k] if c==0 else fst_o[k][c-1]
                p_in = 0 if k==0 else ps_o[k-1][c]
                weff = bus[k][c] if f_in else w[k][c]
                nxt_val_o[k][c]=v_in; nxt_fst_o[k][c]=1 if (f_in and v_in) else 0
                if v_in:
                    if f_in: nw[k][c]=bus[k][c]
                    nxt_act_o[k][c]=a_in; nxt_ps_o[k][c]=p_in+a_in*weff
        # output aligned row (visible now): col c delayed by (N-1-c) regs
        row=[]
        for c in range(N):
            row.append(ps_o[N-1][c] if c==N-1 else dsk[c][-1])
        rv=val_o[N-1][N-1]
        if rv: rows.append((cyc,row))
        # clock edge
        for k in range(1,N):
            sk_a[k]=[araw[k]]+sk_a[k][:-1]; sk_v[k]=[v]+sk_v[k][:-1]; sk_f[k]=[f]+sk_f[k][:-1]
        for c in range(N-1):
            dsk[c]=[ps_o[N-1][c]]+dsk[c][:-1]
        act_o,val_o,fst_o,ps_o,w=nxt_act_o,nxt_val_o,nxt_fst_o,nxt_ps_o,nw
    return rows
def check(B,T=6,seed=0):
    rng=np.random.default_rng(seed)
    W=[rng.integers(-128,128,(N,N)) for _ in range(T)]
    A=[rng.integers(-128,128,(B,N)) for _ in range(T)]
    rows=run(W,A,B)
    ok=len(rows)==T*B
    first_cycle=rows[0][0] if rows else None
    for n,(cyc,row) in enumerate(rows):
        t,i=divmod(n,B)
        exp=A[t][i]@W[t]
        if list(exp)!=list(row): ok=False
        if cyc!=n+7: ok=False
    return ok,first_cycle,len(rows)
for B in (4,6,7,8,16):
    print("B=%d"%B, check(B))

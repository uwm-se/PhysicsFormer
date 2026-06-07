import h5py, json
from collections import Counter
H5 = (r"C:\Users\jpoko\source\repos\homework\CascadeProjects\windsurf-project"
      r"\compsac_2026_code\data\clevrer_training_expanded.h5")
def dec(x): return x.decode("utf-8") if isinstance(x,(bytes,bytearray)) else str(x)
with h5py.File(H5,"r") as f:
    n=f["questions"].shape[0]
    qt=[dec(x) for x in f["questions"].shape[0] and f["question_types"][:]]
    print("total samples:", n)
    print("ALL question_types:", Counter(qt))
    md=[dec(x) for x in f["metadata"][:]]
    # scene_index presence + range
    has=0; smin=10**9; smax=-1
    causal_idx=[]
    CAUSAL={"counterfactual","explanatory","predictive"}
    for i,(t,m) in enumerate(zip(qt,md)):
        try: mm=json.loads(m)
        except: mm={}
        s=mm.get("scene_index")
        if s is not None:
            has+=1; smin=min(smin,s); smax=max(smax,s)
        if t.lower() in CAUSAL: causal_idx.append(i)
    print(f"metadata has scene_index: {has}/{n}  range [{smin},{smax}]")
    print("causal-typed (exact match counterfactual/explanatory/predictive):", len(causal_idx))
    # show a causal sample fully (or whichever non-descriptive)
    for i,t in enumerate(qt):
        if "descriptive" not in t:
            print("\nNON-DESCRIPTIVE sample idx",i,"type",t)
            print("  Q:",dec(f["questions"][i])[:160])
            print("  A:",dec(f["answers"][i])[:120])
            print("  META:",md[i][:300])
            break

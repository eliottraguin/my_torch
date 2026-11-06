import sys
from collections import Counter
truth=[" ".join(l.split()[6:]) for l in open(sys.argv[1]) if l.strip()]
pred=[l.strip() for l in open(sys.argv[2])]
assert len(truth)==len(pred)
ok=Counter(t for t,p in zip(truth,pred) if t==p); tot=Counter(truth)
print(f"{sum(ok.values())/len(truth)*100:6.2f} %  {sys.argv[3]}   " + "  ".join(f"{k}:{ok[k]/tot[k]*100:.0f}" for k in sorted(tot)))

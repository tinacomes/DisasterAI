"""Does the model reach steady state at T=200? Run to T=600 and look."""
import sys, json, random
import numpy as np
sys.path.insert(0, '/home/user/DisasterAI')
from DisasterAI_Model import DisasterModel

BASE = dict(share_exploitative=0.5, share_of_disaster=0.15, initial_trust=0.3,
            initial_ai_trust=0.25, number_of_humans=100, share_confirming=0.7,
            disaster_dynamics=2, width=30, height=30, learning_rate=0.1,
            epsilon=0.3, exploit_trust_lr=0.015, explor_trust_lr=0.03,
            rumor_probability=1.0, ticks=600)
CFG = dict(main=dict(mobility=1, network_type='spatial_bridged', query_scope='network'),
           control=dict(mobility=0, network_type='components', query_scope='global'))
res = {}
for cfg, flags in CFG.items():
    for alpha in (0.0, 1.0):
        runs = []
        for seed in (0, 1):
            random.seed(seed); np.random.seed(seed)
            m = DisasterModel(**{**BASE, **flags, 'ai_alignment_level': alpha})
            rec = {k: [] for k in ('tick','seci_ex','seci_er','ie_ex','ie_er','pool_ex','pool_er')}
            for t in range(BASE['ticks']):
                m.step()
                if t % 5 == 0:
                    rec['tick'].append(t)
                    s = m.seci_data[-1] if m.seci_data else (0, np.nan, np.nan)
                    i = m.aeci_ie_data[-1] if m.aeci_ie_data else (0, np.nan, np.nan)
                    b = m.belief_pool_data[-1] if m.belief_pool_data else (0, np.nan, np.nan)
                    rec['seci_ex'].append(float(s[1])); rec['seci_er'].append(float(s[2]))
                    rec['ie_ex'].append(float(i[1]));   rec['ie_er'].append(float(i[2]))
                    rec['pool_ex'].append(float(b[1])); rec['pool_er'].append(float(b[2]))
            runs.append(rec)
            print(f"{cfg} a={alpha} seed={seed} done", flush=True)
        agg = {k: np.nanmean([r[k] for r in runs], axis=0).tolist()
               for k in runs[0] if k != 'tick'}
        agg['tick'] = runs[0]['tick']
        res[f"{cfg}|{alpha}"] = agg
        json.dump(res, open(sys.argv[1], 'w'))
print("DONE")

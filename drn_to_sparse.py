import re
import numpy as np
import scipy.sparse as sp

# storm --prism crowds.pm --constants "TotalRuns=3,CrowdSize=10" --buildfull --prismcompat --engine sparse --exportbuild test_base.drn
'''
storm --prism brp.pm --constants N=16,MAX=2 --buildfull --prismcompat --engine sparse --exportbuild brp_16_2.drn
storm --prism brp.pm --constants N=16,MAX=2 --buildfull --prismcompat --engine dd --exportbuild brp_16_2.drdd
'''


def read_drn(filename):
    with open(filename) as f:
        content = f.read()

    # currently does not capture rewards
    pat_with_sqr = r"state (\d+)(?: \[[\d., ]+\])? ?([\w ]*)\n\taction 0(?: \[[\d., ]+\])?\n([\s\d:.\/]*)"

    rows_match = re.finditer(pat_with_sqr, content)
    num_states = int(re.findall(r"@nr_states\n(\d+)",content)[0])
    
    t_mat = sp.dok_array((num_states, num_states))
    label_arrs = {}
    for match in rows_match:
        num_states += 1
        idx = int(match.group(1).strip())
        labels = match.group(2).strip().split()
        for l in labels:
            if l not in label_arrs:
                label_arrs[l] = [idx]
            else:
                label_arrs[l].append(idx)
        body = match.group(3).strip()
        ts_strs = re.finditer(r"(\d+) : ([\d.\/]+)", body)
        for ts_match in ts_strs:
            t = int(ts_match.group(1))
            if '/' in ts_match.group(2):
                raise ValueError("Can't process rational fractions")
            p = float(ts_match.group(2))
            t_mat[idx, t] = p
            #print(f'{idx} - {t}:{p} ({labels})')

    for k, v in label_arrs.items():
        label_arrs[k] = np.array(v, dtype=int)
    transitions = t_mat.tocsr()
    label_arrs['trans'] = transitions 
    assert transitions.nnz > 0, "Input drn was not processed correctly"
    
    return label_arrs

if __name__ == '__main__':
    filename = "dtmcs/robot.drn"
    res = read_drn(filename)
    print(res['init'])
    print(res['target'])
    # sp.save_npz("dice_mat.npz", transitions)

import re

entry_pat = r"--- \w+\/([a-zA-Z]+)_?([\S_]+)\.drdd - (\d+) ---\n"

output_pat = r"[\w ]+input: ([\d.]+)ms.\
[\w ]+state: ([\d.]+)\
[\w ]+ADD: ([\d.]+) nodes\
[\w ]+functions: ([\d.]+)ms.\
(?:[#\d\-\s]*Taken ([\d.]+)ms per sample|No matching traces)"

head = "dtmcs/logs/"
fnames = '''add_sampling-add_brp-17220919.out     add_sampling-add_herman-17235107.out  add_sampling-nand-17220736.out
add_sampling-add_crowds-17235089.out  add_sampling-add_leader-17235755.out 
add_sampling-add_egl-17235075.out     add_sampling-add_nand-17235081.out  '''.split()


res = [] # model, params, length, output
for fname in fnames:
    with open(head+fname) as f:
        content = f.read()
    seq = re.split(entry_pat, content) # [preamble, model, params, length, content, model,...]
    for i in range(1, len(seq), 4):
        name, params, length, output_content = seq[i:i+4]
        output_type = "ok"
        parsetime = varnum = addsize = precomptime = tracetime = '-1'
        if "Segmentation fault" in output_content:
            output_type = "segfault"
        elif "CANCELLED" in output_content:
            output_type = "timeout"
        elif "CUDD appears to have run out of memory." in output_content:
            output_type = "mem"
        else:
            stats = re.search(output_pat, output_content)
            if stats:
                parsetime, varnum, addsize, precomptime, tracetime = stats.groups()
            else:
                print(f"Issue processing {name}-{params} ({length}): {output_content}")
            if not tracetime:
                tracetime = "-1"
        res.append(','.join([name, params, length, output_type,
                            parsetime, varnum, addsize, precomptime, tracetime]))

fname = 'dtmcs/add_timing.csv'
with open(fname, 'w') as f:
    f.write("name,params,length,output_type,parsetime,vars_per_state,add_size,precomp_time,trace_time\n")
    f.write('\n'.join(res))

print(f"Written {len(res)} entries to {fname}")
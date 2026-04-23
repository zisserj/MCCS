import re
import numpy as np

with open('presentation/bee_traces_99.txt') as f:
    content = f.read()
labels_content, traces_content = content.split('---')


matches = re.finditer(r'([\w]+): \[([\d+,\s]+)\]', labels_content)


labels_dict = {}
for m in matches:
    labels_dict[m.group(1)] = np.fromstring(m.group(2), sep=' ', dtype=int)


count_on6 = 0
total = 0
traces = traces_content.split('\n')[1:]
for t in traces:
    t_arr = np.fromstring(t, sep=',', dtype=int)
    state_visited = np.isin(t_arr, labels_dict['all_visited'])
    finished_idx = np.searchsorted(state_visited,True)
    if t_arr[finished_idx] in labels_dict['on_hand6']:
        count_on6 += 1
        total += finished_idx
print(total/count_on6)
# %%
import os.path as op
import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import re
# %%
data_root = '../../HAD-MEEG_data'
res_dir = '../../HAD-MEEG_results'
upload_dir = '../../HAD-MEEG_upload/derivatives/detailed_events'
# %%
bh_meg_dir = op.join(data_root, 'MEG', 'behaviour')
bh_eeg_dir = op.join(data_root, 'EEG', 'behaviour')
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

class_info = pd.read_csv('../class_info.csv')
HACS_info = pd.read_csv('../HACS_clips_v1.1.1.csv')
HACS_info = HACS_info[HACS_info['subset']=='training']

meg_subs = sorted([i[-2:] for i in os.listdir(bh_meg_dir) if 'sub' in i])
eeg_subs = sorted([i[-2:] for i in os.listdir(bh_eeg_dir) if 'sub' in i])
runs = ['{0:0>2d}'.format(run) for run in np.arange(1,9)]
eeg_subs.remove('10')  # sub10 has no enough behavioural data

meg_bh_fps = {
    sub: [
        op.join(bh_meg_dir,f"sub{sub}", "sess01", f"sub{sub}_sess01_run{run}.mat") for run in runs
    ] for sub in meg_subs
}
eeg_bh_fps = {
    sub: [
        op.join(bh_eeg_dir,f"sub{sub}", "sess01", f"sub{sub}_sess01_run{run}.mat") for run in runs
    ] for sub in eeg_subs
}
#%%
def compute_beh(p_mat):
    data = loadmat(p_mat)
    # Extract relevant information from the .mat file
    # This will depend on the structure of your .mat files
    # For example:
    # beh_data = data['behavioural_data']
    # return beh_data
    class_name = data["runClass"]
    truth_label = [
        class_info['sports_label'][
            class_info['className']==name[0][0]
            ].iloc[0] for name in class_name
        ]
    resp_mat = data['trial']
      # nTrial * 7 array: [onset, class, dur, key, RT, realTimePresent, realTimeFinish]
    response = resp_mat[:,3] # every trial's response key
    n_miss = len(response) - np.sum(response!=0)
    response_rate = 1 - (n_miss / len(response))
    response_acc = ( response==truth_label ).sum() / ( len(response) - n_miss )
    reaction_time = np.mean(resp_mat[:, 4])
    return {
        'response_rate': response_rate,
        'response_acc': response_acc,
        'reaction_time': reaction_time
    }
def generate_meta_data(p_mat):

    data = loadmat(p_mat)
    resp_key = data['trial'][:,3]
    class_name = [i[0] for i in data["runClass"].flatten().tolist()]
    super_class_name = [
        class_info[
            class_info['className']==name
            ]['superClassName'].iloc[0] for name in class_name
    ]
    stim_is_sports = [
        class_info[
            class_info['className']==name
            ]['sports_label'].iloc[0] for name in class_name
    ]
    stim_is_sports = [True if i==1 else False for i in stim_is_sports]
    resp_is_sports = [True if i==1 else False for i in resp_key]
    resp_is_right = [
        stim==resp for stim, resp in zip(stim_is_sports, resp_is_sports)
    ]
    videos = data['runStim']
    videos_idx = [
        re.search(r'id_(.*?)_start', str(i)).group(1)
        for i in videos.flatten().tolist()
    ]
    video_class_id = [class_info[
        class_info['className']==name
        ]['classID'].iloc[0] for name in class_name]
    video_super_class_id = [
        class_info[
            class_info['className']==name
            ]['superClassID'].iloc[0] for name in class_name]
    
    RT = data['trial'][:,4]
    response = [True if i!=0 else False for i in resp_key]
    meta_data = {
        'event_name': 'video_on',
        'task': 'action',
        'subject': p_mat.split('/')[-3][-2:],
        'session': p_mat.split('/')[-5].lower(),
        'run': p_mat.split('/')[-1].split('_')[-1].split('.')[0][-2:],
        'video_id': videos_idx,
        'class_id': video_class_id,
        'super_class_id': video_super_class_id,
        'class_name': class_name,
        'super_class_name': super_class_name,
        'is_resp': response,
        'stim_is_sports': stim_is_sports,
        'resp_is_sports': resp_is_sports,
        'resp_is_right': resp_is_right,
        'resp_time': RT,
    }
    return pd.DataFrame(meta_data)

def compute_detailed_beh():
    meg_res = []
    eeg_res = []
    for sub in tqdm(meg_subs, desc="Computing MEG detailed behaviour"):
        for run in runs:
            meg_res.append({'sub': sub, 'run': run})
            p_mat = meg_bh_fps[sub][int(run)-1]
            meg_res[-1].update(compute_beh(p_mat))

    for sub in tqdm(eeg_subs, desc="Computing EEG detailed behaviour"):
        for run in runs:
            eeg_res.append({'sub': sub, 'run': run})
            p_mat = eeg_bh_fps[sub][int(run)-1]
            eeg_res[-1].update(compute_beh(p_mat))

    meg_res = pd.DataFrame(meg_res)
    eeg_res = pd.DataFrame(eeg_res)
    os.makedirs(op.join(res_dir, 'behaviour'), exist_ok=True)
    return meg_res, eeg_res

def generate_detailed_events():
    all_subs = sorted(list(set(meg_subs).union(set(eeg_subs))))

    for sub in tqdm(all_subs, desc="Generating detailed events"):
        sub_meta = []

        for run in runs:
            if sub in meg_bh_fps:
                try:
                    p_mat_meg = meg_bh_fps[sub][int(run)-1]
                    meta_data_meg = generate_meta_data(p_mat_meg)
                    sub_meta.append(meta_data_meg)
                except IndexError:
                    tqdm.write(f"⚠️ MEG: sub {sub}, run {run} 不存在")
            
            if sub in eeg_bh_fps:
                try:
                    p_mat_eeg = eeg_bh_fps[sub][int(run)-1]
                    meta_data_eeg = generate_meta_data(p_mat_eeg)
                    sub_meta.append(meta_data_eeg)
                except IndexError:
                    tqdm.write(f"⚠️ EEG: sub {sub}, run {run} 不存在")
        
        if sub_meta:
            sub_meta = pd.concat(sub_meta, ignore_index=True)
            sub_meta.to_csv(
                op.join(upload_dir, f'sub-{sub}_events.csv'),
                index=False
            )
        else:
            tqdm.write(f"❌ sub {sub} 没有可用数据")

if __name__ == '__main__':
    meg_res, eeg_res = compute_detailed_beh()
    meg_res.to_csv(op.join(res_dir, 'behaviour', 'meg_behaviour.csv'), index=False)
    eeg_res.to_csv(op.join(res_dir, 'behaviour', 'eeg_behaviour.csv'), index=False)
    generate_detailed_events()

#%%

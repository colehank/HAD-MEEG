# %%
import mne
from mne_bids import (
    BIDSPath, 
    read_raw_bids, 
    find_matching_paths,
    write_raw_bids,
    get_bids_path_from_fname
)

import os
import os.path as op
import pandas as pd
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
from mne.io import curry
from orgnize_tri import run_in_one as tri_run_in_one

BIDS_ROOT_RAW = '../../BIN/action/HAD-MEEG-BIDS'
BIDS_ROOT_DST = '../../HAD-MEEG_upload'
DESIGN_EVENT_ID = {"begin": 1, "video on": 2, "video off": 3, "resp": 4, "end": 5}
EEG_RENAME_MAP = {
    'FP1': 'Fp1', 'FP2': 'Fp2', 'FPZ': 'Fpz',
    'FZ': 'Fz', 'FCZ': 'FCz', 'CZ': 'Cz',
    'CPZ': 'CPz', 'PZ': 'Pz', 'POZ': 'POz', 'OZ': 'Oz'
}


deriv_dir = op.join(BIDS_ROOT_DST, 'derivatives')
ev_dir = op.join(deriv_dir, 'detailed_events')
all_evs = sorted(os.listdir(ev_dir))
all_subinfo = pd.read_excel('/nfs/z1/userhome/zzl-zhangguohao/workingdir/BIN/action/action原始编号.xlsx')
all_empty_bids = find_matching_paths(
        BIDS_ROOT_RAW,
        subjects='emptyroom',
        extensions='.ds',
        datatypes='meg'
)
# %%
def gen_meta_info(return_bids=True):
    meta_info = {}
    for ev in tqdm(all_evs):
        sub = ev.split('_')[0][-2:]
        meta_info[sub] = {}
        ev_fp = op.join(ev_dir, ev)
        ev = pd.read_csv(ev_fp)
        sess = ev['session'].unique().tolist()
        for ses in sess:
            extention = 'ds' if ses == 'meg' else 'set'
            runs = ev[ev['session']==ses]['run'].unique().tolist()
            meta_info[sub][ses] = {}
            for run in runs:
                run = f"{run:0>2d}"
                raw_fp = op.join(
                    BIDS_ROOT_RAW, f'sub-{sub}', 
                    f'ses-{ses.lower()}', ses.lower(), 
                    f'sub-{sub}_ses-{ses.lower()}_task-action_run-{run}_{ses.lower()}.{extention}')
                raw_fp = get_bids_path_from_fname(raw_fp) if return_bids else raw_fp
                meta_info[sub][ses][run] = raw_fp
    return meta_info


def correct_event(raw):
    corrected_raw, correcting_info = tri_run_in_one(raw)
    to_des = ("Annotation corrected under experimental design." 
              if correcting_info != "sucess" else None)
    corrected_raw.info['description'] = to_des
    return corrected_raw

def make_dst_bids(raw_bids):
    dst_bids = BIDSPath(
        subject=raw_bids.subject,
        session=raw_bids.session,
        task=raw_bids.task,
        run=raw_bids.run,
        root=BIDS_ROOT_DST,
        extension='.fif' if raw_bids.datatype == 'meg' else '.set',
        datatype= raw_bids.datatype,
    )
    return dst_bids

def fill_missing_date(raw, sub):
    if raw.info.get('meas_date') is not None:
        return raw

    curry.curry.FILE_EXTENSIONS['Curry 8'].update({
        'info': '.cdt.dpo',
        'labels': '.cdt.dpo'
    })
    orig_root = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/BIN/action/EEG/eeg_raw_idAligned'
    orig_fp = f"{orig_root}/sub{sub}.cdt"
    ori_raw = mne.io.read_raw_curry(orig_fp)
    mes_date = ori_raw.info['meas_date']
    raw.set_meas_date(mes_date)
    return raw

def get_sub_info(sub, raw):
    mes_date = raw.info['meas_date']
    sub_id  = f"sub-{sub}"
    sub_row = all_subinfo[all_subinfo['participant_id']==sub_id]
    assert len(sub_row) == 1, f"Subject {sub} length mismatch 1!"
    raw_info = sub_row.to_dict(orient='records')[0]
    to_return = {
        'id': int(raw_info['subject_id']),
        'his_id': str(raw_info['participant_id']),
        'last_name': str(raw_info['subject_name'][0]),
        'first_name': str(raw_info['subject_name'][1:]),
        'birthday': mes_date.date().replace(mes_date.year - int(raw_info['age'])),
        'sex': 1 if raw_info['sex']=='M' else 2,
        'hand': 1 if raw_info['hand']=='R' else 2,
    }
    return to_return

def apply_raw_side(bids):
    corrected_raw = correct_event(str(bids.fpath))
    if bids.datatype == 'eeg':
        corrected_raw = fill_missing_date(corrected_raw, bids.subject)
    corrected_raw.info['subject_info'] = get_sub_info(bids.subject, corrected_raw)
    corrected_raw.info['experimenter'] = 'ghz'
    corrected_raw.info['line_freq'] = 50.0
    return corrected_raw

def find_empty_room_raw(date):
    for empty_bids in all_empty_bids:
        this_date = empty_bids.session
        if this_date == date:
            return read_raw_bids(empty_bids)
    return None


def write_dst_bids(raw, dst_bids):
    ev, evid = mne.events_from_annotations(raw)
    dtype = dst_bids.datatype
    if dtype == 'meg':
        empty_raw = find_empty_room_raw(raw.info['meas_date'].strftime("%Y%m%d"))
        if empty_raw is None:
            tqdm.write(f"⚠️ No matching empty room for date {raw.info['meas_date'].strftime('%Y%m%d')}")

    write_raw_bids(
        raw=raw,
        bids_path=dst_bids,
        events=ev,
        event_id=evid,
        empty_room=empty_raw if dtype == 'meg' else None,
        format='FIF' if dtype == 'meg' else 'EEGLAB',
        verbose=False,
        overwrite=True,
        allow_preload=True
    )

def fit_standard_montage(raw):
    raw.set_channel_types({"HEO": "eog", "VEO": "eog"})
    raw.rename_channels(EEG_RENAME_MAP)
    if 'CB1' in raw.ch_names:
        raw.drop_channels(['CB1', 'CB2'])# not satisfied with the 1020
    if 'EKG' in raw.ch_names:
        raw.set_channel_types({'EKG': 'ecg'})
    if 'EMG' in raw.ch_names:
        raw.set_channel_types({'EMG': 'emg'})
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    return raw

def process_single(sub, ses, run, bids):
    try:
        raw = apply_raw_side(bids)

        if ses == 'eeg':
            raw = fit_standard_montage(raw)

        dst_bids = make_dst_bids(bids)
        write_dst_bids(raw, dst_bids)
        return None
    except Exception as e:
        return (f'{sub}_{ses}_{run}', str(e))

def get_tasks(meta_info):
    tasks = []
    # 先收集所有任务
    for sub in meta_info.keys():
        for ses in meta_info[sub].keys():
            for run_key in meta_info[sub][ses].keys():
                bids = meta_info[sub][ses][run_key]
                tasks.append((sub, ses, run_key, bids))
    return tasks

def run(meta_info, n_jobs=-1):
    # BIDS写入依赖于硬盘上的文件结构，试了一下并行很容易出错，这里建议n_jobs=1串行执行
    tasks = get_tasks(meta_info)
    with tqdm_joblib(total=len(tasks), desc='Organizing EEG/MEG data'):
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single)(sub, ses, run_key, bids)
            for sub, ses, run_key, bids in tasks
        )
    results = [res for res in results if res is not None]
    return results
# %%
if __name__ == '__main__':
    import pickle

    meta_info = gen_meta_info()
    # res = run(meta_info, 1)
    side = []
    tasks = get_tasks(meta_info)
    for task in tqdm(tasks, desc="Organizing EEG/MEG data"):
        res = process_single(*task)
        if res is not None:
            side.append(res)
    with open('orgnize_neu_res.pkl', 'wb') as f:
        pickle.dump(side, f)
    # %%

#%%

# %%
from __future__ import annotations

import json
import shutil
from datetime import datetime

import pandas as pd
from mne_bids import BIDSPath
from mne_bids import find_matching_paths
from mne_bids import get_entity_vals
from mne_bids import write_anat
from tqdm.auto import tqdm

from src.config import DataConfig
fMRI_ROOT = '/nfs/z1/userhome/zzl-zhangguohao/workingdir/BIN/data_upload/HAD'
# %%
config = DataConfig()
fmri_subs = get_entity_vals(fMRI_ROOT, 'subject')
MEEG_ROOT = config.bids_root
# %%
anat_imgs = {
    sub: find_matching_paths(
        fMRI_ROOT, subjects=sub,
        extensions='.nii.gz', datatypes='anat',
    ) for sub in fmri_subs
}

for sub in tqdm(anat_imgs):
    src_bids = anat_imgs[sub][0]
    dst_bids = BIDSPath(
        subject=src_bids.subject,
        session='mri',
        task=None,
        run=None,
        root=MEEG_ROOT,
        extension='.nii.gz',
        datatype='anat',
    )
    sidecar_json = src_bids.copy().update(
        extension='.json',
    ).fpath
    dst_sidecar_json = f'{MEEG_ROOT}/sub-{sub}/ses-mri/anat/sub-{sub}_ses-mri_T1w.json'
    shutil.copy(sidecar_json, dst_sidecar_json)

    # scan_dir = sidecar_json.parent.parent
    # with open(sidecar_json, 'r') as f:
    #     scan_info = json.load(f)
    # acq_time = to_iso8601(scan_info['AcquisitionTime'])
    # scan = [
    #     {
    #     'filename': f"anat/sub-{sub}_ses-mri_T1w.nii.gz",
    #     'acq_time': acq_time
    #     }]
    # scan = pd.DataFrame(scan)
    # scan.to_csv(f"{MEEG_ROOT}/sub-{sub}/ses-mri/sub-{sub}_ses-mri_scans.tsv", sep='\t', index=False)

    write_anat(
        image=src_bids.fpath,
        bids_path=dst_bids,
        overwrite=True,
    )

# %%
sub_evs = {
    sub: config.derivatives_root /
    f'detailed_events/sub-{sub}_events.csv' for sub in config.subjects
}
all_evs = pd.concat([pd.read_csv(fp) for fp in sub_evs.values()], axis=0)
# %%
uni_cls = all_evs['class_id'].unique().tolist()
uni_super_cls = all_evs['super_class_id'].unique().tolist()
dfs = []
for clss in uni_cls:
    df = all_evs[all_evs['class_id'] == clss]
    uni_vid = df['video_id'].unique().tolist()
    print(
        f'Class {clss} has {len(df)} events.'
        f'  covering {len(uni_vid)} videos.',
    )
# %%

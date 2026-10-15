from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from pypdf import PdfReader
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'cet_findings_report'
RUN=ROOT.parent/'common/partial_batch/test_one__0399/cet/20261006T015545971089Z'
rows=json.loads((RUN/'motions.json').read_text());summary=json.loads((RUN/'summary.json').read_text());csv=pd.read_csv(RUN/'motions.csv')
assert len(rows)==len(csv)==91
for r in rows:
 z=np.load(RUN/r['arrays_path'],allow_pickle=False)
 x=csv.loc[csv.original_motion_index.eq(r['original_motion_index'])].iloc[0]
 assert np.isclose(x.cet_degrees,r['cet_degrees'])
 p=z['dtw_frame_pairs'];e=np.abs(z['prediction_curl_degrees'][p[:,0]]-z['GT_curl_degrees'][p[:,1]])
 assert np.allclose(e.mean(0),r['per_joint_degrees'])
 assert np.allclose(e.mean(0).reshape(5,3).mean(1),r['per_finger_degrees'])
 assert np.isclose(e.mean(),r['cet_degrees'])
 for prefix,name in [('GT','GT_curl_degrees'),('prediction','prediction_curl_degrees')]:
  a=z[name];ds={'mean_signed_flexion_degrees':a.mean(0),'mean_absolute_flexion_degrees':np.abs(a).mean(0),'max_absolute_flexion_degrees':np.abs(a).max(0),'flexion_range_degrees':np.ptp(a,axis=0),'total_angular_variation_degrees':np.abs(np.diff(a,axis=0)).sum(0)}
  ds['variation_degrees_per_second']=ds['total_angular_variation_degrees']/r['duration_seconds']
  for k,v in ds.items():
   saved=r[prefix+'_curl_descriptors'][k]
   assert np.allclose(v,saved['per_joint']) and np.allclose(v.reshape(5,3).mean(1),saved['per_finger']) and np.isclose(v.mean(),saved['whole_hand'])
assert set(sorted(rows,key=lambda r:r['cet_degrees'],reverse=True)[i]['original_motion_index'] for i in range(5))=={287,552,696,905,964}
stats=json.loads((OUT/'analysis/statistics.json').read_text())
for f,h in stats['source_hashes'].items():assert hashlib.sha256((RUN/f).read_bytes()).hexdigest()==h
result={'verified_motions':91,'csv_json_CET_match':True,'all_CET_finger_joint_values_reconstructed':True,'all_GT_and_prediction_descriptors_reconstructed':True,'source_results_unchanged':True,'current_evaluator_matches_saved_hash':hashlib.sha256((ROOT.parent/'Scripts/evaluate_cet.py').read_bytes()).hexdigest()==summary['script_sha256']}
pdf=PdfReader(OUT/'deliverables/TaMF_CET_Research_Findings.pdf');result['report_pages']=len(pdf.pages)
assert 8<=len(pdf.pages)<=15
with ZipFile(OUT/'deliverables/TaMF_CET_Findings_Presentation.pptx') as z:result['slides']=sum(n.startswith('ppt/slides/slide') and n.endswith('.xml') for n in z.namelist())
assert result['slides']==11
(OUT/'analysis/evidence_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))

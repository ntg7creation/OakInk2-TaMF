from pathlib import Path
import sys,json
OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(OUT/'build/plot_deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
RUN=OUT.parents[1]/'common/partial_batch/test_one__0399/cet/20261006T015545971089Z'
rows=json.loads((RUN/'motions.json').read_text()); by={r['original_motion_index']:r for r in rows}
s=json.loads((RUN/'summary.json').read_text()); stats=json.loads((OUT/'analysis/statistics.json').read_text())
BLUE='#26617D'; TEAL='#1B827E'; ORANGE='#B96330'; GRAY='#65727B'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#A9B1B7','grid.color':'#E4E8EA','axes.axisbelow':True,'figure.facecolor':'white','savefig.facecolor':'white'})
def save(fig,name):
    fig.savefig(OUT/f'assets/{name}.png',dpi=220,bbox_inches='tight'); fig.savefig(OUT/f'assets/{name}.svg',bbox_inches='tight');plt.close(fig)
def values(key):return np.array([r['GT_curl_descriptors'][key]['whole_hand'] for r in rows])
cet=np.array([r['cet_degrees'] for r in rows]); ids=np.array([r['original_motion_index'] for r in rows])
fig,ax=plt.subplots(2,1,figsize=(10,7),height_ratios=[1,1.2],layout='constrained')
ax[0].hist(cet,bins=np.arange(2,36,2),color=BLUE,edgecolor='white'); ax[0].axvline(np.median(cet),color=ORANGE,ls='--',label=f'Median {np.median(cet):.2f}°');ax[0].set(xlabel='CET (degrees)',ylabel='Motions',title='Finger Curl Error Across Evaluated Motions');ax[0].legend(frameon=False)
ax[1].scatter(ids,cet,c=BLUE,s=25);ax[1].axhline(np.median(cet),color=ORANGE,ls='--');ax[1].set(xlabel='Original motion cache index',ylabel='CET (degrees)',title='All 91 unique motions retain their original IDs')
for i in [287,552,696,905,964]:
    r=by[i];ax[1].annotate(str(i),(i,r['cet_degrees']),xytext=(3,6),textcoords='offset points',fontsize=10)
for a in ax:a.grid(axis='y',alpha=.7)
save(fig,'01_overall')
order=[4,0,1,3,2]; names=['Thumb','Index','Middle','Ring','Pinky']
f=np.array([r['per_finger_degrees'] for r in rows])[:,order]
fig,ax=plt.subplots(1,2,figsize=(11,5.7),width_ratios=[1,1.1],layout='constrained')
ax[0].boxplot(f,tick_labels=names,showfliers=True,patch_artist=True,boxprops={'facecolor':'#DEEAF0','edgecolor':BLUE},medianprops={'color':ORANGE},flierprops={'marker':'.','markersize':3});ax[0].scatter(np.arange(1,6),f.mean(0),marker='D',c=TEAL,s=35,label='Mean',zorder=5);ax[0].set(ylabel='Per-finger CET (degrees)',title='Pinky has the largest mean error');ax[0].legend(frameon=False);ax[0].grid(axis='y')
j=np.array([r['per_joint_degrees'] for r in rows]).mean(0).reshape(5,3)[order]
im=ax[1].imshow(j,cmap='Blues',vmin=0,vmax=16,aspect='auto');ax[1].set_yticks(range(5),names);ax[1].set_xticks(range(3),['Base¹','Middle¹','Distal¹']);ax[1].set_title('Mean CET by anatomical joint (degrees)')
for y in range(5):
 for x in range(3):ax[1].text(x,y,f'{j[y,x]:.2f}',ha='center',va='center',color='white' if j[y,x]>10 else '#152B39')
fig.supxlabel('¹ Thumb: CMC / MCP / IP. Other fingers: MCP / PIP / DIP.',fontsize=10)
save(fig,'02_anatomy')
keys=['mean_absolute_flexion_degrees','flexion_range_degrees','total_angular_variation_degrees','variation_degrees_per_second']
labels=['Mean absolute GT flexion (degrees)','Mean GT joint range (degrees)','Mean GT total angular variation (degrees)','Mean GT variation rate (degrees/second)']
titles=['Curl depth','Excursion during the clip','Accumulated angular activity','Duration-normalized activity']
fig,axes=plt.subplots(2,2,figsize=(10,6.4),layout='constrained')
for a,k,label,title in zip(axes.flat,keys,labels,titles):
    v=values(k);a.hist(v,bins=12,color=TEAL,edgecolor='white');a.axvline(np.median(v),color=ORANGE,ls='--');a.set(title=title,xlabel=label,ylabel='Motions');a.grid(axis='y')
save(fig,'03_gt_descriptors')
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for a,k,label,title in zip(axes.flat,keys,labels,titles):
    v=values(k);a.scatter(v,cet,c=BLUE,s=22,alpha=.8);a.set(title=f"{title}   r = {stats['correlations'][k]['pearson']:.2f}",xlabel=label,ylabel='CET (degrees)');a.grid(alpha=.7)
    for i in [37,389,552,696,934]:
        ix=list(ids).index(i);a.annotate(str(i),(v[ix],cet[ix]),xytext=(3,4),textcoords='offset points',fontsize=8)
save(fig,'04_associations')
def case_axes(axes,mid):
 r=by[mid];z=np.load(RUN/r['arrays_path']);p=z['dtw_frame_pairs'];js=np.argsort(r['per_joint_degrees'])[::-1][:2];x=np.linspace(0,100,len(p))
 for a,j in zip(axes,js):
    a.plot(x,z['GT_curl_degrees'][p[:,1],j],color='#273B46',lw=1.8,label='Ground truth');a.plot(x,z['prediction_curl_degrees'][p[:,0],j],color=TEAL,lw=1.8,label='TaMF output');a.set_title(f"Motion {mid} · {s['joint_names'][j].replace('little','pinky').replace('_',' ')}\nJoint CET {r['per_joint_degrees'][j]:.2f}°",fontsize=12);a.set(xlabel='Shared DTW path progress (%)',ylabel='Signed flexion (degrees)');a.grid(alpha=.7)
for mids,name in [([389,37],'05_active_cases'),([552,696],'06_mismatch_cases'),([934],'07_pour_case')]:
 fig,axes=plt.subplots(len(mids),2,figsize=(10,3.3*len(mids)),squeeze=False,layout='constrained')
 for aa,mid in zip(axes,mids):case_axes(aa,mid)
 axes[0,0].legend(frameon=False,fontsize=9);save(fig,name)
for mid in [389,37,552,696,934]:
 fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained');case_axes(axes,mid);axes[0].legend(frameon=False,fontsize=9);save(fig,f'case_{mid}')
fig,ax=plt.subplots(figsize=(9,4.4),layout='constrained')
depth=np.array([r['GT_curl_descriptors']['mean_absolute_flexion_degrees']['per_finger'][2] for r in rows]);err=np.array([r['per_finger_degrees'][2] for r in rows])
ax.scatter(depth,err,c=BLUE,s=28);xx=np.array([0,80]);ax.plot(xx,.2*xx,color=ORANGE,ls='--',label='Error / GT mean absolute flexion = 20%')
for mid in [0,82,492,934,552,1179]:
 ix=list(ids).index(mid);ax.annotate(str(mid),(depth[ix],err[ix]),xytext=(4,5),textcoords='offset points',fontsize=10)
ax.set(xlabel='Pinky GT mean absolute flexion (degrees)',ylabel='Pinky CET (degrees)',title='Relative magnitude depends on the reference denominator');ax.grid(alpha=.7);ax.legend(frameon=False,fontsize=10)
save(fig,'08_relative')
# Reuse existing notebook evidence; change only the title strip in this new copy.
src=OUT.parent/'exports/cet_graphs_20261006T025302388891Z/019_R_Curl_depth_preservation_R_Curl_activity_preservation.png'
im=Image.open(src).convert('RGB');draw=ImageDraw.Draw(im);w,h=im.size;strip=int(h*.055);draw.rectangle((0,0,w,strip),fill='white');font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',int(h*.029))
for x,t in [(w*.24,'Curl depth agreement'),(w*.74,'Angular activity agreement')]:draw.text((x,strip*.38),t,font=font,fill='black',anchor='mm')
im.save(OUT/'assets/09_preservation_reused.png')
print('Created 9 report figures and 5 case detail assets.')

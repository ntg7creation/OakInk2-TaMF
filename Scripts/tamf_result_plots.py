"""Render validated saved-data analysis as portable PNG figures."""
from __future__ import annotations
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np

def render(state, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.dpi': 120, 'axes.titlesize': 12, 'axes.labelsize': 11, 'legend.fontsize': 9})
    COLORS = {'gt': '#2878B5', 'refined': '#E58632', 'neutral': '#667085'}
    POP_LABELS = ['GT', 'Refined']
    POP_COLORS = [COLORS['gt'], COLORS['refined']]
    NAMES = {'cr': 'CR', 'siv': 'SIV', 'psklj': 'PSKL-J', 'fid': 'FID'}
    METRICS = ('cr', 'siv', 'psklj', 'fid')
    arrays = {m: [np.asarray(a) for a in pair] for m, pair in state['arrays'].items()}
    motion_values = {m: np.asarray(v) for m, v in state['motion_values'].items()}
    motion_rows = state['motion_rows']
    scores, metadata_ok = state['scores'], state['metadata_ok']
    generated, paired = state['generated'], state['paired']
    mapping_reasons, limitations = state['mapping_reasons'], state['limitations']
    fid, fid_terms, fid_formula_ok = state['fid'], state['fid_terms'], state['fid_formula_ok']
    mean_term, covariance_term = state['mean_term'], state['covariance_term']
    DISPLAY_TIMEZONE = state['display_timezone']
    runtime_items = [(m, t, datetime.fromisoformat(start) if isinstance(start, str) else start,
                      datetime.fromisoformat(end) if isinstance(end, str) else end)
                     for m, t, start, end in state['runtime_items']]
    output_dir = Path(output_dir)
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)
    figures = []
    section = ''
    def skip(what, reason):
        limitations.append(f'{what}: {reason}')
    def ready(metric, paired=False):
        if metric not in arrays:
            skip(NAMES[metric], 'saved arrays failed provenance checks; see the export audit')
            return False
        if paired and metric not in motion_values:
            skip(f'{NAMES[metric]} paired/per-motion plots', '; '.join(mapping_reasons) or 'boundaries did not reproduce saved scores')
            return False
        return True
    def grid(ax):
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
    def show_plot(fig):
        rel = f'plots/figure_{len(figures) + 1:02d}.png'
        fig.savefig(output_dir / rel, bbox_inches='tight', facecolor='white')
        figures.append({'section': section, 'path': rel,
                        'title': ' / '.join(ax.get_title() for ax in fig.axes if ax.get_title())})
        plt.close(fig)
    def note(reading, observation):
        figures[-1].update(how_to_read=reading, observation=observation)
    indices = np.array([r['index'] for r in motion_rows]) if np is not None else []
    def index_axis(ax):
        # Every marker sits at the actual original index; sparse ticks use actual keys.
        positions = np.unique(np.rint(np.linspace(0, len(indices)-1, min(13, len(indices)))).astype(int))
        ax.set_xticks(indices[positions])
        ax.set_xlim(float(indices.min()) - 15, float(indices.max()) + 15)
        ax.set_xlabel('Original generated motion / cache index (first occurrence)')
        grid(ax)
    def label_extremes(ax, values, count=4):
        top = np.argsort(np.abs(values))[-count:]
        for j, k in enumerate(top):
            ax.annotate(str(indices[k]), (indices[k], values[k]), xytext=(5, 12 if j % 2 == 0 else -20),
                        textcoords='offset points', fontsize=9,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.85),
                        arrowprops=dict(arrowstyle='-', color=COLORS['neutral'], lw=.7))

    section = 'Motion-level failures'
    if ready('cr', paired=True):
        fig, ax = plt.subplots(figsize=(13.5, 4.4))
        for vals, label, color in zip(motion_values['cr'], POP_LABELS, POP_COLORS):
            ax.plot(indices, vals * 100, marker='o', ms=4, lw=.8, alpha=.85, label=label, color=color)
        ax.set(title='CR per unique motion · valid frames at distance < 5 mm', ylabel='Contact frames (%)', ylim=(-3, 103))
        index_axis(ax); ax.legend(loc='lower left')
        show_plot(fig)
        low = np.argmin(motion_values['cr'][1])
        note('Compare the two markers at each original index. Each ratio uses that motion’s valid frames; the official pooled score weights motions by frame count.',
             f'Lowest refined CR: index **{indices[low]}**, **{motion_values["cr"][1, low]:.2%}** (GT **{motion_values["cr"][0, low]:.2%}**). '
             f'Official pooled GT/refined: **{scores["cr"][0]:.2%} / {scores["cr"][1]:.2%}**.')
    if ready('siv', paired=True):
        fig, ax = plt.subplots(figsize=(13.5, 4.4))
        for vals, label, color in zip(motion_values['siv'], POP_LABELS, POP_COLORS):
            ax.plot(indices, vals, marker='o', ms=4, lw=.8, alpha=.85, label=label, color=color)
        ax.set(title='Mean SIV per unique motion · offsets 0, 20, 40, … within valid length', ylabel='Mean intersection volume (cm³)')
        ax.set_ylim(bottom=-.5); index_axis(ax); ax.legend()
        show_plot(fig)
        high = np.argmax(motion_values['siv'][1])
        note('Higher values indicate more sampled mesh interpenetration. Zero is visible; each motion mean uses only its official stride-20 samples.',
             f'Largest refined mean: index **{indices[high]}**, **{motion_values["siv"][1, high]:.3f} cm³** '
             f'(GT **{motion_values["siv"][0, high]:.3f} cm³**, {motion_rows[high]["siv_samples"]} samples). '
             f'The frame-weighted pooled means are **{scores["siv"][0]:.3f} / {scores["siv"][1]:.3f} cm³**.')
    if all(ready(m, paired=True) for m in ('cr', 'siv')):
        fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.3), sharex=True)
        observations = []
        for ax, m, scale, unit in zip(axes, ('cr', 'siv'), (100, 1), ('percentage points', 'cm³')):
            delta = (motion_values[m][1] - motion_values[m][0]) * scale
            ax.axhline(0, color=COLORS['neutral'], lw=1)
            ax.plot(indices, delta, 'o-', color=COLORS['refined'], ms=4, lw=.8)
            ax.set(title=f'{NAMES[m]} per-motion difference · refined − GT', ylabel=f'Difference ({unit})')
            index_axis(ax); label_extremes(ax, delta)
            ax.margins(y=.25)
            top = np.argsort(np.abs(delta))[-3:][::-1]
            observations.append(f'{NAMES[m]} largest absolute changes: ' + ', '.join(f'index **{indices[k]}** ({delta[k]:+.3f} {unit})' for k in top) + '.')
        fig.tight_layout(); show_plot(fig)
        gain = np.argmax(motion_values['cr'][1] - motion_values['cr'][0])
        coupling = (f' At index **{indices[gain]}**, the largest contact gain '
                    f'(**{100 * (motion_values["cr"][1, gain] - motion_values["cr"][0, gain]):+.2f} percentage points**) '
                    f'accompanies a penetration change of **{motion_values["siv"][1, gain] - motion_values["siv"][0, gain]:+.3f} cm³** '
                    f'over {motion_rows[gain]["siv_samples"]} SIV samples. More contact can coexist with severe overlap.')
        note('The zero line means no change. Labels identify the largest absolute changes by original index. Negative CR means less contact; positive SIV means more overlap. These are motion summaries, not new official population scores.',
             ' '.join(observations) + coupling)

    section = 'Coverage'
    if metadata_ok and plt is not None:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
        counts = [generated, paired]
        b = axes[0].bar(['Generated\nsegments', 'Evaluated unique\nmotions'], counts, color=[COLORS['neutral'], '#6B9D79'])
        axes[0].bar_label(b, padding=4); axes[0].set(title='Segments versus deduplicated motions', ylabel='Count', ylim=(0, generated * 1.15))
        sides = Counter(r['hand'] for r in motion_rows)
        b = axes[1].bar(['Left', 'Right'], [sides['lh'], sides['rh']], color=COLORS['neutral'])
        axes[1].bar_label(b, padding=4); axes[1].set(title='Hand coverage · unique motions', ylabel='Unique motions', ylim=(0, max(sides.values()) * 1.2))
        lens = np.array([r['valid_frames'] for r in motion_rows])
        axes[2].hist(lens, bins=np.arange(0, 177, 16), color=COLORS['neutral'], edgecolor='white')
        axes[2].axvline(np.median(lens), ls='--', color='black', label=f'Median {np.median(lens):g}')
        axes[2].set(title='Valid length · first-occurrence motions', xlabel='Valid frames (not padded length)', ylabel='Unique motions', xlim=(0, 170)); axes[2].legend()
        for ax in axes: grid(ax)
        fig.tight_layout(); show_plot(fig)
        note('The left panel counts inputs and deduplicated evaluation units separately. The middle and right panels count each unique motion once.',
             f'**{generated - paired:,} duplicate segments** are excluded; {sides["lh"]} left and {sides["rh"]} right motions remain. '
             f'Valid lengths span **{lens.min()}–{lens.max()} frames**, median **{np.median(lens):g}**.')
    else:
        skip('Coverage distributions', '; '.join(mapping_reasons) or 'plotting unavailable')
    if metadata_ok and plt is not None:
        primitive_counts = Counter(r['primitive'].split(':')[0] for r in motion_rows)
        labels, values = zip(*sorted(primitive_counts.items(), key=lambda x: (-x[1], x[0])))
        fig, ax = plt.subplots(figsize=(10, 6.6))
        b = ax.barh(labels, values, color=COLORS['neutral']); ax.bar_label(b, padding=4)
        ax.invert_yaxis(); ax.set(title='Primitive coverage · unique evaluated motions', xlabel='Unique motions', xlim=(0, max(values) * 1.18)); grid(ax)
        show_plot(fig)
        note('Each bar counts deduplicated sequence / primitive / hand keys in an action family; repeated generated segments contribute no additional count.',
             f'**{len(labels)} primitive families** are represented. **{labels[0]}** has the most unique motions ({values[0]}); '
             f'**{sum(v == 1 for v in values)} families** have just one motion, limiting comparisons between primitives.')
    if metadata_ok and plt is not None:
        object_counts = Counter(oid for row in motion_rows for oid in row['objects'])
        object_ids, counts = zip(*sorted(object_counts.items(), key=lambda x: (-x[1], x[0])))
        fig, ax = plt.subplots(figsize=(11, 8))
        b = ax.barh(object_ids, counts, color=COLORS['neutral']); ax.bar_label(b, padding=3)
        ax.invert_yaxis(); ax.set(title='Object coverage · unique motions containing each object', xlabel='Unique motions containing object', xlim=(0, max(counts) * 1.14)); grid(ax)
        show_plot(fig)
        note('Objects come from the validated cached trajectories and matching refined obj_list. A motion may contain multiple objects, so bars need not sum to the motion count.',
             f'**{len(object_ids)} object identities** appear in {len(motion_rows)} evaluated motions. '
             f'**{object_ids[0]}** is most frequent ({counts[0]} motions); **{sum(c == 1 for c in counts)} objects** appear in only one motion.')

    section = 'CR contact distance'
    if ready('cr'):
        cr_mm = [a * 1000 for a in arrays['cr']]
        full = max(a.max() for a in cr_mm)
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
        for a, label, color in zip(cr_mm, POP_LABELS, POP_COLORS):
            axes[0].hist(a, bins=np.linspace(0, full, 65), weights=np.full(len(a), 100 / len(a)), histtype='step', lw=1.8, label=label, color=color)
            axes[1].hist(a, bins=np.linspace(0, 10, 61), weights=np.full(len(a), 100 / len(a)), histtype='step', lw=1.8, label=label, color=color)
        axes[0].set(title='CR distance histogram · full range', xlabel='Minimum distance (mm)', ylabel='All valid frames per bin (%)', xlim=(0, full * 1.02), yscale='log')
        axes[1].set(title='CR near-contact distances · 0–10 mm', xlabel='Minimum distance (mm)', ylabel='All valid frames per bin (%)', xlim=(0, 10))
        for ax in axes[:2]:
            ax.axvline(5, ls='--', color=COLORS['neutral'], label='Official threshold: 5 mm'); ax.legend()
        bp = axes[2].boxplot(cr_mm, patch_artist=True, showfliers=True, medianprops=dict(color='black'))
        axes[2].set_xticks([1, 2], POP_LABELS)
        for patch, color in zip(bp['boxes'], POP_COLORS): patch.set_facecolor(color); patch.set_alpha(.65)
        axes[2].axhline(5, ls='--', color=COLORS['neutral'], label='5 mm')
        axes[2].set(title='CR median, spread and outliers', ylabel='Minimum distance (mm)', yscale='symlog', ylim=(-.1, full * 1.08)); axes[2].legend()
        for ax in axes: grid(ax)
        fig.tight_layout(); show_plot(fig)
        note('Histograms overlay both populations. Full-range log height reveals rare tail bins; the close-up uses all frames as denominator and omits distances >10 mm. Boxes show median, IQR, 1.5×IQR whiskers and outliers; the distance box axis is symmetric-log (linear below 2 mm), retaining zeros.',
             f'GT/refined medians: **{np.median(cr_mm[0]):.3f} / {np.median(cr_mm[1]):.3f} mm**; '
             f'maxima: **{cr_mm[0].max():.2f} / {cr_mm[1].max():.2f} mm**. '
             f'Above 10 mm: **{(cr_mm[0] > 10).mean():.2%} / {(cr_mm[1] > 10).mean():.2%}** of frames.')
    if ready('cr'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for a, label, color in zip(cr_mm, POP_LABELS, POP_COLORS):
            x = np.sort(a); y = np.arange(1, len(x) + 1) / len(x) * 100
            for ax in axes: ax.step(x, y, where='post', color=color, label=label, lw=1.8)
        axes[0].set(title='CR cumulative distances · full range', xlim=(0, max(a.max() for a in cr_mm) * 1.02), ylim=(0, 101))
        axes[1].set(title='CR cumulative distances · around 5 mm', xlim=(3, 7), ylim=(85, 100.5))
        for ax in axes:
            ax.axvline(5, ls='--', color=COLORS['neutral'], label='Official threshold: 5 mm')
            ax.set(xlabel='Minimum distance (mm)', ylabel='Frames at or below distance (%)'); ax.legend(); grid(ax)
        fig.tight_layout(); show_plot(fig)
        note('A higher curve has more frames at or below each distance. The close-up magnifies the threshold neighborhood. CDF uses ≤; official CR uses strict <. Full range is shown at left.',
             f'Official CR: **{scores["cr"][0]:.2%} GT / {scores["cr"][1]:.2%} refined** '
             f'(**{100 * (scores["cr"][1] - scores["cr"][0]):+.3f} percentage points**). '
             f'Frames exactly at 5 mm: GT **{np.sum(cr_mm[0] == 5)}**, refined **{np.sum(cr_mm[1] == 5)}**.')
    if ready('cr'):
        thresholds = np.linspace(0, 20, 401)
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for a, label, color in zip(cr_mm, POP_LABELS, POP_COLORS):
            rates = np.searchsorted(np.sort(a), thresholds, side='left') / len(a) * 100
            ax.plot(thresholds, rates, color=color, lw=2, label=label)
        ax.axvline(5, ls='--', color=COLORS['neutral'], label='Official CR remains fixed at 5 mm')
        ax.set(title='Exploratory sensitivity · contact rate versus distance threshold', xlabel='Exploratory distance threshold (mm)', ylabel='Frames below threshold (%)', xlim=(0, 20), ylim=(0, 101)); ax.legend(); grid(ax)
        show_plot(fig)
        vals = [[(a < t).mean() for a in cr_mm] for t in (2, 10)]
        note('Each point applies a hypothetical strict distance threshold to the same saved frames. This exploratory sensitivity plot does not change the official 5 mm CR.',
             f'At **2 mm**, GT/refined rates are **{vals[0][0]:.2%} / {vals[0][1]:.2%}**; '
             f'at **10 mm**, **{vals[1][0]:.2%} / {vals[1][1]:.2%}**. Threshold choice changes the apparent comparison.')
    if ready('cr', paired=True):
        gt, ref = cr_mm; delta = ref - gt
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
        top = max(gt.max(), ref.max()) * 1.04
        axes[0].scatter(gt, ref, s=8, alpha=.22, color=COLORS['refined'], rasterized=True, label='Paired valid frames')
        axes[0].plot([0, top], [0, top], color=COLORS['neutral'], label='Equality')
        axes[0].axvline(5, ls='--', color=COLORS['gt']); axes[0].axhline(5, ls='--', color=COLORS['refined'], label='5 mm boundaries')
        axes[0].set(title='CR paired frame distances', xlabel='GT distance (mm)', ylabel='Refined distance (mm)', xscale='symlog', yscale='symlog', xlim=(0, top), ylim=(0, top)); axes[0].legend(fontsize=8)
        axes[1].hist(delta, bins=70, color=COLORS['refined'], alpha=.8); axes[1].axvline(0, color=COLORS['neutral'], ls='--')
        axes[1].set(title='CR paired distance changes', xlabel='Refined − GT distance (mm)', ylabel='Valid frames', yscale='log')
        # Rows GT state, columns refined state; counts and % of ALL paired frames.
        gc, rc = arrays['cr'][0] < .005, arrays['cr'][1] < .005
        transition = np.array([[np.sum((gc == g) & (rc == r)) for r in (True, False)] for g in (True, False)])
        axes[2].imshow(transition / len(gc) * 100, cmap='Blues', vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                value = transition[i, j]
                axes[2].text(j, i, f'{value:,}\n{value / len(gc):.2%}', ha='center', va='center', color='white' if value / len(gc) > .5 else 'black', fontsize=11)
        axes[2].set(title='Contact transitions · threshold <5 mm', xticks=[0, 1], yticks=[0, 1], xticklabels=['Contact', 'Noncontact'], yticklabels=['Contact', 'Noncontact'], xlabel='Refined state', ylabel='GT state')
        for ax in axes[:2]: grid(ax)
        fig.tight_layout(); show_plot(fig)
        note('Scatter below equality means closer after refinement; symmetric-log axes retain zero and the full tail. Negative histogram values mean shorter distances. Matrix rows are GT and columns are refined, with percentages of all paired frames. Pairing follows verified motion slices and identical valid frame lists.',
             f'**{transition[0, 1]:,} frames** lose contact; **{transition[1, 0]:,}** gain it. '
             f'Median paired distance change: **{np.median(delta):+.4f} mm**; mean **{delta.mean():+.4f} mm**. '
             'The large GT tail can lower mean distance after refinement even when official CR falls.')

    section = 'SIV penetration volume'
    if ready('siv'):
        siv = arrays['siv']; top = max(a.max() for a in siv)
        typical = float(np.quantile(np.concatenate(siv), .95))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        for a, label, color in zip(siv, POP_LABELS, POP_COLORS):
            for ax, bins in zip(axes, (np.linspace(0, top, 55), np.linspace(0, typical, 40))):
                ax.hist(a, bins=bins, weights=np.full(len(a), 100 / len(a)), histtype='step', lw=2, color=color, label=label)
        axes[0].set(title='SIV distribution · full range including zeros', xlim=(0, top * 1.02), yscale='symlog')
        axes[1].set(title=f'SIV typical-value close-up · 0–{typical:.2f} cm³', xlim=(0, typical))
        for ax in axes: ax.set(xlabel='Intersection volume (cm³)', ylabel='All sampled frames per bin (%)'); ax.legend(); grid(ax)
        fig.tight_layout(); show_plot(fig)
        note('The full-range histogram uses symmetric-log height to reveal rare tail values. The close-up ends at the combined-population 95th percentile; bin percentages retain all samples as denominator, including those outside the view.',
             f'Maximum GT/refined penetration: **{siv[0].max():.2f} / {siv[1].max():.2f} cm³**. '
             f'Above the close-up limit: **{(siv[0] > typical).mean():.2%} / {(siv[1] > typical).mean():.2%}** of GT/refined samples.')
    if ready('siv'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        bp = axes[0].boxplot(siv, patch_artist=True, showfliers=True, medianprops=dict(color='black'))
        axes[0].set_xticks([1, 2], POP_LABELS)
        for patch, color in zip(bp['boxes'], POP_COLORS): patch.set_facecolor(color); patch.set_alpha(.65)
        # Explicit zero points: exact zeros otherwise sit on a whisker.
        for k, a in enumerate(siv, 1):
            if np.any(a == 0): axes[0].scatter(k, 0, marker='D', s=45, color=POP_COLORS[k-1], edgecolor='black', zorder=4)
        axes[0].set(title='SIV spread · zeros and all outliers visible', ylabel='Intersection volume (cm³)', yscale='symlog', ylim=(-.2, max(a.max() for a in siv) * 1.12))
        zero_pct = np.array([(a == 0).mean() * 100 for a in siv])
        b = axes[1].bar(POP_LABELS, zero_pct, color=POP_COLORS, width=.55)
        axes[1].bar_label(b, labels=[f'{x:.2f}%' for x in zero_pct], padding=5)
        axes[1].set(title='Zero sampled penetration', ylabel='Stride-20 samples with SIV = 0 (%)', ylim=(0, max(zero_pct) * 1.3))
        for ax in axes: grid(ax)
        fig.tight_layout(); show_plot(fig)
        note('Boxes show median/IQR, 1.5×IQR whiskers and every outlier on a symmetric-log volume axis (linear below 2 cm³). Diamonds mark exact zeros. Bars show the fraction of sampled frames with exactly zero overlap, not collision-free motion rates.',
             f'Exact-zero samples: GT **{np.sum(siv[0] == 0)} / {len(siv[0])} ({zero_pct[0]:.2f}%)**, refined '
             f'**{np.sum(siv[1] == 0)} / {len(siv[1])} ({zero_pct[1]:.2f}%)**. Refined median is **{np.median(siv[1]):.3f} cm³**, GT **{np.median(siv[0]):.3f} cm³**.')
    if ready('siv'):
        fig, ax = plt.subplots(figsize=(10, 4.2))
        for a, label, color in zip(siv, POP_LABELS, POP_COLORS):
            x = np.sort(a); y = np.arange(1, len(a) + 1) / len(a) * 100
            ax.step(x, y, where='post', label=label, color=color, lw=2)
            ax.scatter(0, (a == 0).mean() * 100, color=color, s=35, zorder=3)
        ax.set(title='SIV cumulative distribution · full range', xlabel='Intersection volume (cm³), symmetric-log axis', ylabel='Samples at or below volume (%)', xscale='symlog', xlim=(0, max(a.max() for a in siv) * 1.04), ylim=(0, 101)); ax.legend(); grid(ax)
        show_plot(fig)
        note('Higher curves mean more samples have low penetration. Dots at zero show the initial mass; a symmetric-log horizontal axis keeps both zero and the full outlier range visible.',
             f'At **1 cm³**, **{(siv[0] <= 1).mean():.2%}** of GT and **{(siv[1] <= 1).mean():.2%}** of refined samples are at or below the limit.')
    if ready('siv'):
        quantiles = [.5, .9, .95, .99]
        qvalues = np.array([np.quantile(a, quantiles) for a in siv])
        x = np.arange(4); fig, ax = plt.subplots(figsize=(10, 4.2))
        for k, label, color in zip(range(2), POP_LABELS, POP_COLORS):
            ax.plot(x, qvalues[k], 'o-', color=color, label=label, lw=1.5)
            for j, y in enumerate(qvalues[k]):
                offset = ((-12, 10) if k == 0 else (12, 25)) if j == 0 else (0, 7 if k else -15)
                ax.annotate(f'{y:.2f}', (j, y), xytext=offset, textcoords='offset points', ha='center', fontsize=9,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.7))
        ax.set(title='SIV median and upper-tail percentiles', xticks=x, xticklabels=['Median', '90th', '95th', '99th'], ylabel='Intersection volume (cm³)', ylim=(-1, qvalues.max() * 1.2)); ax.legend(); grid(ax)
        show_plot(fig)
        note('Markers compare the median and tail cutoffs over pooled sampled frames. Lines guide the eye between discrete percentile levels; percentiles are not confidence intervals.',
             f'GT/refined 95th percentiles: **{qvalues[0, 2]:.3f} / {qvalues[1, 2]:.3f} cm³**; '
             f'99th: **{qvalues[0, 3]:.3f} / {qvalues[1, 3]:.3f} cm³**. The median and upper tail both increase after refinement.')
    if ready('siv', paired=True):
        gt, ref = siv; delta = ref - gt; top = max(gt.max(), ref.max()) * 1.04
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].scatter(gt, ref, color=COLORS['refined'], s=15, alpha=.45, label='Paired stride-20 samples')
        axes[0].plot([0, top], [0, top], color=COLORS['neutral'], label='Equality')
        axes[0].set(title='SIV paired sampled volumes', xlabel='GT intersection volume (cm³)', ylabel='Refined intersection volume (cm³)', xscale='symlog', yscale='symlog', xlim=(0, top), ylim=(0, top)); axes[0].legend()
        axes[1].hist(delta, bins=55, color=COLORS['refined'], alpha=.8)
        axes[1].axvline(0, color=COLORS['neutral'], ls='--')
        axes[1].set(title='SIV paired sample changes', xlabel='Refined − GT volume (cm³)', ylabel='Sampled frames', yscale='log')
        for ax in axes: grid(ax)
        fig.tight_layout(); show_plot(fig)
        note('Above equality indicates greater penetration after refinement. Symmetric-log scatter axes retain zeros and outliers. Positive histogram values indicate increased overlap; all pairs use the same validated motion and frame offset.',
             f'Penetration increases in **{(delta > 0).mean():.2%}** of samples, decreases in **{(delta < 0).mean():.2%}**, '
             f'and is unchanged in **{(delta == 0).mean():.2%}**. Mean change **{delta.mean():+.3f} cm³**, median **{np.median(delta):+.3f} cm³**.')

    section = 'PSKL-J population scores'
    if plt is not None and all(v is not None for v in scores['psklj']):
        fig, ax = plt.subplots(figsize=(8, 3.8))
        vals = scores['psklj']
        b = ax.bar(['GT → Refined', 'Refined → GT'], vals, color=POP_COLORS, width=.55)
        ax.bar_label(b, labels=[f'{v:.6f}' for v in vals], padding=5)
        ax.set(title='PSKL-J · official directional population scores', ylabel='Divergence (unitless)', ylim=(0, max(vals) * 1.25)); grid(ax)
        show_plot(fig)
        note('The two bars are different KL directions between the pooled GT and refined acceleration power spectra. Lower means more similar spectra; neither bar is a score for an individual motion.',
             f'GT → refined **{vals[0]:.6f}**; refined → GT **{vals[1]:.6f}**. Directional difference **{vals[1] - vals[0]:+.6f}**.')
    else:
        skip('PSKL-J directional bars', 'successful saved scores or plotting dependencies unavailable')



    section = 'FID population diagnostics'
    if fid_formula_ok and plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        vals = [mean_term, covariance_term]
        b = ax.bar(['Squared mean-distance term', 'Combined covariance term'], vals, color=[COLORS['neutral'], COLORS['refined']], width=.6)
        ax.bar_label(b, labels=[f'{v:.6f}' for v in vals], padding=5)
        ax.set(title='FID diagnostic contributions · cancellation verified', ylabel='Official encoder feature-space contribution', ylim=(0, max(vals) * 1.25)); grid(ax)
        show_plot(fig)
        note('The first term is ||μGT − μrefined||². The second combines covariance traces as tr(ΣGT) + tr(Σrefined) − 2tr(sqrt(ΣGTΣrefined)); the large trace terms cancel, so they are not separate positive FID contributions.',
             f'Mean term **{mean_term:.6f}** + covariance term **{covariance_term:.6f}** = **{mean_term + covariance_term:.6f}**, '
             f'agreeing with saved FID **{fid:.6f}** within **{abs(mean_term + covariance_term - fid):.2g}**. '
             f'Covariance traces sum to **{fid_terms["tr1"] + fid_terms["tr2"]:.6f}**, minus **{2 * fid_terms["tr_covmean"]:.6f}**.')
    else:
        skip('FID diagnostic plot', 'terms missing or official formula/log cancellation did not verify')


    section = 'Runtime'
    if plt is not None and runtime_items:
        import matplotlib.dates as mdates
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.1), gridspec_kw={'width_ratios': [1, 1.5]})
        labels = [NAMES[m] for m, *_ in runtime_items]; times = [t for _, t, *_ in runtime_items]
        b = axes[0].barh(labels, times, color=COLORS['neutral']); axes[0].bar_label(b, labels=[f'{t:.1f} s' for t in times], padding=4)
        axes[0].invert_yaxis(); axes[0].set(title='Metric runtime comparison', xlabel='Recorded wall-clock seconds', xlim=(0, max(times) * 1.25))
        for i, (m, t, start, end) in enumerate(runtime_items):
            axes[1].barh(i, mdates.date2num(end) - mdates.date2num(start), left=mdates.date2num(start), color=COLORS['neutral'], height=.55)
        axes[1].set_yticks(range(len(labels)), labels); axes[1].invert_yaxis()
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S', tz=ZoneInfo(DISPLAY_TIMEZONE)))
        axes[1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        axes[1].set(title=f'Run timeline · {runtime_items[0][2].astimezone(ZoneInfo(DISPLAY_TIMEZONE)):%Y-%m-%d}', xlabel=f'Saved start/end times ({DISPLAY_TIMEZONE})')
        axes[1].tick_params(axis='x', rotation=20)
        for ax in axes: grid(ax)
        fig.tight_layout(); show_plot(fig)
        span = (max(x[3] for x in runtime_items) - min(x[2] for x in runtime_items)).total_seconds()
        longest = max(runtime_items, key=lambda x: x[1])
        note('Left compares the recorded metric durations. Right places each subprocess on its saved start/end timeline, shown in the user’s time zone. Small gaps are runner overhead, not evaluation work.',
             f'**{NAMES[longest[0]]}** is longest at **{longest[1]:.1f} s**, **{longest[1] / sum(times):.1%}** of metric runtime. '
             f'The first-start to last-end span is **{span:.1f} s**; summed metric runtimes **{sum(times):.1f} s**.')
    else:
        skip('Runtime/timeline', 'no valid saved start/end timestamps or plotting unavailable')
    return figures

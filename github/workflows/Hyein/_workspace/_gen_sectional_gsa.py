# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = r'D:\pykan\github\workflows\Hyein\analytical_results'
FIGDIR = r'D:\pykan\github\workflows\Hyein\figures_for_paper'
WS = r'D:\pykan\github\workflows\Hyein\_workspace'

DATASETS = ['exponential', 'logarithm', 'log2', 'rosenbrock']

SA_RC = {
    'figure.dpi': 150, 'figure.facecolor': 'white', 'figure.autolayout': True,
    'axes.facecolor': 'white', 'axes.edgecolor': '#444444', 'axes.linewidth': 0.8,
    'axes.spines.top': True, 'axes.spines.right': True,
    'axes.labelsize': 11, 'axes.labelcolor': 'black', 'axes.grid': False,
    'xtick.labelsize': 9, 'xtick.color': 'black', 'xtick.direction': 'out',
    'ytick.labelsize': 9, 'ytick.color': 'black', 'ytick.direction': 'out',
    'font.family': 'sans-serif', 'font.size': 9, 'font.weight': '300',
    'axes.labelweight': '500', 'text.color': 'black',
    'legend.fontsize': 7, 'legend.framealpha': 0.0,
    'lines.linewidth': 1.2, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
}

TOP_COLOR = '#1f77b4'   # blue, top feature
SECOND_COLOR = '#d62728'  # red, second feature
KAN_COLOR = 'green'
AGSM_COLOR = 'orange'
ANALYTIC_COLOR = 'black'

# Analytical true transition for exponential:
# f(x0,x1)=exp(-2 x0)+x1 -> |df/dx0|=|df/dx1| -> 2 exp(-2 x0)=1 -> x0=ln(2)/2
EXP_TRUE = np.log(2.0) / 2.0  # ~ +0.34657


def denorm_points(norm_vals, scaler, feat_idx):
    """Inverse-transform normalized scalar values (feature feat_idx) to raw space."""
    out = []
    for v in norm_vals:
        row = np.zeros((1, scaler.n_features_in_))
        # MinMaxScaler.inverse_transform needs all columns; fill others with mid value
        row[0, feat_idx] = v
        raw = scaler.inverse_transform(row)[0, feat_idx]
        out.append(float(raw))
    return out


def compute_crossings(centers, s_top, s_second):
    """AGSM transition = sign change of (s_top - s_second), linear interpolated."""
    diff = np.asarray(s_top) - np.asarray(s_second)
    cross = []
    for i in range(len(diff) - 1):
        if diff[i] == 0:
            cross.append(float(centers[i]))
            continue
        if diff[i] * diff[i + 1] < 0:
            x = centers[i] + (centers[i + 1] - centers[i]) * abs(diff[i]) / (abs(diff[i]) + abs(diff[i + 1]))
            cross.append(float(x))
    return cross


def load_all():
    data = {}
    for name in DATASETS:
        csv = os.path.join(BASE, name, 'kan_models', name + '_agsm_sectional.csv')
        pkl = os.path.join(BASE, name, 'kan_models', name + '_range_split_data.pkl')
        df = pd.read_csv(csv)
        d = joblib.load(pkl)
        scaler = d['scaler_X']
        feats = list(df['Feature'].unique())
        top_feat, second_feat = feats[0], feats[1]
        top_idx = int(df[df['Feature'] == top_feat]['Feature_idx'].iloc[0])
        second_idx = int(df[df['Feature'] == second_feat]['Feature_idx'].iloc[0])

        d_top = df[df['Feature'] == top_feat].sort_values('Section_center')
        d_sec = df[df['Feature'] == second_feat].sort_values('Section_center')
        centers = d_top['Section_center'].values
        s_top = d_top['S_a'].values
        s_sec = d_sec['Section_center'].values  # placeholder
        s_sec = d_sec['S_a'].values

        # AGSM transitions (raw space = Section_center is already raw space)
        agsm_tps = compute_crossings(centers, s_top, s_sec)

        # KAN inflection points per input (normalized) -> denorm raw
        ip_norm = d['inflection_points_per_input']
        ip_top_norm = ip_norm[top_idx] if top_idx < len(ip_norm) else []
        ip_top_raw = denorm_points(ip_top_norm, scaler, top_idx)

        # domain width of top feature (raw)
        lo = float(scaler.data_min_[top_idx])
        hi = float(scaler.data_max_[top_idx])
        width = hi - lo

        data[name] = dict(
            df=df, scaler=scaler, top_feat=top_feat, second_feat=second_feat,
            top_idx=top_idx, second_idx=second_idx,
            centers=centers, s_top=s_top, s_sec=s_sec,
            agsm_tps=agsm_tps, ip_top_raw=ip_top_raw,
            lo=lo, hi=hi, width=width,
        )
    return data


def build_metrics(data):
    rows = []
    for name in DATASETS:
        D = data[name]
        ips = D['ip_top_raw']
        tps = D['agsm_tps']
        width = D['width']
        if len(ips) == 0:
            # no KAN inflection point on top feature
            nearest_tp = tps[0] if len(tps) else np.nan
            rows.append(dict(
                dataset=name, feature=D['top_feat'],
                kan_inflection_raw=np.nan,
                agsm_transition_raw=nearest_tp,
                abs_diff=np.nan,
                domain_width=width,
                relative_error=np.nan,
                kan_in_agsm_section=np.nan,
            ))
            continue
        for ip in ips:
            if len(tps) == 0:
                rows.append(dict(
                    dataset=name, feature=D['top_feat'],
                    kan_inflection_raw=ip,
                    agsm_transition_raw=np.nan,
                    abs_diff=np.nan,
                    domain_width=width,
                    relative_error=np.nan,
                    kan_in_agsm_section=np.nan,
                ))
                continue
            # nearest AGSM transition
            tps_arr = np.asarray(tps)
            j = int(np.argmin(np.abs(tps_arr - ip)))
            nearest_tp = float(tps_arr[j])
            abs_diff = abs(ip - nearest_tp)
            rel = abs_diff / width if width else np.nan
            # AGSM section containing the transition: section centers are equal-width;
            # half-width = (centers[1]-centers[0])/2
            centers = D['centers']
            hw = (centers[1] - centers[0]) / 2.0 if len(centers) > 1 else 0.0
            # find section index whose [center-hw, center+hw] contains nearest_tp
            sec_lo = nearest_tp - hw
            sec_hi = nearest_tp + hw
            # the transition lies between two section centers; its containing section
            # is the section center nearest the transition.
            kidx = int(np.argmin(np.abs(centers - nearest_tp)))
            csec_lo = centers[kidx] - hw
            csec_hi = centers[kidx] + hw
            in_sec = bool(csec_lo <= ip <= csec_hi)
            rows.append(dict(
                dataset=name, feature=D['top_feat'],
                kan_inflection_raw=ip,
                agsm_transition_raw=nearest_tp,
                abs_diff=abs_diff,
                domain_width=width,
                relative_error=rel,
                kan_in_agsm_section=in_sec,
            ))
    metrics = pd.DataFrame(rows, columns=[
        'dataset', 'feature', 'kan_inflection_raw', 'agsm_transition_raw',
        'abs_diff', 'domain_width', 'relative_error', 'kan_in_agsm_section'])
    out = os.path.join(FIGDIR, 'VS_sectional_gsa_metrics.csv')
    metrics.to_csv(out, index=False)
    print('Wrote', out)
    return metrics


def make_figure(data):
    plt.rcParams.update(SA_RC)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()
    panel_order = ['exponential', 'logarithm', 'log2', 'rosenbrock']
    for ax, name in zip(axes, panel_order):
        D = data[name]
        ax.plot(D['centers'], D['s_top'], drawstyle='steps-mid',
                color=TOP_COLOR, label='S_a ' + D['top_feat'])
        ax.plot(D['centers'], D['s_sec'], drawstyle='steps-mid',
                color=SECOND_COLOR, label='S_a ' + D['second_feat'])
        # KAN inflection (green dashed)
        for k, ip in enumerate(D['ip_top_raw']):
            ax.axvline(ip, color=KAN_COLOR, linestyle='--', linewidth=1.0,
                       label='KAN inflection' if k == 0 else None)
        # AGSM transition (orange dotted)
        for k, tp in enumerate(D['agsm_tps']):
            ax.axvline(tp, color=AGSM_COLOR, linestyle=':', linewidth=1.2,
                       label='AGSM transition' if k == 0 else None)
        # Analytical line for exponential
        if name == 'exponential':
            ax.axvline(EXP_TRUE, color=ANALYTIC_COLOR, linestyle='--', linewidth=1.0,
                       label='Analytical')
        ax.set_title(name, fontsize=11, fontweight='500')
        ax.set_xlabel(D['top_feat'])
        ax.set_ylabel(r'$S^{a}_{l,[k]}$')
        ax.legend(loc='best')
    fig.suptitle('Sectional GSA (AGSM) transitions vs KAN inflection points',
                 fontsize=12, fontweight='500')
    for ext in ['png', 'svg', 'eps']:
        out = os.path.join(FIGDIR, 'VS_sectional_gsa_Analytic.' + ext)
        fig.savefig(out)
        print('Wrote', out)
    plt.close(fig)


def print_summary(data, metrics):
    lines = []
    header = "Dataset         | True transition | KAN inflection | AGSM transition | KAN err | AGSM err"
    sep    = "----------------|-----------------|----------------|-----------------|---------|---------"
    lines.append(header)
    lines.append(sep)
    for name in DATASETS:
        D = data[name]
        ips = D['ip_top_raw']
        tps = D['agsm_tps']
        if name == 'exponential':
            true_t = EXP_TRUE
            kan = ips[0] if ips else np.nan
            agsm = tps[0] if tps else np.nan
            kan_err = abs(kan - true_t) if not np.isnan(kan) else np.nan
            agsm_err = abs(agsm - true_t) if not np.isnan(agsm) else np.nan
            lines.append("%-15s | %15.4f | %14.4f | %15.4f | %7.4f | %8.4f" %
                         (name, true_t, kan, agsm, kan_err, agsm_err))
        else:
            kan = ('%.4f' % ips[0]) if ips else 'none'
            agsm = ('%.4f' % tps[0]) if tps else 'none'
            kan_disp = kan if len(ips) <= 1 else ','.join('%.4f' % v for v in ips)
            agsm_disp = agsm if len(tps) <= 1 else ','.join('%.4f' % v for v in tps)
            lines.append("%-15s | %15s | %14s | %15s | %7s | %8s" %
                         (name, 'N/A', kan_disp, agsm_disp, '-', '-'))
    table = '\n'.join(lines)
    print(table)
    return table


def main():
    data = load_all()
    metrics = build_metrics(data)
    print()
    print(metrics.to_string(index=False))
    print()
    make_figure(data)
    print()
    table = print_summary(data, metrics)
    return data, metrics, table


if __name__ == '__main__':
    main()

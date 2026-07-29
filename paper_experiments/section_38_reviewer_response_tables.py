#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JOBS_ROOT = ROOT / 'output' / 'mortality_paper_jobs'
NEW_SWEEP = 'rahmatullaev_reviewer_response_mortality_reviewer_response_v1'
DATASET_MAP = {
    'mimic3_mortality_48h_tabular': 'mimic3',
    'mimic4_mortality_48h_tabular': 'mimic4',
    'eicu_mortality_48h_tabular': 'eicu',
}
METRICS = ['accuracy','balanced_accuracy','f1_macro','mcc','cohen_kappa','log_loss','auprc_ovr','brier_score','ece_10','ece_20','sensitivity','specificity','ppv','npv','net_benefit_0_10','net_benefit_0_20','roc_auc_ovr']
LOWER = {'log_loss','brier_score','ece_10','ece_20'}

def load(root: Path) -> pd.DataFrame:
    frames=[]
    for p in sorted(root.glob('*/**/compare_datasets*.csv')):
        try:
            df=pd.read_csv(p)
        except Exception:
            continue
        if df.empty or not {'dataset','fold','variant','label'}.issubset(df.columns):
            continue
        rel=p.relative_to(root).parts
        df['sweep']=rel[0]
        df['dataset_axis']=rel[1] if len(rel)>1 else ''
        df['stage']=rel[2] if len(rel)>2 else ''
        df['csv_path']=str(p)
        frames.append(df)
    out=pd.concat(frames, ignore_index=True, sort=False)
    out['dataset_short']=out['dataset'].map(DATASET_MAP).fillna(out['dataset_axis'])
    out['is_new']=out['sweep'].eq(NEW_SWEEP)
    for c in ['fold','n_test','n_branches',*METRICS]:
        if c in out.columns:
            out[c]=pd.to_numeric(out[c], errors='coerce')
    return out

def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metrics=[m for m in METRICS if m in rows]
    group=['dataset_short','sweep','stage','csv_path','is_new','rule_source','variant','label']
    agg={m:['mean','std'] for m in metrics}
    agg.update({'fold':'nunique','n_test':'sum','n_branches':'mean'})
    s=rows.groupby(group, dropna=False).agg(agg)
    s.columns=['_'.join(x).rstrip('_') for x in s.columns]
    s=s.reset_index().rename(columns={'fold_nunique':'n_folds','n_test_sum':'n_test_total','n_branches_mean':'n_branches_mean'})
    return s[s.n_folds.ge(3)].copy()

def best(df, metric):
    col=f'{metric}_mean'
    if col not in df: return pd.DataFrame()
    asc=metric in LOWER
    rows=[]
    for ds,sub in df.dropna(subset=[col]).groupby('dataset_short'):
        if sub.empty: continue
        rows.append(sub.sort_values(col, ascending=asc).iloc[0])
    return pd.DataFrame(rows)

def fmt_md(df, cols):
    if df.empty: return '_No rows._'
    d=df[cols].copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c]=d[c].map(lambda x: '' if pd.isna(x) else f'{x:.4f}')
        else:
            d[c]=d[c].fillna('').astype(str)
    widths=[len(c) for c in cols]
    vals=d.values.tolist()
    for row in vals:
        widths=[max(w,len(str(v))) for w,v in zip(widths,row)]
    def row(vs): return '| '+' | '.join(str(v).ljust(w) for v,w in zip(vs,widths))+' |'
    return '\n'.join([row(cols),'| '+' | '.join('-'*w for w in widths)+' |',*[row(r) for r in vals]])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, default=JOBS_ROOT/'common_tables'/'rahmatullaev_reviewer_response_mortality_reviewer_response_v1')
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows=load(JOBS_ROOT)
    s=summarize(rows)
    old=s[~s.is_new].copy(); new=s[s.is_new].copy()
    s.sort_values(['dataset_short','mcc_mean'], ascending=[True,False]).to_csv(args.output_dir/'mortality_all_with_reviewer_response_summary.csv', index=False)
    new.sort_values(['dataset_short','mcc_mean'], ascending=[True,False]).to_csv(args.output_dir/'reviewer_response_new_summary.csv', index=False)
    comps=[]
    for metric in ['mcc','balanced_accuracy','sensitivity','auprc_ovr','brier_score','ece_10','net_benefit_0_10','log_loss']:
        ob=best(old, metric).set_index('dataset_short')
        nb=best(new, metric).set_index('dataset_short')
        col=f'{metric}_mean'; asc=metric in LOWER
        for ds in sorted(set(ob.index)&set(nb.index)):
            o=ob.loc[ds]; n=nb.loc[ds]
            delta=(o[col]-n[col]) if asc else (n[col]-o[col])
            comps.append({'dataset':ds,'selection_metric':metric,'previous_label':o.label,'previous_sweep':o.sweep,'previous_stage':o.stage,'previous_rule_source':o.rule_source,'new_label':n.label,'new_stage':n.stage,'new_rule_source':n.rule_source,'previous_metric':o[col],'new_metric':n[col],'delta_positive_is_better':delta,
                          'previous_mcc':o.get('mcc_mean',np.nan),'new_mcc':n.get('mcc_mean',np.nan),
                          'previous_balanced_accuracy':o.get('balanced_accuracy_mean',np.nan),'new_balanced_accuracy':n.get('balanced_accuracy_mean',np.nan),
                          'previous_sensitivity':o.get('sensitivity_mean',np.nan),'new_sensitivity':n.get('sensitivity_mean',np.nan),
                          'previous_brier_score':o.get('brier_score_mean',np.nan),'new_brier_score':n.get('brier_score_mean',np.nan),
                          'previous_ece_10':o.get('ece_10_mean',np.nan),'new_ece_10':n.get('ece_10_mean',np.nan)})
    comp=pd.DataFrame(comps)
    comp.to_csv(args.output_dir/'previous_best_vs_reviewer_response_new.csv', index=False)
    top=s.sort_values(['dataset_short','mcc_mean'], ascending=[True,False]).groupby('dataset_short').head(25)
    top.to_csv(args.output_dir/'dataset_top25_mcc_with_reviewer_response.csv', index=False)
    lines=['# Reviewer-Response Sweep Leaderboard','',f'New sweep: `{NEW_SWEEP}`','','## Best New By MCC','']
    cols=['dataset_short','label','stage','rule_source','mcc_mean','balanced_accuracy_mean','sensitivity_mean','auprc_ovr_mean','brier_score_mean','ece_10_mean']
    lines.append(fmt_md(best(new,'mcc').sort_values('dataset_short'), cols)); lines += ['','## Previous Best vs New Best by MCC','']
    show=['dataset','previous_label','previous_metric','new_label','new_metric','delta_positive_is_better','new_balanced_accuracy','new_sensitivity','new_brier_score','new_ece_10']
    lines.append(fmt_md(comp[comp.selection_metric.eq('mcc')].sort_values('dataset'), show)); lines += ['','## Best New By Sensitivity','']
    lines.append(fmt_md(best(new,'sensitivity').sort_values('dataset_short'), cols)); lines += ['','## Best New By Brier','']
    lines.append(fmt_md(best(new,'brier_score').sort_values('dataset_short'), cols))
    (args.output_dir/'REVIEWER_RESPONSE_LEADERBOARD.md').write_text('\n'.join(lines)+'\n')
    print(f'raw_rows={len(rows)} complete={len(s)} new_complete={len(new)}')
    for f in ['reviewer_response_new_summary.csv','previous_best_vs_reviewer_response_new.csv','dataset_top25_mcc_with_reviewer_response.csv','REVIEWER_RESPONSE_LEADERBOARD.md']:
        print(args.output_dir/f)
if __name__=='__main__': main()

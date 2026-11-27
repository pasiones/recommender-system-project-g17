"""Export Track A using the unchanged course template and plot the sweep."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openpyxl import load_workbook


def save_part_3(results, groups, protocol, out):
    out = Path(out)
    workbook = load_workbook(out / 'results.xlsx')
    workbook['Instructions']['B6'] = 'A'
    sheet = workbook['Part 3A - Diversity']
    records = []
    for row in results.to_dict('records'):
        notes = (f"Full unseen catalogue; rank relevance; {int(row['evaluated_users'])} users; "
                 f"calibration n={row['calibration_users']}; TV distance; "
                 f"NDCG 95% CI {row['ndcg_ci']}; paired delta vs base {row['paired_difference_ci']}")
        records.append([row['model'], 'Unmodified base' if row['trade_off'] == 1 else 'MMR',
                        row['trade_off'], 'Cosine distance on multi-hot genres',
                        row['precision@10'], row['recall@10'], row['ndcg@10'],
                        row['intra_list_diversity'], row['catalogue_coverage'],
                        row['gini'], row['calibration_error'], notes])
    for r, values in enumerate(records, 7):
        for c, value in enumerate(values, 1):
            sheet.cell(r, c, value)
    workbook.save(out / 'results.xlsx')
    headers = [sheet.cell(6, c).value for c in range(1, 13)]
    pd.DataFrame(records, columns=headers).to_csv(out / 'part_3a_diversity.csv', index=False)
    results.to_csv(out / 'part_3a_metrics.csv', index=False)
    groups.to_csv(out / 'part_3a_activity.csv', index=False)
    (out / 'part_3a_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(results.intra_list_diversity, results['ndcg@10'], 'o-')
    for row in results.to_dict('records'):
        offset = {0.0: (-85, -8), 0.1: (-85, -26), 0.25: (-85, -44),
                  0.5: (-70, 12)}.get(row['trade_off'], (5, 5))
        axes[0].annotate(f"λ={row['trade_off']:g}",
                         (row['intra_list_diversity'], row['ndcg@10']), fontsize=8,
                         xytext=offset, textcoords='offset points',
                         arrowprops=dict(arrowstyle='-', color='gray', lw=.5))
        axes[0].vlines(row['intra_list_diversity'], *row['ndcg_ci'], alpha=.4)
    axes[0].set(xlabel='Intra-list diversity (cosine distance)', ylabel='NDCG@10',
                title='Accuracy–diversity sweep (95% bootstrap CI)')
    for column in ['catalogue_coverage', 'gini', 'calibration_error']:
        axes[1].plot(results.trade_off, results[column], 'o-', label=column.replace('_', ' '))
    axes[1].set(xlabel='λ (1 = unmodified base)', ylabel='Metric value', title='Exposure and calibration')
    axes[1].legend(loc='center', bbox_to_anchor=(.6, .7))
    fig.tight_layout()
    fig.savefig(out / 'part_3a_frontier.png', dpi=150)
    plt.close(fig)
"""Merge per-condition factor results into a primary-sweep results file.

Usage:
    python3 tools/merge_factor_results.py PRIMARY_JSON FACTOR_DIR OUT_JSON

Reads bubble_factor_*.json files (written by test_filter_bubbles.py
--single-factor) from FACTOR_DIR, inserts them as rumor/disaster/mix results
into a copy of PRIMARY_JSON, and writes OUT_JSON, which then supports
`test_filter_bubbles.py --plots-only` including the factor-comparison figure.
"""
import glob
import json
import os
import sys


def main(primary_path, factor_dir, out_path):
    with open(primary_path) as f:
        primary = json.load(f)
    buckets = {'rumor': {}, 'disaster': {}, 'mix': {}}
    for path in sorted(glob.glob(os.path.join(factor_dir, 'bubble_factor_*.json'))):
        with open(path) as f:
            d = json.load(f)
        key = d['factor_value']
        if d['factor_type'] == 'disaster':
            key = int(key)
        buckets[d['factor_type']][key] = d['result']
    primary['rumor_results'] = {str(k): v for k, v in buckets['rumor'].items()}
    primary['disaster_results'] = {str(k): v for k, v in buckets['disaster'].items()}
    primary['mix_results'] = {str(k): v for k, v in buckets['mix'].items()}
    with open(out_path, 'w') as f:
        json.dump(primary, f)
    print('merged:', {t: sorted(b) for t, b in buckets.items()})


if __name__ == '__main__':
    main(*sys.argv[1:4])

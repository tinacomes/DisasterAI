"""Download a workflow run's artifacts with proper pagination.

actions/download-artifact@v4 only sees the first page (~100) of a run's
artifacts, silently dropping the rest on large matrix runs (the S7 gap sweep
uploads 133). This script paginates the artifacts API, filters by a glob
pattern (with {a,b} brace alternatives, matching the archive workflow's
input syntax), and unzips each artifact into DEST/<artifact-name>/.

Usage:
    GH_TOKEN=... python3 tools/download_run_artifacts.py OWNER/REPO RUN_ID PATTERN DEST
"""
import fnmatch
import io
import json
import os
import sys
import urllib.request
import zipfile


def expand_braces(pattern):
    start = pattern.find('{')
    if start == -1:
        return [pattern]
    depth, end = 0, -1
    for i in range(start, len(pattern)):
        if pattern[i] == '{':
            depth += 1
        elif pattern[i] == '}':
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return [pattern]
    out = []
    for alt in pattern[start + 1:end].split(','):
        out.extend(expand_braces(pattern[:start] + alt + pattern[end + 1:]))
    return out


def api(url, token):
    req = urllib.request.Request(url, headers={
        'Authorization': f'Bearer {token}',
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'archive-artifacts',
    })
    return urllib.request.urlopen(req)


def main(repo, run_id, pattern, dest):
    token = os.environ['GH_TOKEN']
    variants = expand_braces(pattern)
    page, matched, total = 1, 0, 0
    while True:
        url = (f'https://api.github.com/repos/{repo}/actions/runs/{run_id}'
               f'/artifacts?per_page=100&page={page}')
        with api(url, token) as r:
            data = json.load(r)
        arts = data.get('artifacts', [])
        if not arts:
            break
        for a in arts:
            total += 1
            if a['expired'] or not any(fnmatch.fnmatch(a['name'], v) for v in variants):
                continue
            matched += 1
            out_dir = os.path.join(dest, a['name'])
            os.makedirs(out_dir, exist_ok=True)
            with api(a['archive_download_url'], token) as r:
                blob = r.read()
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                z.extractall(out_dir)
            print(f'  {a["name"]} -> {out_dir}')
        page += 1
    print(f'{matched} of {total} artifacts matched pattern {pattern!r}')
    if matched == 0:
        sys.exit(f'No artifacts matched {pattern!r} in run {run_id}')


if __name__ == '__main__':
    main(*sys.argv[1:5])

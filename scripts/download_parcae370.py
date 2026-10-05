"""Download only pinned 370M artifacts into a fresh project-private HF layout."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from prepare_parcae370 import FILES,stream_fingerprint


def fetch(url,destination,spec,update,opener=urllib.request.urlopen):
    """Bounded streaming with exact length/SHA; never promote partial bytes."""
    destination=Path(destination);temporary=destination.with_suffix('.partial')
    if destination.exists() or temporary.exists():raise FileExistsError('Fresh artifact required')
    count=0;h=hashlib.sha256();reported=0
    with opener(url,timeout=60) as response,temporary.open('xb') as output:
        while True:
            block=response.read(8*1024*1024)
            if not block:break
            count+=len(block)
            if count>spec['bytes']:raise ValueError('Artifact exceeds registered size')
            h.update(block);output.write(block)
            if count-reported>=64*1024*1024:update(count);reported=count
    if count!=spec['bytes'] or h.hexdigest()!=spec['sha256']:raise ValueError('Artifact length or SHA mismatch')
    temporary.rename(destination);update(count)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True,help='Previously pinned local tokenizer, read-only')
    a=p.parse_args();root=a.cache_dir.resolve();root.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',started_at=datetime.now(timezone.utc).isoformat(),files={},shared_cache_modified=False)
    def save():
        t=root/'status.tmp';t.write_text(json.dumps(state,indent=2)+'\n');t.replace(root/'download_status.json')
    save()
    try:
        for name,spec in FILES.items():
            repository=root/('models--'+spec['repo_id'].replace('/','--'))
            blob=repository/'blobs'/spec['sha256'];blob.parent.mkdir(parents=True,exist_ok=True)
            url=f"https://huggingface.co/{spec['repo_id']}/resolve/{spec['revision']}/{name}"
            state['active_file']=name;state['files'][name]=dict(url=url,expected=spec,status='running',bytes_received=0);save()
            def update(count):state['files'][name]['bytes_received']=count;save()
            if name=='tokenizer.json':
                if stream_fingerprint(a.tokenizer)!={k:spec[k] for k in ('bytes','sha256')}:raise ValueError('Tokenizer changed')
                shutil.copyfile(a.tokenizer,blob)
            else:fetch(url,blob,spec,update)
            if stream_fingerprint(blob)!={k:spec[k] for k in ('bytes','sha256')}:raise ValueError('Stored artifact changed')
            snapshot=repository/'snapshots'/spec['revision'];snapshot.mkdir(parents=True,exist_ok=True)
            (snapshot/name).symlink_to(blob)
            state['files'][name].update(status='verified',bytes_received=spec['bytes']);save()
        state.update(status='PASS',finished_at=datetime.now(timezone.utc).isoformat());save()
    except BaseException as exc:
        state.update(status='FAIL',error=repr(exc));save();raise

if __name__=='__main__':main()

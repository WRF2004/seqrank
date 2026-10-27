"""Optional real CPU DDP integration, explicitly enabled by the server operator."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


@pytest.mark.skipif(os.getenv('SEQRANK_DDP_TESTS')!='1',reason='Set SEQRANK_DDP_TESTS=1 to launch two CPU ranks')
@pytest.mark.parametrize('stage',['retrieval','ranking'])
def test_two_rank_checkpoint_and_resume(prepared,tmp_path,stage):
    config=prepared
    config['training'].update(device='cpu',epochs=1,batch_size=4,workers=0)
    config['threads']=1
    config['run']['output']=str(tmp_path/'distributed')
    path=tmp_path/'resolved.yaml'
    path.write_text(yaml.safe_dump(config))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1')
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
             '-m','seqrank.cli',f'train-{stage}','--config',str(path)]
    subprocess.run(command,check=True,env=env,timeout=180)
    saved=Path(config['run']['output'])/stage/'last.pt'
    checkpoint=torch.load(saved,weights_only=True)
    assert checkpoint['world_size']==2 and len(checkpoint['rng'])==2
    assert checkpoint['epoch']==0
    # Already completed epoch plan: reload then exit without an extra update.
    subprocess.run(command+['--resume',str(saved)],check=True,env=env,timeout=180)

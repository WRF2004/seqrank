"""Configuration-first research CLI; legacy commands remain in seqrank.cli."""
import argparse
import json
from pathlib import Path

from .config import load_config, file_hash

COMMANDS = {'schema','convert','prepare-data','quality','verify-data','train-retrieval','train-ranking',
            'evaluate-retrieval','evaluate-ranking','evaluate-e2e','build-ann','benchmark-ann',
            'diagnose-gradients','serve-research','benchmark','experiments','report'}


def main(argv=None):
    parser=argparse.ArgumentParser(description='SeqRank research v2: explicit data, tasks and protocols')
    parser.add_argument('command',choices=sorted(COMMANDS))
    parser.add_argument('--config',required=True)
    parser.add_argument('--set',action='append',default=[],metavar='KEY=YAML_VALUE')
    parser.add_argument('--split',choices=['valid','test'],default='test')
    parser.add_argument('--resume')
    parser.add_argument('--execute',action='store_true',help='Execute experiment matrix; otherwise write resolved plan only')
    args=parser.parse_args(argv)
    config=load_config(args.config,args.set)
    import torch
    torch.set_num_threads(int(config.get('threads',4)))
    cmd=args.command
    if cmd in {'schema','convert'}:
        from .adapters import inspect,convert
        result=(inspect if cmd=='schema' else convert)(config)
    elif cmd=='prepare-data':
        from .dataset import prepare_data
        result=prepare_data(config)
    elif cmd=='quality':
        from .quality import quality_report
        result=quality_report(config)
    elif cmd=='verify-data':
        from .dataset import load_artifacts
        manifest,_,_=load_artifacts(config['data']['prepared'])
        for shards in manifest['shards'].values():
            for shard in shards:
                if file_hash(Path(config['data']['prepared'])/shard['file'])!=shard['sha256']:
                    raise ValueError(f"Shard checksum mismatch: {shard['file']}")
        result={'verified':manifest['fingerprint']}
    elif cmd.startswith('train-'):
        from .training import train
        train(config,cmd.split('-')[1],args.resume)
        result={'output':config['run']['output'],'stage':cmd.split('-')[1]}
    elif cmd.startswith('evaluate-'):
        from .evaluation import evaluate
        result=evaluate(config,cmd.split('-')[1],args.split)
    elif cmd=='build-ann':
        from .dataset import load_artifacts
        from .evaluation import checkpoint_path
        from .training import load_checkpoint
        from .retrievers import build_index
        manifest,_,_=load_artifacts(config['data']['prepared'])
        path=checkpoint_path(config,'retrieval')
        model,_=load_checkpoint(path,config.get('evaluation',{}).get('device','cpu'),manifest['fingerprint'])
        result=build_index(model,config['ann']['path'],file_hash(path),manifest['fingerprint'],config['ann'])
    elif cmd=='benchmark-ann':
        from .evaluation import benchmark_ann
        result=benchmark_ann(config)
    elif cmd=='diagnose-gradients':
        from .diagnostics import diagnose
        result=diagnose(config)
    elif cmd=='serve-research':
        import uvicorn
        from .serving import create_app
        spec=config.get('serving',{})
        uvicorn.run(create_app(config),host=spec.get('host','127.0.0.1'),port=int(spec.get('port',8000)))
        return
    elif cmd=='benchmark':
        from .serving import benchmark
        result=benchmark(config)
    elif cmd=='experiments':
        from .experiments import matrix
        result=matrix(config,args.execute)
    else:
        from .experiments import report
        result=report(config)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

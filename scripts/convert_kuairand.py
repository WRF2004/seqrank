"""Compatibility wrapper. New experiments should use seqrank.cli convert --config.

Legacy output has a click field only: conversion now requires explicit acknowledgement
that the selected input contains genuine two-column click semantics.
"""
import argparse
import csv
from pathlib import Path


def convert(sources, output, *, verified_click_semantics=False):
    if not verified_click_semantics:
        raise ValueError('Ambiguous is_click: use v2 adapter, or explicitly verify/filter genuine click traffic')
    target=Path(output)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    target.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with target.open('x',newline='') as out:
        writer=csv.writer(out)
        writer.writerow(['user_id','item_id','timestamp','click','conversion','dwell'])
        for path in sources:
            with open(path,newline='') as source:
                rows=csv.DictReader(source)
                required={'user_id','video_id','time_ms','is_click','play_time_ms'}
                if not required<=set(rows.fieldnames or []):
                    raise ValueError(f'{path}: missing {required-set(rows.fieldnames or [])}')
                for row in rows:
                    writer.writerow([row['user_id'],row['video_id'],row['time_ms'],row['is_click'],-1,
                                     float(row['play_time_ms'])/1000])
                    count+=1
    return count


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--verified-click-semantics',action='store_true')
    args=parser.parse_args()
    print({'rows':convert(args.input,args.output,verified_click_semantics=args.verified_click_semantics)})

"""Exact catalogue scoring for neural models, popularity and sparse ItemKNN."""
from __future__ import annotations

import time
import os
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

from .config import file_hash, write_json


def resident_bytes():
    """Linux current process RSS; includes allocator overhead, not just index storage."""
    try:
        return int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, IndexError):
        return None


class CatalogueBaseline:
    def __init__(self, state: dict):
        self.kind = state['kind']
        self.n_items = int(state['n_items'])
        self.state = state
        self.popularity = state['popularity'].numpy()
        if self.kind=='itemknn':
            self.similarity = sparse.csr_matrix((state['values'].numpy(),state['indices'].numpy(),state['indptr'].numpy()),
                                                 shape=(self.n_items,self.n_items))

    def eval(self):
        return self

    def score_block(self, history: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        if self.kind=='popularity':
            scores = np.broadcast_to(self.popularity[start:stop],(history.size(0),stop-start)).copy()
        else:
            h = history.cpu().numpy()
            r = np.repeat(np.arange(len(h)),h.shape[1])
            ids = h.reshape(-1)
            mask = ids>=2
            x = sparse.csr_matrix((np.ones(mask.sum()),(r[mask],ids[mask])),shape=(len(h),self.n_items))
            scores = (x@self.similarity[:,start:stop]).toarray()
            # Empty known history uses train popularity, explicitly defined cold-start rule.
            empty = (h>=2).sum(-1)==0
            scores[empty] = self.popularity[start:stop]
        return torch.as_tensor(scores,dtype=torch.float32,device=history.device)


def fit_baseline(dataset, spec: dict) -> dict:
    popularity = np.load(dataset.root/'popularity.npy').astype(np.float32)
    state = {'kind':spec['model'],'n_items':len(popularity),'popularity':torch.from_numpy(popularity)}
    if spec['model']=='popularity':
        return state
    if spec['model']!='itemknn':
        raise ValueError('Expected popularity or itemknn')
    user, item = [], []
    for i in range(len(dataset)):
        r = dataset[i]
        if r['label']==1 and r['item']>=2:
            user.append(int(r['user']))
            item.append(int(r['item']))
    x = sparse.coo_matrix((np.ones(len(item)),(user,item)),shape=(dataset.manifest['n_users'],len(popularity))).tocsr()
    x.data[:] = 1  # binary unique user-item support
    norms = np.sqrt(np.asarray(x.sum(0)).ravel()).clip(1)
    x = x @ sparse.diags(1/norms)
    result = []
    top = int(spec.get('knn_neighbors',100))
    block = int(spec.get('knn_block',256))
    if top<1 or block<1:
        raise ValueError('KNN neighbor count and block must be positive')
    for start in range(0,x.shape[1],block):
        co = (x[:,start:start+block].T@x).tocsr()
        for row in range(co.shape[0]):
            a,b = co.indptr[row:row+2]
            idx, values = co.indices[a:b], co.data[a:b]
            keep = (idx!=start+row)&(idx>=2)
            idx,values = idx[keep],values[keep]
            order = np.argsort(-values,kind='stable')[:top]
            result.append(sparse.csr_matrix((values[order],(np.zeros(len(order)),idx[order])),shape=(1,x.shape[1])))
    matrix = sparse.vstack(result).tocsr()
    state.update(values=torch.from_numpy(matrix.data.astype(np.float32)),
                 indices=torch.from_numpy(matrix.indices.astype(np.int64)),
                 indptr=torch.from_numpy(matrix.indptr.astype(np.int64)))
    return state


@torch.inference_mode()
def exact_topk(model, history: torch.Tensor, k=100, chunk=4096, exclude_seen=False):
    if k<1 or chunk<1:
        raise ValueError('k and chunk must be positive')
    k = min(k,model.n_items-2)
    query = None if isinstance(model,CatalogueBaseline) else model.user_vectors(history)
    scores = torch.empty(history.size(0),0,device=history.device)
    ids = torch.empty(history.size(0),0,device=history.device,dtype=torch.long)
    for start in range(2,model.n_items,chunk):
        stop = min(model.n_items,start+chunk)
        block_ids = torch.arange(start,stop,device=history.device)
        s = model.score_block(history,start,stop) if query is None else model.score(query,block_ids)
        if exclude_seen:
            row = torch.arange(history.size(0),device=history.device)[:,None].expand_as(history)
            mask = (history>=start)&(history<stop)
            s[row[mask],history[mask]-start] = -torch.inf
        merged = torch.cat([scores,s],1)
        merged_ids = torch.cat([ids,block_ids.expand(history.size(0),-1)],1)
        # Stable sort makes tie policy deterministic: lowest ID first (blocks ascending).
        # Re-sort by ID before score so previous top-k ordering cannot perturb ties.
        by_id = merged_ids.argsort(dim=1,stable=True)
        merged, merged_ids = merged.gather(1,by_id), merged_ids.gather(1,by_id)
        order = merged.argsort(dim=1,descending=True,stable=True)[:,:k]
        scores,ids = merged.gather(1,order),merged_ids.gather(1,order)
    ids = ids.masked_fill(~torch.isfinite(scores),0)
    return scores,ids


@torch.inference_mode()
def build_index(model, path, checkpoint_hash, dataset_hash, spec):
    import faiss
    if isinstance(model,CatalogueBaseline):
        raise ValueError('HNSW export only supports GRU/SASRec embeddings; baselines use sparse/exact scoring')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    rss_before = resident_bytes()
    start = time.perf_counter()
    index = faiss.IndexHNSWFlat(model.dim,int(spec.get('m',32)),faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = int(spec.get('ef_construction',200))
    for first in range(2,model.n_items,10000):
        ids = torch.arange(first,min(first+10000,model.n_items),device=next(model.parameters()).device)
        vectors = model.item_vectors(ids).cpu().numpy().astype('float32')/np.sqrt(model.dim)
        index.add(np.ascontiguousarray(vectors))
    faiss.write_index(index,str(path))
    rss_after = resident_bytes()
    meta = {'process_rss_before':rss_before,'process_rss_with_built_index':rss_after,
            'process_rss_delta':rss_after-rss_before if rss_before is not None and rss_after is not None else None,
            'memory_note':'RSS delta includes temporary vectors/allocator; serialized_bytes is exact index file size',
            'checkpoint_hash':checkpoint_hash,'dataset_hash':dataset_hash,'dim':model.dim,
            'n_items':model.n_items,'offset':2,'metric':'inner_product/sqrt(dim)',
            'index_sha256':file_hash(path),'build_seconds':time.perf_counter()-start,
            'serialized_bytes':path.stat().st_size,'spec':spec}
    write_json(str(path)+'.json',meta)
    return meta


class ANN:
    def __init__(self,path,checkpoint_hash,dataset_hash,ef_search=128):
        import faiss
        import json
        self.meta = json.loads(Path(str(path)+'.json').read_text())
        if self.meta['checkpoint_hash']!=checkpoint_hash or self.meta['dataset_hash']!=dataset_hash:
            raise ValueError('ANN/model/dataset identity mismatch')
        if self.meta['index_sha256']!=file_hash(path):
            raise ValueError('ANN bytes changed')
        self.index = faiss.read_index(str(path))
        if self.index.ntotal!=self.meta['n_items']-2 or self.index.d!=self.meta['dim']:
            raise ValueError('ANN shape differs from metadata')
        self.index.hnsw.efSearch = int(ef_search)

    def search(self,query,k):
        scores,ids = self.index.search(np.ascontiguousarray(query,dtype='float32'),min(k,self.index.ntotal))
        return scores,np.where(ids>=0,ids+2,0)

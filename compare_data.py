import os
import multiprocessing

import numpy as np
import igraph as ig
import pandas as pd
import networkx as nx

from tqdm import tqdm
from collections import deque
from matplotlib import pyplot as plt

from HBN import *

from goatools.obo_parser import GODag
from goatools.semantic import deepest_common_ancestor, common_parent_go_ids
from goatools.godag.go_tasks import get_go2parents
from goatools.gosubdag.gosubdag import GoSubDag
from goatools.gosubdag.plot.gosubdag_plot import GoSubDagPlot

# Node embedding
# from pecanpy import pecanpy, node2vec
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation

# Cross-validation and scaler
from sklearn.preprocessing import MinMaxScaler

g = GODag("data/go-basic.obo")

# %%

g2t = pd.read_csv("data/data_gene_term.csv")
terms = g2t.Term.unique().tolist()
terms = [t for t in terms if t in g and t != "GO:0008150"]
print("Number of terms (non-obsolet): {0}".format(len(terms)))
g2t = g2t[g2t.Term.isin(terms)].reset_index(drop=True)
# g2t.to_csv("data/gene_term.csv", index=False)
print("Number of relations: {0}".format(len(g2t)))

isa = list()
for t in tqdm(terms):
  q = deque()
  for p in g[t].parents:
    q.append((t, p.id))

  while len(q) > 0:
    c, p = q.pop()
    if p != "GO:0008150":
      isa.append((c,p))
      for gp in g[p].parents:
        q.append((p, gp.id))

isa = pd.DataFrame(isa, columns=['Child','Parent'])
isa = isa.drop_duplicates().reset_index(drop=True)

all_terms = np.union1d(np.union1d(isa.Child, isa.Parent), terms)
term_def = pd.DataFrame()
term_def["Term"] = all_terms
term_def["Desc"] = [g[t].name for t in all_terms]

print('Number of terms: {0}'.format(len(all_terms)))
print('Number of relations: {0}'.format(len(isa)))

data_gcn = pd.read_csv("data/data_gcn.csv")
genes = np.union1d(data_gcn.Source, data_gcn.Target)
ng, idg = len(genes), dict([(g,i) for i,g in enumerate(genes)])

terms = all_terms.copy()
nt, idt = len(terms), dict([(t,i) for i,t in enumerate(terms)])

# GCN matrix
# ng:number of genes, idg:gene index map
gcn = np.zeros((ng,ng))
for edge in tqdm([tuple(x) for x in data_gcn.to_numpy()]):
  u, v = idg[edge[0]], idg[edge[1]]
  gcn[u][v] = gcn[v][u] = 1

# go by go matrix
# nt:number of terms, idt:term index map
go_by_go = np.zeros((nt,nt))
for edge in tqdm([tuple(x) for x in isa.to_numpy()]):
  u, v = idt[edge[0]], idt[edge[1]]
  go_by_go[u,v] = 1

# compute the transitive closure of the ancestor of a term (idx)
def ancestors(term):
  tmp = np.nonzero(go_by_go[term,:])[0]
  ancs = list()
  while len(tmp) > 0:
    tmp1 = list()
    for i in tmp:
      ancs.append(i)
      tmp1 += np.nonzero(go_by_go[i,:])[0].tolist()
    tmp = list(set(tmp1))
  return ancs

# gene by go matrix
gene_by_go = np.zeros((ng,nt))
for edge in tqdm([tuple(x) for x in g2t.to_numpy()]):
  u, v = idg[edge[0]], idt[edge[1]]
  gene_by_go[u,v] = 1
  gene_by_go[u,ancestors(v)] = 1

print()
print('**Old data**')
print('Genes: \t\t{0:6}'.format(ng))
print('Gene annot.: \t{0:6}'.format(np.count_nonzero(gene_by_go)))
print('Co-expression: \t{0:6.0f}'.format(np.sum(gcn)/2))
print('GO terms: \t{0:6}'.format(nt))
print('GO hier.: \t{0:6.0f}'.format(np.sum(go_by_go)))

# %%

g2t = pd.read_csv("data/gene_term_new.csv")
g2t = g2t[g2t.Term.isin(terms)].reset_index(drop=True)
print("Number of relations: {0}".format(len(g2t)))

# gene by go matrix
new_gene_by_go = np.zeros((ng,nt))
for edge in tqdm([tuple(x) for x in g2t.to_numpy()]):
  if edge[0] in idg:
    u, v = idg[edge[0]], idt[edge[1]]
    new_gene_by_go[u,v] = 1
    new_gene_by_go[u,ancestors(v)] = 1

print()
print('**New data**')
print('Gene annot.: \t{0:6}'.format(np.count_nonzero(new_gene_by_go)))

# %%

#####################################
# 2. Prepare term data for prediction
#####################################

# Graph for subhiearchies creation
g2g_edg = np.transpose(np.nonzero(np.transpose(go_by_go))).tolist()
g2g = nx.DiGraph()
g2g.add_nodes_from(np.arange(nt))
g2g.add_edges_from(g2g_edg)
print('GO graph (all terms): nodes {0}, edges {1}'.format(g2g.number_of_nodes(), g2g.number_of_edges()))
print('Number of weakly conn. components: {}'.format(nx.number_weakly_connected_components(g2g)))

# Prune terms according to paper, very specific and extremes with little to
# no information terms are avoided. Select genes used for prediction
# Accoding to restriction 5 <= genes annotated <= 300
ft_idx = list() # list of terms filtered according to the previous explanation
for i in range(nt):
  if 10 <= np.count_nonzero(gene_by_go[:,i]) <= 300:
    ft_idx.append(i)
print('Number of filtered terms: {0}'.format(len(ft_idx)))

# Including the ancestor of the selected terms
pt_idx = list(ft_idx)
for i in ft_idx:
  pt_idx += np.nonzero(go_by_go[i,:])[0].tolist()
pt_idx = np.array(sorted(list(set(pt_idx))))
print('Number of filtered terms incl. parents: {0}'.format(len(pt_idx)))

# Subgraph from terms to predict
sub_go_by_go = go_by_go[np.ix_(pt_idx,pt_idx)].copy()
sg2g_edg = np.transpose(np.nonzero(np.transpose(sub_go_by_go))).tolist()
sg2g = nx.DiGraph()
sg2g.add_nodes_from(np.arange(len(pt_idx)))
sg2g.add_edges_from(sg2g_edg)
print('GO subgraph (pred terms): nodes {0}, edges {1}'.format(sg2g.number_of_nodes(), sg2g.number_of_edges()))
print('Number of weakly conn. components: {}'.format(nx.number_weakly_connected_components(sg2g)))

# %%

# find possible root terms in go subgraph
proot_idx = list() # possible hierarchy roots
for i in range(len(pt_idx)):
  if np.count_nonzero(sub_go_by_go[i,:]) == 0: # terms wo ancestors
    proot_idx.append(i)
proot_idx = np.array(proot_idx)
print('Number of roots in GO subgraph: {0}'.format(len(proot_idx)))

# convert a bfs object to a list
def nodes_in_bfs(bfs, root):
  nodes = sorted(list(set([u for u,v in bfs] + [v for u,v in bfs])))
  nodes = np.setdiff1d(nodes, [root]).tolist()
  nodes = [root] + nodes
  return nodes

# detect isolated terms and create sub-hierarchies
hpt = list() # terms to predict and all terms in hierarchy
hroot_idx = list()
for root in proot_idx:
  bfs = nx.bfs_tree(sg2g, root).edges()

  if len(bfs) > 0: # if no isolated term
    hroot_idx.append(pt_idx[root])
    hpt.append(pt_idx[nodes_in_bfs(bfs, root)])

hroot_idx = np.array(hroot_idx)
len_hpt = [len(x) for x in hpt]
print('Number of isolated terms: {0}'.format(len(proot_idx)-len(hroot_idx)))
print('Number of sub-hierarchies: {0}'.format(len(hroot_idx)))
print('Average terms in sub-hierarchies: {0:.2f} [{1}-{2}]'.format(
  np.mean(len_hpt),
  np.min(len_hpt),
  np.max(len_hpt)))

# %%

# list sub-hierarchies
df_subh = pd.DataFrame(columns=['Root_idx','Root','Terms','Genes','Desc','Level'])
for i, rid in enumerate(hroot_idx):
  root = terms[rid]
  data = [rid, root, len(hpt[i])] # number of terms to predict in sub-hier.
  data += [np.count_nonzero(gene_by_go[:,rid])] # number of genes in sub.
  data += [term_def[term_def.Term==root].Desc.tolist()[0], g[root].level]
  df_subh.loc[i] = data

df_subh = df_subh.sort_values(by=['Terms','Genes'], ascending=False).reset_index(drop=True)
# df_subh.to_csv('data/subhierarchies.csv', index=False)
# print(str(df_subh))

# %%

# sub-hierarchies used for prediction
test_df_subh = df_subh[df_subh.Terms >= 9].sort_values(by=['Terms','Genes'], ascending=True).reset_index(drop=True)
test_r = test_df_subh.Root.tolist()
test_rid = test_df_subh.Root_idx.tolist()

test_hpt = list()
for i, root in enumerate(test_rid):
  idx = np.where(hroot_idx==root)[0][0]
  test_hpt.append(hpt[idx])

print(test_df_subh)

# %%

def list2file(l, name):
  file = open(name, 'w')
  file.write('\n'.join([str(x) for x in l]))
  file.close()


def create_path(path):
  try: os.makedirs(path)
  except: pass


# Scale data
def scale_data(data):
  # MinMaxScaler does not modify the distribution of data
  minmax_scaler = MinMaxScaler() # Must be first option

  new_data = pd.DataFrame()
  for fn in data.columns:
    scaled_feature = minmax_scaler.fit_transform(data[fn].values.reshape(-1,1))
    new_data[fn] = scaled_feature[:,0].tolist()

  return new_data

# %%

# compute graph properties and feature embedding for each sub-hierarchy
root_list = list()
for i, (root, hterms) in enumerate(zip(test_rid, test_hpt)):

  term = terms[root]
  hgenes = np.nonzero(gene_by_go[:,root])[0] # genes associated to root in old data
  n_hgenes = np.nonzero(new_gene_by_go[:,root])[0] # genes associated to root in new data
  t_hgenes = np.setdiff1d(n_hgenes, hgenes) # genes associated to root in new data but not associated in old data
  n_hgenes = np.union1d(hgenes, n_hgenes) # genes associated to root in any of the datasets

  ''' Get giant connected component '''
  sgcn_adj = gcn[np.ix_(n_hgenes,n_hgenes)].copy() # create sub matrix terms_hier_idx hierarchy
  sgcn_edgelist = [(x,y) for x,y in np.transpose(np.nonzero(sgcn_adj)).tolist()]
  sgcn = nx.Graph() # create graph for gcn (hierarchy gcn)
  sgcn.add_nodes_from(np.arange(len(n_hgenes)))
  sgcn.add_edges_from(sgcn_edgelist)
  gcc = sorted(nx.connected_components(sgcn), key=len, reverse=True)
  sgcn = sgcn.subgraph(gcc[0])
  assert nx.is_connected(sgcn)
  n_hgenes = n_hgenes[sgcn.nodes]
  hgenes = np.intersect1d(n_hgenes, hgenes)
  t_hgenes = np.intersect1d(n_hgenes, t_hgenes)

  gbgO = gene_by_go[np.ix_(hgenes,hterms)].copy() # GCN for old data
  gbgN = new_gene_by_go[np.ix_(hgenes,hterms)].copy() # GCN for old genes in new data

  if gbgN.sum() - gbgO.sum() > 0:
    root_list.append(root)
    path = "data/{0}".format(term.replace(':',''))
    create_path(path)

    list2file(np.searchsorted(n_hgenes, hgenes), "{0}/genes_co.txt".format(path))
    list2file(genes[np.searchsorted(n_hgenes, hgenes)], "{0}/genes.txt".format(path))
    list2file(np.searchsorted(n_hgenes, t_hgenes), "{0}/genes_test.txt".format(path))
    list2file(terms[hterms[1:]], "{0}/terms.txt".format(path))

    ''' Labels '''
    gbgT = new_gene_by_go[np.ix_(t_hgenes,hterms)].copy() # GCN for new genes in new data (test)

    gbgO_df = pd.DataFrame()
    gbgN_df = pd.DataFrame()
    gbgT_df = pd.DataFrame()
    for tidx, trm in enumerate(terms[hterms[1:]]):
      gbgO_df[trm] = pd.Series(gbgO[:,tidx+1])
      gbgN_df[trm] = pd.Series(gbgN[:,tidx+1])
      gbgT_df[trm] = pd.Series(gbgT[:,tidx+1])
    gbgO_df.to_csv('{0}/labels_old.csv'.format(path), index=False)
    gbgN_df.to_csv('{0}/labels_new.csv'.format(path), index=False)
    gbgT_df.to_csv('{0}/labels_test.csv'.format(path), index=False)

    ''' Node embedding '''
    dimensions = len(hterms)
    p, q = 1, 0.5
    seed = 220209

    sgcn_adj = gcn[np.ix_(n_hgenes,n_hgenes)].copy() # create sub matrix terms_hier_idx hierarchy
    sgcn_edgelist = [(x,y) for x,y in np.transpose(np.nonzero(sgcn_adj)).tolist()]
    sgcn = nx.Graph() # create graph for gcn (hierarchy gcn)
    sgcn.add_nodes_from(np.arange(len(n_hgenes)))
    sgcn.add_edges_from(sgcn_edgelist)
    nx.write_edgelist(sgcn, "{0}/gcn.edg".format(path), delimiter='\t')

    n2v = pecanpy.DenseOTF(p=p, q=q, workers=8, verbose=False)
    n2v.read_edg("{0}/gcn.edg".format(path), weighted=False, directed=False)
    embeddings = n2v.embed(dim=dimensions, num_walks=300, walk_length=5, window_size=5, epochs=1, verbose=False)
    assert not np.isnan(embeddings).any()
    # embeddings = n2v.embed(dim=dimensions, num_walks=10, walk_length=80, window_size=10, epochs=1, verbose=False)

    tsne = TSNE(n_components=2, random_state=7, perplexity=15) # dimensionality reduction for clustering
    embeddings_2d = tsne.fit_transform(embeddings)
    clustering_model = AffinityPropagation(damping=0.9, random_state=seed)
    clustering_model.fit(embeddings_2d)
    yhat = clustering_model.predict(embeddings_2d)

    sh_df = pd.DataFrame()
    for didx in range(dimensions):
      sh_df['emb_{0}'.format(didx)] = pd.Series(embeddings[:,didx])
    sh_df['emb_clust'] = pd.Series(yhat)
    sh_df = scale_data(sh_df)
    sh_df.to_csv('{0}/embedding.csv'.format(path), index=False)

    ''' Node properties '''
    sgcn = ig.Graph.Adjacency((sgcn_adj > 0).tolist())
    sgcn.to_undirected()
    sgcn.simplify(combine_edges="max")
    assert sgcn.is_connected() and sgcn.is_simple() and not sgcn.is_directed()
    # ig.summary(sgcn)

    # get node properties form graph
    deg = np.array(sgcn.degree())
    neigh_deg = np.array(sgcn.knn()[0])
    eccec = np.array(sgcn.eccentricity())
    clust = np.array(sgcn.transitivity_local_undirected(mode="zero"))
    centr_clos = np.array(sgcn.closeness())
    centr_betw = np.array(sgcn.betweenness(directed=False))
    hubs = np.array(sgcn.hub_score())
    auths = np.array(sgcn.authority_score())
    coren = np.array(sgcn.coreness())

    sh_df = pd.DataFrame()
    sh_df['deg'] = pd.Series(deg) # degree
    sh_df['neigh_deg'] = pd.Series(neigh_deg) # average_neighbor_degree
    sh_df['eccec'] = pd.Series(eccec) # eccentricity
    sh_df['clust'] = pd.Series(clust) # clustering
    sh_df['clos'] = pd.Series(centr_clos) # closeness_centrality
    sh_df['betw'] = pd.Series(centr_betw) # betweenness_centrality
    sh_df['hubs'] = pd.Series(hubs) # hub score
    sh_df['auths'] = pd.Series(auths) # authority score
    sh_df['coren'] = pd.Series(coren) # coreness
    assert not np.isnan(sh_df.values).any()
    # sh_df = sh_df.dropna(axis=1)
    sh_df = scale_data(sh_df)
    sh_df.to_csv('{0}/properties.csv'.format(path), index=False)

    ''' Probabilities '''
    # Conver DAG to tree, will be used for prediction
    tree = mst(n_hgenes, hterms, gene_by_go.copy(), go_by_go.copy())
    hg2g = np.zeros((len(hterms),len(hterms)))
    for hidx, ht in enumerate(hterms):
      parents = direct_pa(ht, hterms, tree)
      parents = [np.where(hterms == p)[0][0] for p in parents]
      hg2g[hidx, parents] = 1

    q = deque()
    q.append((0,0)) # parent, level
    lcn = list()
    lcpn = list()
    lcl, lastl, pterms, cterms = list(), 0, list(), list()
    prob_df = pd.DataFrame()

    while len(q) > 0:
      pos, l = q.popleft()
      children = np.nonzero(hg2g[:,pos])[0]

      if lastl != l: # lcl order of prediction
        lastl = l
        lcl.append("{0}= {1}".format(','.join(pterms), ','.join(cterms)))
        pterms, cterms = list(), list()
      pterms.append(terms[hterms[pos]])
      cterms += list(terms[hterms[children]])

      if len(children) > 0: # is a parent
        lcpn.append(("{0}= {1}".format(terms[hterms[pos]], ','.join(terms[hterms[children]])))) # save lcpn order of prediction

        for c in children:
          lcn.append(("{0}= {1}".format(terms[hterms[pos]], terms[hterms[c]]))) # save lcn order of prediction
          q.append((c,l+1))

          prb_feat = list() # neighborhood_information(n_hgenes, c, pos)
          for gid in n_hgenes:
            neighbors = np.nonzero(gcn[:,gid])[0]
            ntotal = len(neighbors)
            nassc = np.count_nonzero(gene_by_go[neighbors,c])
            prb_feat.append(nassc/ntotal) # proba. of being associated to class from neigh.

          prob_df["{0}_prb".format(terms[hterms[c]])] = prb_feat

    prob_df.to_csv('{0}/neighbors.csv'.format(path), index=False)
    list2file(lcn, "{0}/lcn.txt".format(path))
    list2file(lcpn, "{0}/lcpn.txt".format(path))
    list2file(lcl, "{0}/lcl.txt".format(path))


    print()
    print('###################')
    print('{0:2}. Root {1}'.format(i+1,term))
    print('Terms: {0:4}'.format(len(hterms)))
    print('Genes: {0:4} -> {1:4}'.format(len(hgenes), len(n_hgenes)))
    print("Labels --> Old: {0:6}\tNew: {1:6}\tTest: {2:6}".format(gbgO.sum(), gbgN.sum(), gbgT.sum()))
    print("New labels: {0:6} ({1:6})".format(gbgN[(gbgO == 0) & (gbgN == 1)].sum(), gbgN.sum() - gbgO.sum()))
    print()

    # break

sys.exit()

list2file(terms[root_list], "data/roots.txt")
test_df_subh = test_df_subh[test_df_subh.Root_idx.isin(root_list)].reset_index(drop=True)
test_df_subh.to_csv('data/hierarchies.csv', index=False)
print(str(test_df_subh))

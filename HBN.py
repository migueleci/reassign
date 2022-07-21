#!/usr/bin/env python
# coding: utf-8

# HBN method
# Miguel Romero, nov 3rd

import scipy.stats
import numpy as np
from tqdm import tqdm

# Finding the direct parent for a given term in a given heirarchy the direct parent are the
# group of terms that do not have any other descendant. Verified aug 25th, 2020
def direct_pa(go, terms, hie):
  go_id = np.nonzero(terms == go)[0]

  cand_pa_id = np.nonzero(hie[go_id,:])[1] # Find all candidate parent terms of term go
  sub_hie_idx = np.hstack((cand_pa_id, go_id))
  sub_hie = hie[np.ix_(sub_hie_idx, sub_hie_idx)]

  for i in range(len(cand_pa_id)):
    # If the number of descendants of candidate parent term i is greater than 1, it is NOT the
    # direct parent term. Discard all terms with more than 1 descendant
    if np.count_nonzero(sub_hie[:,i]) > 1:
      cand_pa_id[i] = -1 # Python index from 0, -1 means that is no the direct parent

  pa_id = cand_pa_id[cand_pa_id >= 0] # any value diff from -1 is related as a direct parent
  return terms[pa_id] # return id of parent(s) for term go


# Minimal Spanning Tree Algorithm on GO hierarchy. Verified may 24th
def mst(genes, terms, gene_by_go, go_by_go):
  hie = go_by_go[np.ix_(terms,terms)].copy()
  n = len(terms)

  # From the 2nd level (level directly below the root) and lower
  # discard terms with exactly one parent (thery are already a tree)
  leaf = terms[1:].copy()
  leaf[np.nonzero(hie.sum(axis=1) == 1)[0]-1] = -1
  leaf = leaf[leaf >= 0]

  if len(leaf) > 0: # If no 2nd level terms, keep the current hierarchy
    for i in range(len(leaf)):
      parents = direct_pa(leaf[i], terms, hie)
      if len(parents) > 1: # check if leaf has only one or multiple ancestors
        leaf_id = np.nonzero(terms == leaf[i])[0]
        p = np.zeros(len(parents))

        for j in range(len(parents)):
          a = gene_by_go[genes, leaf[i]]
          b = gene_by_go[genes, parents[j]]

          both, one = np.count_nonzero(a+b == 2), np.count_nonzero(a+b == 1)
          if both + one > 0:
            p[j] = both / (both + one)

        pa = parents[p == np.max(p)]
        if len(pa) > 1:
          pa = pa[np.random.randint(0, len(pa), 1)]
        pa_id = np.nonzero(terms == pa)[0]

        # Relabel the entries in hie
        hie[leaf_id,:] = 0
        ance_id1 = np.hstack((np.nonzero(hie[pa_id,:])[1], pa_id))
        hie[leaf_id, ance_id1] = 1
        ch_id = np.nonzero(hie[:, leaf_id])[0]
        if len(ch_id) > 0:
          hie[ch_id,:] = 0
          ance_id2 = np.hstack((np.nonzero(hie[leaf_id,:])[1], leaf_id))
          hie[np.ix_(ch_id, ance_id2)] = 1

  return hie


# Hierarchical Binomial-Neighborhood Algorithm on ONE. Verified may 24th
e = 10e-6

def hbn_prob_one(ch, pa, gene, train, ppi, gene_by_go):
  f = 0

  # gene's neighbors in the training set
  # neighbor = train[np.nonzero(ppi[train, gene])[0]]
  neighbor = np.intersect1d(train,np.nonzero(ppi[train, gene])[0])

  ch, pa = int(ch), int(pa)
  k_ch = np.count_nonzero(gene_by_go[neighbor, ch])
  k_pa = np.count_nonzero(gene_by_go[neighbor, pa])

  # If anybody in the training set is associated with parent GO term
  if np.count_nonzero(gene_by_go[train, pa]) > 0:
    # Estimate f1 = P(a protein has ch | it has pa ) by the training set
    f1 = float(np.count_nonzero(gene_by_go[train, ch]) / np.count_nonzero(gene_by_go[train, pa]))
    f2 = 1 - f1

    if f1 > 0:# < 1e-9
      # Estimate p1, p0 by MLE
      # p1 = P(a neighbor has ch and pa | gene has ch and pa)
      # p0 = P(a neighbor has ch and pa | gene DO NOT have ch but does have pa)
      A, B, C = 0, 0, 0

      # Delete genes NOT labeled with parent GO term. They are automatically NOT labeled with the child GO term
      genes_to_delete = np.nonzero(gene_by_go[train, pa] == 0)[0]

      # Recover the ppi with genes labeled with at least the parent GO term
      if len(genes_to_delete) > 0:
        train[genes_to_delete] = -1
        train = train[train >= 0]

      if len(train) == 0:
        return f
      else:
        ppi_tr = ppi[np.ix_(train, train)]

      if np.sum(ppi_tr) == 0:
        return f
      else:
        for i in range(len(train)):
          neigh_tr = np.nonzero(ppi_tr[i,:])[0]
          for j in range(len(neigh_tr)):
            val = gene_by_go[train[i], ch] + gene_by_go[train[neigh_tr[j]], ch]
            if val == 2:
              A += 1 # a ch-ch edge
              C += 1 # it is also a pa-pa edge
            elif val == 1:
              B += 1 # a ch-pa edge
            else:
              C += 1 # a pa-pa edge

      if A + B == 0 or B + C == 0:
        return f
      else:
        p1 = 2*A/(2*A + B)
        p0 = B/(2*C + B)

      a = np.log(scipy.stats.binom.cdf(k_ch, k_pa, p1) + e) + np.log(f1 + e)
      b = np.log(scipy.stats.binom.cdf(k_ch, k_pa, p0) + e) + np.log(f2 + e)
      f = np.exp(a)/(np.exp(a)+np.exp(b))

  return f


# Hierarchical Binomial-Neighborhood Algorithm on a set of genes and GO terms from a hierarchy
# OK, verified may 24th
def hbn_prob(genes, terms, train, ppi, gene_by_go, go_by_go):
  f = np.zeros((len(genes), len(terms-1)))

  root = terms[0]
  # Transform DAG into tree
  tree = mst(train, terms, gene_by_go, go_by_go)

  # precalculate ancestor list for each term
  ance_list = list()
  for j in range(1, len(terms)):
    ance = terms[np.nonzero(tree[j,:])[0]]
    _list = np.zeros(len(ance)+1)
    _list[0] = terms[j]
    for t in range(1, np.count_nonzero(tree[j,:])):
      # Find the direct parent terms for terms along the path
      _list[t] = direct_pa(_list[t-1], terms, tree)
    _list[len(ance)] = root
    ance_list.append(list(_list))

  for i in tqdm(range(len(genes))):
    # memorizing probabilities calculated
    mem = dict()

    for j in range(1, len(terms)):
      _list = ance_list[j-1]

      # compute probabilities for term and its ancestors
      p = np.zeros(len(_list)-1)
      for t in range(len(_list)-1):
        # p[t] = hbn_prob_one(_list[t], _list[t+1], genes[i], train.copy(), ppi, gene_by_go)
        # probability already computed and stored
        if (_list[t], _list[t+1]) in mem:
          p[t] = mem[(_list[t], _list[t+1])]
        else:
          p[t] = hbn_prob_one(_list[t], _list[t+1], genes[i], train.copy(), ppi, gene_by_go)
          mem[(_list[t], _list[t+1])] = p[t]

      # Compute P(gene i has term j) = \prod P(gene i has )
      f[i, j-1] = p.prod()

  return f

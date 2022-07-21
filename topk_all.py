import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm, tnrange
from time import time
from scipy import stats
from collections import deque

import multiprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from evaluate import *

np.seterr(divide='ignore', invalid='ignore')

# Ploting
from matplotlib import rc
import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

rc('font', family='serif', size=18)
rc('text', usetex=True)

# Default colors
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


# read a txt file containing a list, one item per line
def readListFile(filename, cls=str):
  file = open(filename, 'r')
  tmp = [cls(x.strip()) for x in file.readlines()]
  file.close()
  return np.array(tmp)


# pretty print of results for a model
def pprint(model, pooled, macro, macrow, others=None):
    if others is not None:
      others = "\t\t" + "\t\t".join(["{0}:\t{1}".format(k,others[k]) for k in others.keys()])
    print("## {0}:\tpooled:\t{1:.4f}\t\tmacro:\t{2:.4f}\t\tmacrow:\t{3:.4f}{4}".format(
      model,pooled, macro, macrow, "" if others is None else others)
    )


# plot line
def line_plot(data,xticks,labels, markers,xlabel,ylabel,title=None,fname=None,ylim=None,multipdf=None):
  fig, ax = plt.subplots(figsize=(10,10))
  x = np.arange(len(xticks))
  for idx, (y, l, m) in enumerate(zip(data,labels, markers)):
    plt.plot(x, y, m, color=_COLORS[idx], label=l)
  if ylim is not None:
    plt.ylim(ylim)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  if title is not None:
    plt.title(title)
  plt.xticks(x, xticks, rotation=90)
  plt.grid(axis='both', linestyle='--')
  if len(labels) > 1:
    plt.legend(loc='best')
  plt.tight_layout()
  if multipdf is None:
    plt.savefig(fname, format='pdf', dpi=600)
  else:
    multipdf.savefig(dpi=600)
  plt.close()


# %%

root_list = readListFile("data/roots.txt")
seed = 220224 # 220319
N = 10 # cv kfold
n_estimators = 100 # random forest
np.random.seed(seed) # set numpy seed
LEVEL_THR = [0.5*(0.75**i) for i in range(15)]
K = np.arange(0.01,0.21,0.01)

plot_labels = ['K paths (avg)', 'K paths (sum)', 'K paths (min)','Top X','Random X', 'Top X noise (baseline)']
plot_markers = ['--o']*len(plot_labels)

for j, root in enumerate(root_list):
  results = pd.DataFrame(columns=["Root","k","K","X",'kpme','nkpme','kpsu','nkpsu','kpmi','nkpmi','topx','ntopx','ranx','nranx','base','nbase'])

  kpme = list() # top k paths using mean probability
  kpsu = list() # top k paths using sum of probabilities
  kpmi = list() # top k paths using min probability
  topx = list() # top x probabilities
  ranx = list() # random x probabilities
  base = list() # baseline, top x noise points

  if j>0: print()
  print("{0:2}. Root: {1}".format(j+1, root))
  print("{0}------------------".format('--' if j>8 else '-'))

  """ Read sub-hierarchy data:
   - node embeddings of GCN
   - structural properties of GCN
   - probabilities (ratio) of gene neighbohrs associated to each function
   - old, new and test labels. Old and new lables share instances, but not test
   - lcn order, info per function: ancestors and children (map of parents)
  """
  path = "data/{0}".format(root.replace(':',''))
  labels = readListFile("{0}/terms.txt".format(path))
  labels_idx = dict([(x,i) for i,x in enumerate(labels)])
  genes_co = readListFile("{0}/genes_co.txt".format(path), int) # Genes for co-training/pu-learning, genes in old and new data

  data = pd.read_csv("{0}/embedding.csv".format(path)) # node embeddings of the GCN subgraph for subhierarchy
  props = pd.read_csv("{0}/properties.csv".format(path)) # structural properties of the GCN subgraph for subhierarchy
  probs = pd.read_csv("{0}/neighbors.csv".format(path)) # ratio of gene neighbors asscoeiated to each function
  data[props.columns] = props
  data[probs.columns] = probs # ONLY FOR GLOBAL MODEL
  data = data.loc[genes_co].copy().reset_index(drop=True) # data for co-training/pu-learning

  labels_old = pd.read_csv("{0}/labels_old.csv".format(path), dtype='int32') # labels in old data
  labels_new = pd.read_csv("{0}/labels_new.csv".format(path), dtype='int32') # labels in new data (for same genes and functions)
  new_ones = labels_new.values[(labels_new.values == 1) & (labels_old.values == 0)].sum()

  parent_map = open("{0}/lcn.txt".format(path), 'r') # load parent of each function, used for cumulative probs.
  parent_map = dict([(l.split('=')[1].strip(),l.split('=')[0].strip()) for l in parent_map.readlines()])
  # child_map = open("{0}/lcpn.txt".format(path), 'r') # load parent of each function, used for cumulative probs.
  # child_map = dict([(l.split('=')[0].strip(),l.split('=')[1].strip().split(',')) for l in child_map.readlines()])
  leaves = [ch for ch in parent_map.keys() if ch not in parent_map.values()]
  leaves_idx = np.array([np.where(labels == ch)[0][0] for ch in leaves])

  lcl = open("{0}/lcl.txt".format(path), 'r') # load lcl order
  lcl = [x.strip().split('=')[1].strip() for x in lcl.readlines()]
  levels = list()
  for funct in labels:
    for ilv, lv in enumerate(lcl):
      if funct in lv:
        levels.append((funct, ilv+1))
  assert len(levels) == len(labels)
  levels = dict(levels)

  # load random forest classifier and evaluation class
  rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=5, oob_score=True, n_jobs=-1, random_state=seed)
  dt = DecisionTreeClassifier(min_samples_split=5, max_features="auto", random_state=seed)
  e = Evaluate()



  """ 1. Top K paths from leaves with only 0s in the old data """
  kfold = KFold(n_splits=N, shuffle=True, random_state=seed) # set the same folds for each model
  pred = np.zeros(labels_old.shape)
  # for train_index, test_index in tqdm(kfold.split(data), total=N, ascii="-#", desc="1. Top K paths"): # set the same folds for each model
  for train_index, test_index in tqdm(kfold.split(data), total=N, ascii="-#", desc="1. Random forest"): # set the same folds for each model

    """ create y and x dataset for the current hierarchy """
    y_train = labels_old.loc[train_index] # train-test split of y
    X_train, X_test = data.loc[train_index], data.loc[test_index] # train-test split of data

    """ training and prediction for whole hierarchy """
    rf.fit(X_train, y_train)
    _pred = rf.predict_proba(X_test)
    for cidx, x, cls in zip(range(labels_old.shape[1]), _pred, rf.classes_):
      pred[test_index,cidx] = 1 - x[:,0].copy() if cls[0] == 0 else x[:,0].copy()

  ''' compute average probability per gene and path containing only 0s in the old data '''
  paths_mean = list()
  paths_sum = list()
  paths_min = list()
  for leaf, leaf_idx in zip(leaves, leaves_idx):
    z_genes = np.where(labels_old.values[:,leaf_idx] == 0)[0] # genes that are not associated to leaf
    for gidx in z_genes:
      assert labels_old.values[gidx, leaf_idx] == 0

      path = [[],[]] # index of functions in path and probabilities
      cfi, cft = leaf_idx, leaf # gene and function index

      while cft != root and labels_old.values[gidx, cfi] == 0:
        path[0].append(cfi)
        path[1].append(pred[gidx,cfi])
        cft = parent_map[cft]
        if cft != root: cfi = labels_idx[cft]

      paths_mean.append((gidx,leaf_idx,np.mean(path[1]),len(path[0]),path[0],path[1]))
      paths_sum.append((gidx,leaf_idx,np.sum(path[1]),len(path[0]),path[0],path[1]))
      paths_min.append((gidx,leaf_idx,np.min(path[1]),len(path[0]),path[0],path[1]))

  ''' Top K paths mean '''
  paths_mean = pd.DataFrame(paths_mean, columns=["gene","leaf","mean prb","len","path", "prb"])
  paths_mean = paths_mean.sort_values(by=["mean prb"], ascending=False).reset_index(drop=True)

  ''' Top K paths sum '''
  paths_sum = pd.DataFrame(paths_sum, columns=["gene","leaf","sum prb","len","path", "prb"])
  paths_sum = paths_sum.sort_values(by=["sum prb"], ascending=False).reset_index(drop=True)

  ''' Top K paths min '''
  paths_min = pd.DataFrame(paths_min, columns=["gene","leaf","min prb","len","path", "prb"])
  paths_min = paths_min.sort_values(by=["min prb"], ascending=False).reset_index(drop=True)



  """ 2. Top X out of all predictions that are 0 in old data """
  pred_df = [[],[],[]]
  for cidx in range(pred.shape[1]):
    pred_df[0] += labels_old.index.tolist()
    pred_df[1] += [cidx for _ in range(labels_old.shape[0])]
    pred_df[2] += pred[:,cidx].tolist()

  pred_df = pd.DataFrame(np.array(pred_df).T, columns=["row","col","prb"])
  pred_df = pred_df.astype({'row': 'int32', 'col': 'int32'})
  pred_df['old'] = [labels_old.values[a,b] for a,b in zip(pred_df.row.tolist(), pred_df.col.tolist())]
  pred_df['new'] = [labels_new.values[a,b] for a,b in zip(pred_df.row.tolist(), pred_df.col.tolist())]
  pred_df = pred_df[pred_df.old == 0]
  pred_df = pred_df.sort_values(by=["prb"], ascending=False).reset_index(drop=True)



  ''' 4. Baseline, Top X noise labels '''
  # print("Opimizing sampling rate ...")
  sampling_rate = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
  sampling_rate_error = list()
  for m in tqdm(sampling_rate, total=len(sampling_rate), ascii="-#", desc="2. Baseline (sampling rate)"):
    errors = np.zeros(n_estimators)
    size = int(labels_old.shape[0]*m)
    for i in range(n_estimators):
      bag = np.random.choice(np.arange(labels_old.shape[0]),size=size, replace=True)
      oobag = np.array([v for v in np.arange(labels_old.shape[0]) if v not in bag])

      y_train, y_test = labels_old.loc[bag], labels_old.loc[oobag] # train-test split of y
      X_train, X_test = data.loc[bag], data.loc[oobag] # train-test split of data

      dt.fit(X_train, y_train)
      _pred = dt.predict_proba(X_test)
      preds = np.zeros(y_test.shape)
      for cidx, x, cls in zip(range(labels_old.shape[1]), _pred, dt.classes_):
        preds[:,cidx] = 1 - x[:,0].copy() if cls[0] == 0 else x[:,0].copy()

      errors[i] = e.multiclass_classification_measures(preds, y_test)[3]

    sampling_rate_error.append(errors.mean())

  m = sampling_rate[np.argmax(sampling_rate_error)] # sampling error optimizaiton

  bags = list()
  preds = [np.zeros(labels_old.shape) for _ in range(n_estimators)]
  size = int(labels_old.shape[0]*m)
  for i in range(n_estimators):
    bag = np.random.choice(np.arange(labels_old.shape[0]),size=size, replace=True)
    oobag = np.array([v for v in np.arange(labels_old.shape[0]) if v not in bag])
    bags.append(oobag)

    y_train = labels_old.loc[bag] # train-test split of y
    X_train, X_test = data.loc[bag], data.loc[oobag] # train-test split of data

    dt.fit(X_train, y_train)
    _pred = dt.predict_proba(X_test)
    for cidx, x, cls in zip(range(labels_old.shape[1]), _pred, dt.classes_):
      preds[i][oobag,cidx] = 1 - x[:,0].copy() if cls[0] == 0 else x[:,0].copy()

  noise_rate = np.zeros(labels_old.shape) # precompute the noise rate for each gene and function
  # for gene in tqdm(range(labels_old.shape[0]), total=labels_old.shape[0], ascii="-#", desc="4. Noise X (baseline)"):
  for gene in tqdm(range(labels_old.shape[0]), total=labels_old.shape[0], ascii="-#", desc="2. Baseline (disagreement)"):
    for funct in range(labels_old.shape[1]):
      rate = list()
      for tree in range(n_estimators):
        if gene in bags[tree]: # instance out of bag
          thr = LEVEL_THR[levels[labels[funct]]-1]
          label = int(preds[tree][gene,funct] >= thr) # label assigned to prediction in tree
          rate.append(int(label != labels_old.values[gene,funct])) # count if miscalssified
      noise_rate[gene, funct] = np.mean(rate) # if len(rate) > 0 else 0


  for mm, kk in tqdm(enumerate(K), total=len(K), ascii="-#", desc="Iterating K"):

    PATH_THR = int(np.count_nonzero(labels_old) * kk)

    ''' Top K paths mean '''
    _paths_mean = paths_mean.head(PATH_THR)

    topk_mean = list()
    for row in _paths_mean.to_dict('records'):
      for fidx, prb in zip(row["path"], row["prb"]):
        topk_mean.append((row["gene"], fidx, prb, row["mean prb"], labels_old.values[row["gene"], fidx], labels_new.values[row["gene"], fidx]))

    topk_mean = pd.DataFrame(topk_mean, columns=["row","col","prb","path prb","old","new"])
    assert len(topk_mean) == _paths_mean["len"].sum()
    assert topk_mean["old"].sum() == 0
    THRESHOLD = _paths_mean["len"].sum() # threshold is set using k paths with mean

    ''' Top K paths sum '''
    _paths_sum = paths_sum.head(PATH_THR*3)

    topk_sum = list()
    for row in _paths_sum.to_dict('records'):
      for fidx, prb in zip(row["path"], row["prb"]):
        topk_sum.append((row["gene"], fidx, prb, row["sum prb"], labels_old.values[row["gene"], fidx], labels_new.values[row["gene"], fidx]))

    topk_sum = pd.DataFrame(topk_sum, columns=["row","col","prb","path prb","old","new"])
    topk_sum = topk_sum.head(THRESHOLD)
    assert len(topk_sum) == THRESHOLD
    assert topk_sum["old"].sum() == 0

    ''' Top K paths min '''
    _paths_min = paths_min.head(PATH_THR*3)

    topk_min = list()
    for row in _paths_min.to_dict('records'):
      for fidx, prb in zip(row["path"], row["prb"]):
        topk_min.append((row["gene"], fidx, prb, row["min prb"], labels_old.values[row["gene"], fidx], labels_new.values[row["gene"], fidx]))

    topk_min = pd.DataFrame(topk_min, columns=["row","col","prb","path prb","old","new"])
    topk_min = topk_min.head(THRESHOLD)
    assert len(topk_min) == THRESHOLD
    assert topk_min["old"].sum() == 0



    """ 2. Top X out of all predictions that are 0 in old data """
    topx_rf = pred_df.head(THRESHOLD)
    assert topx_rf.shape[0] == THRESHOLD



    """ 3. Random X """
    topx_rand = list()
    # for ridx in tqdm(range(N), total=N, ascii="-#", desc="3. Random X"): # set the same folds for each model
    for ridx in range(N):

      candidates = np.where(labels_old == 0)
      candidates = np.array([candidates[0], candidates[1]]).T
      rand_df = pd.DataFrame(candidates, columns=["row","col"])
      rand_df = rand_df.astype({'row': 'int32', 'col': 'int32'})
      rand_df = rand_df.sample(frac=1).reset_index(drop=True)
      rand_df['new'] = [labels_new.values[a,b] for a,b in candidates]
      candidates = rand_df.to_dict('records')

      _topx_rand = list()
      cidx = 0
      flag = np.zeros(labels_old.shape)
      while len(_topx_rand) < THRESHOLD:
        xi, yi = candidates[cidx]["row"], candidates[cidx]["col"]
        cy, ct = yi, labels[yi]
        while ct != root and labels_old.values[xi, cy] == 0 and flag[xi,cy] == 0:
          _topx_rand.append((xi,cy,labels_new.values[xi, cy]))
          flag[xi,cy] = 1
          ct = parent_map[ct]
          if ct != root:
            cy = np.where(labels == ct)[0][0]

        cidx += 1

      if len(_topx_rand) > THRESHOLD:
        xi, yi = candidates[cidx-1]["row"], candidates[cidx-1]["col"]
        cy, ct = yi, labels[yi]
        while ct != root and flag[xi,cy] == 1 and len(_topx_rand) > THRESHOLD:
          _topx_rand.remove((xi,cy,labels_new.values[xi, cy]))
          ct = parent_map[ct]
          if ct != root:
            cy = np.where(labels == ct)[0][0]

      _topx_rand = pd.DataFrame(_topx_rand, columns=["row","col","new"])
      assert _topx_rand.shape[0] == THRESHOLD
      topx_rand.append((_topx_rand.new.sum(), _topx_rand.new.sum()/THRESHOLD))

    topx_rand = np.array(topx_rand)



    ''' 4. Baseline, Top X noise labels '''
    disagr_rate = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    disagr_rate_error = list()
    disagr_rate_errorn = list()
    for theta in disagr_rate:
      noise = np.where(noise_rate >= theta)
      noise = pd.DataFrame(np.array(noise).T,columns=["r","c"])
      noise["err"] = [noise_rate[a,b] for a,b in zip(noise.r.tolist(), noise.c.tolist())]
      noise['old'] = [labels_old.values[a,b] for a,b in zip(noise.r.tolist(), noise.c.tolist())]
      noise['new'] = [labels_new.values[a,b] for a,b in zip(noise.r.tolist(), noise.c.tolist())]
      noise = noise[noise.old == 0]
      noise = noise.sort_values(by=["err"], ascending=False).reset_index(drop=True)

      topxn = noise.head(THRESHOLD)
      disagr_rate_error.append(topxn.new.sum()/THRESHOLD)
      disagr_rate_errorn.append(topxn.new.sum())

    theta = disagr_rate[np.argmax(disagr_rate_error)] # sampling error optimizaiton
    # topx_noise = disagr_rate_error[np.argmax(disagr_rate_error)]
    topx_noise = disagr_rate_errorn[np.argmax(disagr_rate_error)]



    kpme.append(topk_mean.new.sum()/THRESHOLD)
    kpsu.append(topk_sum.new.sum()/THRESHOLD)
    kpmi.append(topk_min.new.sum()/THRESHOLD)
    topx.append(topx_rf.new.sum()/THRESHOLD)
    ranx.append(topx_rand[:,1].mean())
    base.append(topx_noise/THRESHOLD)

    results.loc[mm] = [root,
                      kk,
                      PATH_THR,
                      THRESHOLD,
                      topk_mean.new.sum()/THRESHOLD,
                      topk_mean.new.sum(),
                      topk_sum.new.sum()/THRESHOLD,
                      topk_sum.new.sum(),
                      topk_min.new.sum()/THRESHOLD,
                      topk_min.new.sum(),
                      topx_rf.new.sum()/THRESHOLD,
                      topx_rf.new.sum(),
                      topx_rand[:,1].mean(),
                      topx_rand[:,0].mean(),
                      topx_noise/THRESHOLD,
                      topx_noise,
                     ]
    results.to_csv('kpred/{0}.csv'.format(root.replace(":","")), index=False)

    title = "Precision@K {0}".format(root)
    xticks = ["{0:.2f}".format(kkk) for kkk in K[:mm+1]]
    pdata = [kpme[:mm+1],kpsu[:mm+1],kpmi[:mm+1],topx[:mm+1],ranx[:mm+1],base[:mm+1]]
    line_plot(pdata,xticks,plot_labels,plot_markers,'K','Presicion',title=title,ylim=[0,0.3],fname='kpred/figs/{0}.pdf'.format(root.replace(":","")))

  break

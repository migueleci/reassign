import os
import numpy as np
import pandas as pd

from tqdm import tqdm, tnrange
from time import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from matplotlib import rc
from matplotlib import pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

rc('font', family='serif', size=22)
rc('text', usetex=False)

# Default colors
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# create directory
def makedir(path):
  try:
    os.makedirs(path, exist_ok=True)
  except OSError as error:
    print("Directory '%s' can not be created")


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
K = np.arange(0.01,0.21,0.01)

models = ['kpme', 'kpsum', 'kpmin']
names = [r'REASSIGN (avg)', r'REASSIGN (sum)', r'REASSIGN (min)']
plot_markers = ['--s','--P','--v']

for j, root in enumerate(root_list):
  results = pd.DataFrame(columns=["Root","k","K","X",'kpme','nkpme','kpsum','nkpsum','kpmin','nkpmin'])
  makedir('pred/{0}'.format(root.replace(':','')))

  kpme = list() # top k paths using mean probability
  kpsum = list() # top k paths using sum of probability
  kpmin = list() # top k paths using minimum probability

  if j>0: print()
  print("{0:2}. Root: {1}".format(j+1, root))
  print("{0}------------------".format('--' if j>8 else '-'))

  """ Read sub-hierarchy data:
  """
  path = "data/{0}".format(root.replace(':',''))
  labels = readListFile("{0}/terms.txt".format(path))
  labels_idx = dict([(x,i) for i,x in enumerate(labels)])
  genes = readListFile("{0}/genes.txt".format(path))
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

  kfold = KFold(n_splits=N, shuffle=True, random_state=seed) # set the same folds for each model
  """ 1. Top K paths from leaves with only 0s in the old data """
  pred = np.zeros(labels_old.shape)
  # for train_index, test_index in tqdm(kfold.split(data), total=N, ascii="-#", desc="1. Top K paths"): # set the same folds for each model
  for train_index, test_index in tqdm(kfold.split(data), total=N, ascii="-#", desc="Random forest CV"): # set the same folds for each model

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


  for mm, kk in tqdm(enumerate(K), total=len(K), ascii="-#", desc="Iterating K"):

    PATH_THR = int(np.count_nonzero(labels_old) * kk)

    ''' Top K paths mean '''
    _paths_mean = paths_mean.head(PATH_THR)

    topk_mean = list()
    for row in _paths_mean.to_dict('records'):
      for fidx, prb in zip(row["path"], row["prb"]):
        topk_mean.append((row["gene"], fidx, prb, row["mean prb"], labels_old.values[row["gene"], fidx], labels_new.values[row["gene"], fidx]))

    topk_mean = pd.DataFrame(topk_mean, columns=["row","col","prb","path prb","old","New"])
    topk_mean = topk_mean.drop_duplicates(subset=["row","col"], keep='first')
    assert topk_mean["old"].sum() == 0
    THRESHOLD = len(topk_mean) # threshold is set using k paths with mean
    if mm == len(K) - 1:
      topk_mean.row = genes[topk_mean.row.values]
      topk_mean.col = labels[topk_mean.col.values]
      topk_mean.columns = ['Gene','Term','Probability','Path probability','Old','New']
      topk_mean = topk_mean[['Gene','Term','Probability','Path probability','New']]
      topk_mean.to_csv('pred/{0}/top_mean.csv'.format(root.replace(':','')),index=False)

    ''' Top K paths sum '''
    _paths_sum = paths_sum.head(PATH_THR*3)

    topk_sum = list()
    for row in _paths_sum.to_dict('records'):
      for fidx, prb in zip(row["path"], row["prb"]):
        topk_sum.append((row["gene"], fidx, prb, row["sum prb"], labels_old.values[row["gene"], fidx], labels_new.values[row["gene"], fidx]))

    topk_sum = pd.DataFrame(topk_sum, columns=["row","col","prb","path prb","old","New"])
    topk_sum = topk_sum.drop_duplicates(subset=["row","col"], keep='first')
    topk_sum = topk_sum.head(THRESHOLD)
    assert len(topk_sum) == THRESHOLD
    assert topk_sum["old"].sum() == 0
    if mm == len(K) - 1:
      topk_sum.row = genes[topk_sum.row.values]
      topk_sum.col = labels[topk_sum.col.values]
      topk_sum.columns = ['Gene','Term','Probability','Path probability','Old','New']
      topk_sum = topk_sum[['Gene','Term','Probability','Path probability','New']]
      topk_sum.to_csv('pred/{0}/top_sum.csv'.format(root.replace(':','')),index=False)

    ''' Top K paths min '''
    _paths_min = paths_min.head(PATH_THR*3)

    topk_min = list()
    for row in _paths_min.to_dict('records'):
      for fidx, prb in zip(row["path"], row["prb"]):
        topk_min.append((row["gene"], fidx, prb, row["min prb"], labels_old.values[row["gene"], fidx], labels_new.values[row["gene"], fidx]))

    topk_min = pd.DataFrame(topk_min, columns=["row","col","prb","path prb","old","New"])
    topk_min = topk_min.drop_duplicates(subset=["row","col"], keep='first')
    topk_min = topk_min.head(THRESHOLD)
    assert len(topk_min) == THRESHOLD
    assert topk_min["old"].sum() == 0
    if mm == len(K) - 1:
      topk_min.row = genes[topk_min.row.values]
      topk_min.col = labels[topk_min.col.values]
      topk_min.columns = ['Gene','Term','Probability','Path probability','Old','New']
      topk_min = topk_min[['Gene','Term','Probability','Path probability','New']]
      topk_min.to_csv('pred/{0}/top_min.csv'.format(root.replace(':','')),index=False)


    kpme.append(topk_mean.New.sum()/THRESHOLD)
    kpsum.append(topk_sum.New.sum()/THRESHOLD)
    kpmin.append(topk_min.New.sum()/THRESHOLD)

    results.loc[mm] = [root,
                      kk,
                      PATH_THR,
                      THRESHOLD,
                      topk_mean.New.sum()/THRESHOLD,
                      topk_mean.New.sum(),
                      topk_sum.New.sum()/THRESHOLD,
                      topk_sum.New.sum(),
                      topk_min.New.sum()/THRESHOLD,
                      topk_min.New.sum(),
                     ]
    results.to_csv('pred/{0}/precision.csv'.format(root.replace(":","")), index=False)

  title = "Precision@K {0}".format(root)
  xticks = ["{0:.2f}".format(kkk) for kkk in K]
  pdata = [results[met].tolist() for met in models]
  ylim = np.max(pdata)*1.05
  line_plot(pdata,xticks,names,plot_markers,'K','Presicion',title=title,ylim=[0,ylim],fname='pred/{0}/precision.pdf'.format(root.replace(":","")))

  # break

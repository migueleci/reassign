import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm, tnrange
from time import time
from scipy import stats
from collections import deque

import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

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

COLORSCSS = ['black', 'dimgray', 'dimgrey', 'gray', 'grey', 'darkgray',
             'darkgrey', 'rosybrown', 'lightcoral', 'indianred', 'brown',
             'firebrick', 'maroon', 'darkred', 'red', 'salmon', 'tomato',
             'darksalmon', 'coral', 'orangered', 'sienna', 'chocolate',
             'saddlebrown', 'sandybrown', 'peru', 'darkorange', 'burlywood',
             'orange', 'darkgoldenrod', 'goldenrod', 'gold', 'olive', 'yellow',
             'olivedrab', 'yellowgreen', 'darkolivegreen', 'greenyellow',
             'chartreuse', 'lawngreen', 'darkseagreen', 'lightgreen',
             'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime',
             'seagreen', 'mediumseagreen', 'springgreen', 'mediumspringgreen',
             'mediumaquamarine', 'aquamarine', 'turquoise', 'lightseagreen',
             'mediumturquoise', 'darkslategray', 'darkslategrey', 'teal',
             'darkcyan', 'aqua', 'cyan', 'darkturquoise', 'cadetblue',
             'deepskyblue', 'skyblue', 'steelblue', 'dodgerblue',
             'lightslategray', 'lightslategrey', 'slategray', 'slategrey',
             'lightsteelblue', 'cornflowerblue', 'royalblue', 'midnightblue',
             'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue',
             'darkslateblue', 'mediumslateblue', 'mediumpurple',
             'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid',
             'darkviolet', 'mediumorchid', 'violet', 'purple', 'darkmagenta',
             'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink',
             'hotpink', 'palevioletred', 'crimson', 'pink']

COLORSCSS = np.random.choice(COLORSCSS, 15, replace=False)


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
results = pd.read_csv("pred/results_paths_k5_b_th.csv")
results = results[['Root', 'K', 'X', 'K1', 'K2', 'K3', 'bsdt']]
results.columns = ['Root','K','X','Kpaths','TopX','RandomX','Baseline']
K = results.K.tolist()
X = results.X.tolist()

seed = 220224
N = 10
n_estimators = 100
np.random.seed(seed)
K1, K2 = [np.nan for _ in root_list], [np.nan for _ in root_list]

plot_labels = ['K paths (avg)','Top X','Random X', 'Baseline', 'K paths (sum)', 'K paths (min)']
plot_markers = ['--o']*len(plot_labels)

for j, root in enumerate(root_list):
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
  child_map = open("{0}/lcpn.txt".format(path), 'r') # load parent of each function, used for cumulative probs.
  child_map = dict([(l.split('=')[0].strip(),l.split('=')[1].strip().split(',')) for l in child_map.readlines()])
  leaves = [ch for ch in parent_map.keys() if ch not in parent_map.values()]
  leaves_idx = np.array([np.where(labels == ch)[0][0] for ch in leaves])

  # lcl = open("{0}/lcl.txt".format(path), 'r') # load lcl order
  # lcl = [x.strip().split('=')[1].strip() for x in lcl.readlines()]
  # gcn = nx.read_edgelist("{0}/gcn.edg".format(path), delimiter='\t', nodetype=int)

  # load random forest classifier and evaluation class
  rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=5, oob_score=True, n_jobs=-1, random_state=seed)
  e = Evaluate()

  PATH_THR = K[j]

  """ 1. Top K paths from leaves with only 0s in the old data """
  kfold = KFold(n_splits=N, shuffle=True, random_state=seed) # set the same folds for each model
  pred = np.zeros(labels_old.shape)
  for train_index, test_index in tqdm(kfold.split(data), total=N, ascii="-#", desc="1. Top K paths"): # set the same folds for each model

    """ create y and x dataset for the current hierarchy """
    y_train, y_test = labels_old.loc[train_index], labels_old.loc[test_index] # train-test split of y
    X_train, X_test = data.loc[train_index], data.loc[test_index] # train-test split of data

    """ training and prediction for whole hierarchy """
    rf.fit(X_train, y_train)
    _pred = rf.predict_proba(X_test)
    for cidx, x, cls in zip(range(labels_old.shape[1]), _pred, rf.classes_):
      pred[test_index,cidx] = 1 - x[:,0].copy() if cls[0] == 0 else x[:,0].copy()

  ''' compute average probability per gene and path containing only 0s in the old data '''
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

      paths_sum.append((gidx,leaf_idx,np.sum(path[1]),len(path[0]),path[0],path[1]))
      paths_min.append((gidx,leaf_idx,np.min(path[1]),len(path[0]),path[0],path[1]))

  paths = pd.DataFrame(paths_sum, columns=["gene","leaf","mean prb","path len","path", "prb"])
  paths = paths.sort_values(by=["mean prb"], ascending=False).reset_index(drop=True)
  paths = paths.head(PATH_THR*3)

  topk_1 = list()
  for row in paths.to_dict('records'):
    for fidx, prb in zip(row["path"], row["prb"]):
      topk_1.append((row["gene"], fidx, prb, row["mean prb"], labels_old.values[row["gene"], fidx], labels_new.values[row["gene"], fidx]))

  topk_1 = pd.DataFrame(topk_1, columns=["row","col","prb","path prb","old","new"])
  topk_1 = topk_1.head(X[j])
  assert len(topk_1) == X[j]
  assert topk_1["old"].sum() == 0

  paths = pd.DataFrame(paths_min, columns=["gene","leaf","mean prb","path len","path", "prb"])
  paths = paths.sort_values(by=["mean prb"], ascending=False).reset_index(drop=True)
  paths = paths.head(PATH_THR*3)

  topk_2 = list()
  for row in paths.to_dict('records'):
    for fidx, prb in zip(row["path"], row["prb"]):
      topk_2.append((row["gene"], fidx, prb, row["mean prb"], labels_old.values[row["gene"], fidx], labels_new.values[row["gene"], fidx]))

  topk_2 = pd.DataFrame(topk_2, columns=["row","col","prb","path prb","old","new"])
  topk_2 = topk_2.head(X[j])
  assert len(topk_2) == X[j]
  assert topk_2["old"].sum() == 0

  K1[j] = (topk_1.new.sum()/X[j])
  K2[j] = (topk_2.new.sum()/X[j])

  results['Kpathssum'] = K1
  results['Kpathsmin'] = K2
  results.to_csv('pred/results_paths_k5_b_th_n.csv', index=False)

  xticks = labels[:j+1]
  presults = results.head(j+1)
  pdata = [presults.Kpaths,presults.TopX,presults.RandomX,presults.Baseline,presults.Kpathssum,presults.Kpathsmin]
  line_plot(pdata,xticks,plot_labels,plot_markers,'Sub-hierarchy (dataset)','Presicion',ylim=[0,0.3],fname='pred/figs/topkp_k5_b_th_n.pdf')

  # break

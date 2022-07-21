import numpy as np
import pandas as pd
import networkx as nx

from tqdm import tqdm, tnrange
from time import time
from scipy import stats
from collections import deque

import multiprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
results = pd.DataFrame(columns=["Root","Models","time",
                                "micro","macro","macrow"])
results_pl = pd.DataFrame(columns=["Root","Level",
                                   "micro","macro","macrow"])
seed = 220316
n_estimators = 100
np.random.seed(seed)
THRS = [0.5*(0.75**i) for i in range(15)]

results = pd.read_csv("pred/results_paths_k20_b.csv")
prec = [np.nan for _ in root_list]
X = results.X.tolist()
assert len(X) == len(root_list)

plot_labels = ['K paths','Top X','Random X', 'Baseline', 'Bs dt']
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

  # print("Opimizing sampling rate ...")
  sampling_rate = [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
  sampling_rate_error = list()
  for m in sampling_rate:
    errors = np.zeros(n_estimators)
    size = int(labels_old.shape[0]*m)
    for i in tqdm(range(n_estimators), total=n_estimators, ascii="-#", desc="m={0:.2f}".format(m)):
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

  print("\nPrecompute the noise rate ...")
  noise_rate = np.zeros(labels_old.shape) # precompute the noise rate for each gene and function
  for gene in tqdm(range(labels_old.shape[0]), total=labels_old.shape[0], ascii="-#", desc="Genes (instances)"):
    for funct in range(labels_old.shape[1]):
      rate = list()
      for tree in range(n_estimators):
        if gene in bags[tree]: # instance out of bag
          thr = THRS[levels[labels[funct]]-1]
          label = int(preds[tree][gene,funct] >= thr) # label assigned to prediction in tree
          # label = int(preds[tree][gene,funct] >= 0.5) # label assigned to prediction in tree
          rate.append(int(label != labels_old.values[gene,funct])) # count if miscalssified
      noise_rate[gene, funct] = np.mean(rate) # if len(rate) > 0 else 0


  print("\nOpimizing disagreement rate ...")
  disagr_rate = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
  disagr_rate_error = list()
  for theta in tqdm(disagr_rate, total=len(disagr_rate), ascii="-#"):
    noise = np.where(noise_rate >= theta)
    noise = pd.DataFrame(np.array(noise).T,columns=["r","c"])
    noise["err"] = [noise_rate[a,b] for a,b in zip(noise.r.tolist(), noise.c.tolist())]
    noise['old'] = [labels_old.values[a,b] for a,b in zip(noise.r.tolist(), noise.c.tolist())]
    noise['new'] = [labels_new.values[a,b] for a,b in zip(noise.r.tolist(), noise.c.tolist())]
    noise = noise[noise.old == 0]
    noise = noise.sort_values(by=["err"], ascending=False).reset_index(drop=True)

    topx = noise.head(X[j])
    disagr_rate_error.append(topx.new.sum()/X[j])

  theta = disagr_rate[np.argmax(disagr_rate_error)] # sampling error optimizaiton
  print(m, theta, disagr_rate_error[np.argmax(disagr_rate_error)])
  prec[j] = disagr_rate_error[np.argmax(disagr_rate_error)]

  results['bsdt'] = prec
  results.to_csv('pred/results_paths_k20_bth.csv', index=False)

  xticks = labels[:j+1]
  presults = results.head(j+1)
  line_plot([presults.K1,presults.K2,presults.K3,presults.baseline,presults.bsdt],xticks,plot_labels,plot_markers,'Sub-hierarchy (dataset)','Presicion',ylim=[0,0.3],fname='pred/figs/results_paths_k20_b_th.pdf')

  # # break

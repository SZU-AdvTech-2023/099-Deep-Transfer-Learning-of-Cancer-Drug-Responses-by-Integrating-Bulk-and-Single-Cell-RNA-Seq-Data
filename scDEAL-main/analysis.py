import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy as sc

import glob
from sklearn.metrics import classification_report
from scipy.stats import pearsonr
import utils as ut
from utils import de_score
import random
import seaborn as sns
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings('ignore')
from scipy.stats import ranksums,ttest_ind
str1 = 'E:\\code\\scDEAL-main\\save/adata\\GSE117872_HN1371214data_GSE117872_HN137_drug_CISPLATIN_bottle_64_edim_256,128_pdim_128,64_model_DAE_dropout_0.3_gene_T_lr_0.1_mod_new_sam_upsampling.h5ad'
name = str1.split("1214")[0].split("\\")[4]
print(name)
# scDEAL
Deep Transfer Learning of Drug Sensitivity by Integrating Bulk and Single-cell RNA-seq data

## 数据准备
### 数据下载
[ scDEAL数据集下载](https://portland-my.sharepoint.com/:u:/g/personal/junyichen8-c_my_cityu_edu_hk/ER2m5OXpYrdPngoAf06pqDoBsiuItm9yvAqg_CjHhNvKSA?e=ckLJ91) 

数据集信息如下
|               |     Author             |     Drug         |     GEO access    |     Cells    |     Species           |     Cancer type                        |
|---------------|------------------------|------------------|-------------------|--------------|-----------------------|----------------------------------------|
|     Data 1&2  |     Sharma, et al.     |     Cisplatin    |     GSE117872     |     548      |     Homo   sapiens    |     Oral   squamous cell carcinomas    |
|     Data 3    |     Kong, et al.       |     Gefitinib    |     GSE112274     |     507      |     Homo   sapiens    |     Lung   cancer                      |
|     Data 4    |     Schnepp, et al.    |     Docetaxel    |     GSE140440     |     324      |     Homo   sapiens    |     Prostate   Cancer                  |
|     Data 5    |     Aissa, et al.      |     Erlotinib    |     GSE149383     |     1496     |     Homo sapiens      |     Lung cancer                        |
|     Data 6    |     Bell, et al.       |     I-BET-762    |     GSE110894     |     1419     |     Mus   musculus    |     Acute   myeloid leukemia           |

代码文件夹中包括如下内容:
```
bulkmodel.py  DaNN  LICENSE    README.md    save       scDEALenv   trainers.py    utils.py
casestudy     data  models.py  sampling.py  scanpypip  scmodel.py  trajectory.py
```

### 目录内容

- data：学习所需的数据集
- save/logs：记录运行状态的日志和错误文件。
- save/figures& figures：运行过程中生成的数据。
- save/models：运行过程中训练的模型。
- save/adata：AnnData 输出结果。
- DaNN：描述模型的 python 脚本。
- scanpypip：实用程序的 python 脚本。

## 运行实例

###代码运行方法
用预训练好的模型对数据集进行预测时，bulkmodel.py和scmodel.py的运行方法如下：
```
python bulkmodel.py --drug "I.BET.762" --dimreduce "DAE" --encoder_h_dims "256,128" --predictor_h_dims "128,64" --bottleneck 512 --data_name "GSE110894" --sampling "upsampling" --dropout 0.1 --lr 0.5 --printgene "F" -mod "new" --checkpoint "save/bulk_pre/integrate_data_GSE110894_drug_I.BET.762_bottle_512_edim_256,128_pdim_128,64_model_DAE_dropout_0.1_gene_F_lr_0.5_mod_new_sam_upsampling"

python scmodel.py --sc_data "GSE110894" --dimreduce "DAE" --drug "I.BET.762" --bulk_h_dims "256,128" --bottleneck 512 --predictor_h_dims "128,64" --dropout 0.1 --printgene "F" -mod "new" --lr 0.5 --sampling "upsampling" --printgene "F" -mod "new" --checkpoint "save/sc_pre/integrate_data_GSE110894_drug_I.BET.762_bottle_512_edim_256,128_pdim_128,64_model_DAE_dropout_0.1_gene_F_lr_0.5_mod_new_sam_upsampling_DaNN.pkl"
```

若想从头开始训练只需设置--checkpoint 为"False"，以下为一个训练例子:
```
python bulkmodel.py --drug "I.BET.762" --dimreduce "DAE" --encoder_h_dims "256,128" --predictor_h_dims "128,64" --bottleneck 512 --data_name "GSE110894" --sampling "upsampling" --dropout 0.1 --lr 0.5 --printgene "F" -mod "new" --checkpoint "False"

python scmodel.py --sc_data "GSE110894" --dimreduce "DAE" --drug "I.BET.762" --bulk_h_dims "256,128" --bottleneck 512 --predictor_h_dims "128,64" --dropout 0.1 --printgene "F" -mod "new" --lr 0.5 --sampling "upsampling" --printgene "F" -mod "new" --checkpoint "False"
```
 


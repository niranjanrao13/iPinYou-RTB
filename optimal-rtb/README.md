# Steps to run the CTR experiments

1. Assuming your pre-processed data files were created safely by using the steps followed in [make-ipinyou-data README](/make-ipinyou-data/README.md), you can run 
```
python lr.py ../make-ipinyou-data/train.yzx.txt ../make-ipinyou-data/test.yzx.txt
```
for the AUC/RMSE report.
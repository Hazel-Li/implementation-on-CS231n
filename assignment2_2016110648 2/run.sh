#/bin/bash
#BSUB -J test_torch
#BSUB -e %J.err
#BSUB -o %J.out
#BSUB -n 1
#BSUB -q  gauss
#BSUB -R "select [ngpus>0] rusage [ngpus_excl_p=4]"
python  /nfsshare/home/dl01/lyt/assignment2_2016110648/Task3_Tensorflow.py
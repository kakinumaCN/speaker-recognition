# VPR

usage: python gmm.py [-h|--help] [-d|--datapath <datapath>] [-m|--modelpath <modelpath>] [-t|--runtype 1|2] [--train] [--test]
arguments: choose runtype 1 to run train()
           choose runtype 2 to run train_path()
default: datapath:../testpath modelpath:../testmodel runtype:2
examples: python gmm.py -d ../persondata/ -m ../models/ -t 2 --train --test
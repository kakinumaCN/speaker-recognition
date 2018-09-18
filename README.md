# VPR

usage: python gmm.py [-h|--help] [-d|--datapath <datapath>] [-m|--modelpath <modelpath>] [-t|--runtype 1|2] [--train] [--test]<br>
arguments: choose runtype 1 to run train()<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;choose runtype 2 to run train_path()<br>
default: datapath:../testpath modelpath:../testmodel runtype:2<br>
examples: python gmm.py -d ../persondata/ -m ../models/ -t 2 --train --test<br>
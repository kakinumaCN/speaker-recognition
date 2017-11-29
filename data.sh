path=test_data
# cd path
mkdir test_ubm_data
par= 
for i in `ls $path`
do
    name=`echo $i | sed 's/.wav//g'`
    echo $name
    # mkdir test_ubm_data'/'$name
    # cp $path'/'$i test_ubm_data'/'$name'/'$i
    par=$par' ~/Developer/VPR/'test_ubm_data'/'$name
done

par=${par:1}
par=\'$par\'
# echo $par
echo python speaker-recognition-master/src/speaker-recognition.py -t enroll -i $par -m speakerM.out

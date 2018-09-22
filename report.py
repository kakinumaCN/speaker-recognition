import getopt
import sys


if __name__ == '__main__':

    FILE = '../re.log'
    opts, args = getopt.getopt(sys.argv[1:], 'hf:', ['help','file='])
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print 'This is help :D'
        elif opt in ('-f', '--file'):
            FILE = arg
    
    f = open(FILE, 'r')
    result_list = f.readlines()
    count = 0
    xcount = 0
    lastspeaker = ' '
    for result1 in result_list:
        file_speaker = result1.split(' ')[0].split('/')[8]
        model_speker = result1.split(' ')[1].split('/')[2].split('.')[0]

        if lastspeaker!=' ' and file_speaker != lastspeaker:
            print lastspeaker +'/'+ str(count) +'/'+ str(xcount) + '/'+str(float(count)/float((count+xcount)))
            count = 0
            xcount = 0

        if file_speaker == model_speker :
            count += 1
        else:
            xcount += 1
        lastspeaker = file_speaker
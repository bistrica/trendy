
from os.path import isfile, join
from os import listdir

main_path='/home/olusiak/AUTOPYTH5/'#UTOPYTH2/'
files = [f for f in listdir(main_path) if isfile(join(main_path, f))]
files.sort()


for file in files:
    #print file
    Xb=0
    Xr=0
    c=0
    for line in open(main_path+file):
        #print line
        if len(line.strip())>60:
            c+=1
            els=line.split(';')
            x_b=float(els[5])/float(els[4])
            #print els[4], els[5], x_b
            x_r = float(els[11]) / float(els[10])
            #x_r = float(els[10]) / float(els[9])
            Xb+=x_b
            Xr+=x_r
    #print 'C ',c
    Xb/=c
    Xr/=c
    print file, ';',Xb,';', Xr
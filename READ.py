
from os.path import isfile, join
from os import listdir

main_path='/home/olusiak/ALLPYTH7/'#UTOPYTH2/'
files = [f for f in listdir(main_path) if isfile(join(main_path, f))]
files.sort()


for file in files:
    print file
    Xb=0
    Xr=0
    c=0
    Xsr=0
    for line in open(main_path+file):

        #print line
        if len(line.strip())>60:
            if c == 5:
                break
            c+=1

            els=line.split(';')
            x_b=float(els[5])/float(els[4])
            #print els[4], els[5], x_b
            try:
                x_sr = abs(float(els[17]))
                x_r = float(els[11]) / float(els[10])
            except:
                x_sr = abs(float(els[13]))
                x_r = float(els[10]) / float(els[9])
            #x_r = float(els[10]) / float(els[9])
            Xb+=x_b
            Xr+=x_r
            Xsr+=x_sr
    #print 'C ',c

    Xb/=c
    Xr/=c
    Xsr/=c

    print file, ';',Xb*100,';', Xr*100,';',Xsr*100
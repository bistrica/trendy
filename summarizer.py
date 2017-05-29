
import cv2
import matplotlib.pyplot as plt
from skimage.morphology import convex_hull_image

import numpy as np

from PIL import Image
from scipy.spatial import ConvexHull

class Summarizer(object):

    min_perc=0.8#6
    mat_min_size=0.008
    value=30

    def __init__(self,min_p,min_s,val):
        self.min_perc=min_p
        self.mat_min_size=min_s
        self.value=val

    def count_with_median(self,img2):

        pix_pic = img2.load()
        img=pix_pic
        dic = dict()
        nonblack = 0

        for i in range(img2.size[0]):
            for j in range(img2.size[1]):
                #print '>',img[i,j]
                if isinstance(img[i,j],int):
                    if not (img[i, j] == 0):
                        nonblack += 1
                        key = img[i, j]
                        if key not in dic.keys():
                            dic[key] = 1
                        else:
                            dic[key] += 1
                else:
                #c=9/0 img[i,j]==-1 and not img[i,j]==1:#
                    if not (img[i,j][0] == 0 and img[i,j][1] == 0 and img[i,j][2] == 0) and not (img[i, j][0] == 255 and img[i,j][1] == 255 and img[i,j][2] == 255):  # ,0,0]:

                        nonblack += 1
                        key = img[i,j]#(img[i,j][0],img[i,j][1],img[i,j][2] )#str(img[i,j][0]) + "_" + str(+img[i,j][1]) + "_" + str(img[i,j][2])
                        if key not in dic.keys():
                            dic[key] = 1
                        else:
                            dic[key] += 1

        ids=[]
        if self.value!=0:
            for k in dic.keys():
                if dic[k]<self.value:
                    ids.append(k)
            for k in ids:
                dic.pop(k)
        keys = dic.values()#keys()
        if len(keys)==0:
            print 'o!'
            plt.imshow(img2)
            plt.show()
        keys.sort()
        #print 'keys ',keys
        #print len(keys),' non ',nonblack,' img2.shape ',img2.size[0]*img2.size[1]

        median = float(keys[len(keys) / 2])
        if len(keys) % 2 == 0:

            median += float(keys[len(keys) / 2 + 1])
            median /= 2
        #print 'medi ',median
        cells = float(nonblack) / median
        return (len(keys),cells)

    def count_tp(self, img, dens, dens_mat):

        if dens.size!=img.size:
            img=img.resize((dens.size))#.thumbnail(dens.size,Image.ANTIALIAS)
            #dens.thumbnail(img.size, Image.BILINEAR)#ANTIALIAS)
            #dens_mat=np.reshape(dens_mat,(img.size[0],img.size[1]))
        #plt.imshow(dens_mat)
        #plt.show()
        #print 'de'
        #plt.imshow(dens)
        #plt.show()
        #print 'im'
        #plt.imshow(img)
        #plt.show()
        pix_pic = img.load()
        pix_dens = dens.load()
        colours=set()
        ccc=set()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                ccc.add(pix_dens[i,j])
        #print 'ccc ',len(ccc)
        #print 'CP ',pix_pic[0,0], pix_dens[0,0]
        COL=set()
        COL_DENS=set()
        DENS=set()
        s=set()
        for i in range(img.size[0]):
            for j in range(img.size[1]):
                cpixel=pix_dens[i,j]
                s.add(cpixel)
                cpixelImg = pix_pic[i, j]
                COL.add(cpixelImg)#pix_pic[i,j])
                if (cpixel!=0):#0 and cpi[0],cpixel[1],cpixel[2])!=(0,0,0) and (cpixel[0],cpixel[1],cpixel[2])!=(255,255,255):

                    #pix_pic[i, j] = (0, 0, 0)
                    try:
                        if (cpixelImg[0], cpixelImg[1], cpixelImg[2]) != (0,0,0) and (cpixelImg[0], cpixelImg[1], cpixelImg[2])!= (-1,-1,-1):#(255,255,255):#0, 0, 0) and (cpixelImg[0], cpixelImg[1], cpixelImg[2]) != (255, 255, 255):
                            descr=cpixel#str(cpixelImg[0])+'_'+str(cpixelImg[1])+'_'+str(cpixelImg[2])+'.'+str(cpixel)
                            colours.add(descr)
                            COL_DENS.add(cpixelImg)

                    except:

                        if (cpixelImg!=0 and cpixelImg!=-1):#0, 0, 0) and (cpixelImg[0], cpixelImg[1], cpixelImg[2]) != (255, 255, 255):
                            descr=cpixel#str(cpixelImg)+'.'+str(cpixel)
                            colours.add(descr)
                            COL_DENS.add(cpixelImg)
                DENS.add(cpixel)
                            #pix_pic[i,j]=99
        colours=list(colours)
        colours.sort()
        fp=0
        for c in COL:
            if c not in COL_DENS:
                fp+=1
        print 'ss',len(s),len(colours)
      #  print 'COL ',len(COL),'DENS ',len(DENS)
        return len(colours),fp

    def make_density_map(self, path, is_red, imm):
        im = Image.open(path)  # "/home/olusiak/Obrazy/schr.png")
        #if im.size!=imm.size:
        #    im.thumbnail(imm.size, Image.ANTIALIAS)

        pix = im.load()
        red=(0,0,255)
        blue=(0,255,255)
        #plt.imshow(im)
        #plt.show()
        color=None#blue
        if is_red:
            color=red
        for i in range(im.size[0]):
            if color is None:
                break
            for j in range(im.size[1]):

                cpixel=pix[i,j]

                if (cpixel[0],cpixel[1],cpixel[2])!=color:
                    c=(0,0,0,cpixel[3])
                    pix[i,j]=c
                #else:
                #    print 'cpix',cpixel

        #plt.imshow(im)
        #plt.show()
        ratio=1
        im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((im.size[1], im.size[0], 4))



        kernel = np.ones((3, 3), np.uint8)
        kernel2 = np.ones((ratio * 2 + 1, ratio * 2 + 1), np.uint8)
        #plt.imshow(im_arr)
        #plt.show()
        erosion = cv2.erode(im_arr, kernel, iterations=1)
        erosion = cv2.dilate(erosion, kernel2, iterations=4)
        erosion=cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
        #plt.matshow(erosion)
        #plt.show()
        ret,thresh=cv2.threshold(erosion,1,255,cv2.THRESH_BINARY)#_INV)

        if False:#imm.size != im.size:
            a = thresh
            a = np.reshape(a, (a.shape[0] / 2, 2, a.shape[1] / 2, 2))
            a = np.sum(a, axis=(1, 3)) >= 2
            print 'a'
            print a
            # plt.imshow(a)
            # plt.show()
            # img2 = Image.fromarray(a)
            thresh=a
#        else:
        thresh = np.uint8(thresh)
        #print 'thres ',thresh.shape
        ret, markers = cv2.connectedComponents(thresh)
        colors=set()
 #       plt.imshow(markers)
 #       plt.show()
        for x in range(markers.shape[0]):
            for y in range(markers.shape[1]):
                #print 'rk ',markers[x,y],type(markers[x,y])
                if markers[x,y].size==1:#isinstance(markers[x,y],int):
                    colors.add(markers[x,y])
                #elif isinstance(markers[x,y][0],int):
                #    colors.add(markers[x,y][0])
                else:
                    colors.add((markers[x, y][0],markers[x, y][1],markers[x, y][2]))

        #print 'c ', colors
        #plt.imshow(markers)
        #plt.show()

        colors=len(colors)-1
        #img2 = np.zeros_like(im)
        #img2[:, :, 0] = markers
        #img2[:, :, 1] = markers
        #img2[:, :, 2] = markers
       # plt.imshow(markers)
       # plt.show()


        #img2 = Image.fromarray(markers)
        #if img2.size!=imm.size:
        #    img2.thumbnail(imm.size, Image.ANTIALIAS)
        #print 'IMG2'

        img2=Image.fromarray(markers)
       # plt.imshow(img2)
       # plt.show()
        return img2,colors,markers

    def check_valid_cell(self,mat,color):
        maxW=-1
        maxH=-1
        minH=9000
        minW=9000
        count=0
        pic=mat.load()
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                #print 'm',type(mat[i][j]),color
                #matt=[mat[i][j][0],mat[i][j][1],mat[i][j][2]]

                'ma', type(pic[i,j]),pic[i,j]
                if np.array_equal(pic[i,j],color):
                    count+=1
                    if i<=minW:
                        minW=i
                    if i>=maxW:
                        maxW=i
                    if j<=minH:
                        minH=j
                    if j>=maxH:
                        maxH=j
        res=(maxW-minW)*(maxH-minH)
        perc=float(count)/float(res)
        valid=(perc>=self.min_perc)
        print  count,res,perc,valid,maxH,minH,maxW,minW
        return count,res,perc,valid

    def PolyArea2D(self,pts):
        lines = np.hstack([pts, np.roll(pts, -1, axis=0)])
        area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in lines))
        return area

    def check_valid_cell_all_colours_convex(self,mat,colors):
        maxW=-1
        maxH=-1
        minH=9000
        minW=9000
        #count=0
        pic=mat.load()
        dic_col=dict()
        for c in colors:
            if c==0 or c==(0,0,0):
                continue
            dic_col[c]=[9000,-1,9000,-1,0,list()]
        for i in range(mat.size[0]):
            for j in range(mat.size[1]):

                if isinstance(pic[i,j],int):
                    key=pic[i,j]
                else:
                    key=tuple(pic[i,j])#.tolist())
                if key in dic_col.keys():#np.array_equal(mat[i][j],color):

                    dic_col[key][4]+=1
                    dic_col[key][5].append((i,j))
                    if i<=dic_col[key][0]:#minW:
                        dic_col[key][0]=i
                    if i>=dic_col[key][1]:
                        dic_col[key][1]=i
                    if j<=dic_col[key][2]:
                        dic_col[key][2]=j
                    if j>=dic_col[key][3]:
                        dic_col[key][3]=j
        #print 'MAT SIZE ',mat.size
        for k in dic_col.keys():
            #print '[4] ',dic_col[k][4]
            if dic_col[k][4]<self.mat_min_size*mat.size[0]:
                dic_col[k] = (dic_col[k][4], 0, 0, False, (0,0,0,0))
                #print '<0.00'
                continue

            #print 'colo ',k
            minW=dic_col[k][0]
            maxW = dic_col[k][1]
            minH = dic_col[k][2]
            maxH = dic_col[k][3]
            imm=np.zeros(((maxW-minW+1,maxH-minH+1)))
            for p in dic_col[k][5]:
                imm[p[0]-minW][p[1]-minH]=255


            convex_count=0
            chull = convex_hull_image(imm)

            #print 'chu ',type(chull)
            for i in range(chull.shape[0]):
                for j in range(chull.shape[1]):
                    #print 'ch: ',chull[i][j]
                    if chull[i][j] == True:
                        convex_count+=1

            #plt.imshow(imm)
            #plt.show()
            #plt.imshow(chull)
            #plt.show()
            dic_col[k][5]=np.asarray(dic_col[k][5])
            #print 'dicc ',dic_col[k][5]
#            hull = ConvexHull(dic_col[k][5])
            #area = hull.area
            #area = cv2.contourArea(hull)
            #print 'area',convex_count,dic_col[k][4]
            if convex_count>20000:
                plt.plot(dic_col[k][5][:, 0], dic_col[k][5][:, 1], 'o')
                #plt.show()
                plt.fill(dic_col[k][5], dic_col[k][5], 'k', alpha=0.3)
                #plt.show()

            #plt.fill(dic_col[k][5][hull.vertices, 0], dic_col[k][5][hull.vertices, 1], 'k', alpha=0.3)
            #plt.show()
            #res=(maxW-minW)*(maxH-minH)

            count=dic_col[k][4]
            perc = float(count) / float(convex_count)
            valid = (perc >= self.min_perc)
            dic_col[k]=(count,convex_count,perc,valid,(minW,maxW,minH,maxH))#dic_col[k])

        #print  count,res,perc,valid,maxH,minH,maxW,minW
        #print 'dic ',dic_col
        #c=9/0
        return dic_col#count,res,perc,valid


    def check_valid_cell_all_colours(self,mat,colors):
        maxW=-1
        maxH=-1
        minH=9000
        minW=9000
        #count=0
        pic=mat.load()
        dic_col=dict()
        for c in colors:
            if c==0 or c==(0,0,0):
                continue
            dic_col[c]=[9000,-1,9000,-1,0]
        for i in range(mat.size[0]):
            for j in range(mat.size[1]):
                #print 'i j ',i,j
                #print 'm',type(mat[i][j]),color
                #matt=[mat[i][j][0],mat[i][j][1],mat[i][j][2]]

                #'ma', type(mat[i][j]),mat[i][j]
                if isinstance(pic[i,j],int):
                    key=pic[i,j]
                else:
                    key=tuple(pic[i,j])#.tolist())
                if key in dic_col.keys():#np.array_equal(mat[i][j],color):
                    dic_col[key][4]+=1
                    if i<=dic_col[key][0]:#minW:
                        dic_col[key][0]=i
                    if i>=dic_col[key][1]:
                        dic_col[key][1]=i
                    if j<=dic_col[key][2]:
                        dic_col[key][2]=j
                    if j>=dic_col[key][3]:
                        dic_col[key][3]=j
        for k in dic_col.keys():
            minW=dic_col[k][0]
            maxW = dic_col[k][1]
            minH = dic_col[k][2]
            maxH = dic_col[k][3]
            res=(maxW-minW)*(maxH-minH)
            count=dic_col[k][4]
            perc = float(count) / float(res)
            valid = (perc >= self.min_perc)
            dic_col[k]=(count,res,perc,valid,dic_col[k])

        #print  count,res,perc,valid,maxH,minH,maxW,minW
        #print 'dic ',dic_col
        #c=9/0
        return dic_col#count,res,perc,valid

    def check_mat(self,img):
        if self.mat_min_size==0 and self.min_perc==0:
            return img
        #im = Image.open(path)  # "/home/olusiak/Obrazy/schr.png")

        #print im_arr[0]
        #plt.imshow(im_arr)
        #plt.show()
        fx = img.load()#im_arr#img.load()
        if isinstance(fx[0,0],int):
            #print 'zeros'
            #c=9/0
            im_arr = np.zeros((img.size[1], img.size[0]))
            for i in range(img.size[0]):
                for j in range(img.size[1]):
                    im_arr[j][i]=fx[i,j]
        else:
            im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((img.size[1], img.size[0], im_arr.size / (img.size[0] * img.size[1])))
        #fx=mat
        xx = set()
       # print fx
        for i in range(img.size[0]):#fx.shape[0]):
            for j in range(img.size[1]):#fx.shape[1]):

                #print 'imm ',im_arr[i][j]

                if isinstance(fx[i,j],int):
                    s=fx[i,j]
                else:
                    s=(fx[i,j][0],fx[i,j][1],fx[i,j][2])
                #s=tuple(im_arr[i][j])
                #print 's ',s
                xx.add(s)#str(fx[i][j][0]) + '_' + str(fx[i][j][1]) + '_' + str(fx[i][j][2]))
        #print 'size ', len(xx)
        invalid=list()
        dic=self.check_valid_cell_all_colours_convex(img,xx)
#        c=8/0

        #for x in xx:
        #    tab = list(x.split('_'))

        #    for i in range(len(tab)):
        #        tab[i] = int(tab[i])
        #    #tab=tuple(tab)
        #    #if len(tab)==1:

        #    #    tab=tab[0]
        #    tab=np.asarray(tab)
        #    print 'tab ',tab
        #    count, res, perc, valid=self.check_valid_cell(fx,tab)
        #    print tab,': ',count,res,perc,valid
        for k in dic.keys():
            (count, res, perc, valid, tup)=dic[k]
            #print 'k: ',k,', ', count, res, perc, valid, '[',tup,']'
            if not valid:
                invalid.append(k)
        change=(0,0,0)
        if isinstance(fx[0,0],int):
            change=0
       # print 'inv ', invalid
        if len(invalid)!=0:
            for i in range(img.size[0]):
                for j in range(img.size[1]):
                    #print 'i,j',i,j,fx[i,j]
                    if isinstance(fx[i,j],int):
                        if fx[i,j] in invalid:# or tuple(fx[i,j]) in invalid:#.tolist()) in invalid:
                            im_arr[j][i]=change
                        #else:
                        #    im_arr[j][i] = fx[i,j]
                    else:
                        if tuple(fx[i,j]) in invalid:
                            im_arr[j][i] = change
                        #else:
                        #    im_arr[j][i] = tuple(fx[i,j])
                #else:
                #    im_arr[i][j]
        #print '>',im_arr
        im_arr = np.uint8(im_arr)
        #print 'imm ',im_arr
        fx = Image.fromarray(im_arr)
        #plt.imshow(fx)
        #plt.show()
        return fx
        #        xx.add(str(fx[i][j][0])+'_'+str(fx[i][j][1])+'_'+str(fx[i][j][2]))
        #print 'size ', len(xx)

    def colorr(self,img):

        tab = [11, 57, 104]#[75, 122, 191]#[200,151,117]#[202,149,113]#[121,109,194]#[75, 122, 191]
        print 'xxxx'
        tab=np.asarray(tab)
        tab2 = [255, 0, 0]
        tab2 = np.asarray(tab2)
        im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((img.size[1], img.size[0], im_arr.size / (img.size[0] * img.size[1])))
        fx = im_arr

        for i in range(fx.shape[0]):
            for j in range(fx.shape[1]):
                if np.array_equal(fx[i][j],tab):
                    fx[i][j]=tab2

        #plt.imshow(fx)
        #plt.show()
        self.check_valid_cell(fx,tab2)
        #plt.imshow(segmentation.mark_boundaries(im_arr, fx))
        #plt.show()
        fx = Image.fromarray(fx)
        #plt.imshow(fx)
        #plt.show()
        return fx
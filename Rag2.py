from skimage import data, segmentation, filters, color
from skimage.future import graph
from matplotlib import pyplot as plt
from skimage import data, io, segmentation, color
from skimage.future import graph
from random import shuffle
import cv2
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image
from os.path import isfile, join
from os import listdir

from summarizer import Summarizer
from Water import process_water, process_quick, process_felzen, process_suzuki

WATER=0
RAG=1
QUICK=2
FELZ=3
SUZUKI=4



def weight_boundary(graph, src, dst, n):
    """
    Handle merging of nodes of a region boundary region adjacency graph.

    This function computes the `"weight"` and the count `"count"`
    attributes of the edge between `n` and the node formed after
    merging `src` and `dst`.


    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the "weight" and "count" attributes to be
        assigned for the merged node.

    """
    default = {'weight': 0.0, 'count': 0}

    try:
        count_src = graph[src].get(n, default)['count']
    except:
        count_src=0
    try:
        count_dst = graph[dst].get(n, default)['count']
    except:
        count_dst=0
    weight_src = graph[src].get(n, default)['weight']
    weight_dst = graph[dst].get(n, default)['weight']
    try:
        weight_src = graph[src].get(n, default)['weight']['weight']


    except:
        weight_src = graph[src].get(n, default)['weight']

    try:

        weight_dst = graph[dst].get(n, default)['weight']['weight']
    except:
        weight_dst = graph[dst].get(n, default)['weight']

    count = count_src + count_dst
    #print '>',count_src , weight_src , count_dst , weight_dst, count
    #print 'c ',count, 'w ',(count_src * weight_src + count_dst * weight_dst)/count
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass


main_path='/home/olusiak/Obrazy/densities/original/'
densities_path=main_path+'densities/'
deconv_path='/home/olusiak/Obrazy/densities/deconv/'


def process(path,N_SEGM,THRESH,ADAP):

    im = Image.open(path)#'/home/olusiak/Obrazy/rois/41136_001.png-2.jpg')
    #ran=1

    #im_arr2 = np.fromstring(im.tobytes(), dtype=np.uint8)
    #im_arr2 = im_arr2.reshape((im.size[1], im.size[0], 3))
    #for x in range(len(im_arr2)):
    #    for y in range(len(im_arr2[0])):
    #        im_arr2[x][y][2]=0#=(im_arr2[0],255,255)
    #        im_arr2[x][y][1]=0
    #plt.imshow(im_arr2)
    #plt.show()

    im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((im.size[1], im.size[0], 3))



    gray = cv2.cvtColor(im_arr,cv2.COLOR_BGR2GRAY)



    if ADAP:
        thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 55, 20)

    else:
        data = dict()
        for i in gray.ravel():
            if data.has_key(i):
                data[i] += 1
            else:
                data[i] = 1
        # if data.has_key(255):
        max = 0
        id = None
        for k in data.keys():
            # print 'k ',k
            if data[k] >= max and k < 240 and k > 130:  # 50:#240:
                id = k
                max = data[k]
        # for k in data.keys():
        #    print 'k ',k,' ',data[k]
        # print 'key: ',id,' - ',max
        id -= THRESH  # 35
        ret, thresh = cv2.threshold(gray,id,255,cv2.ADAPTIVE_THRESH_MEAN_C+cv2.THRESH_BINARY_INV)#+cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+


    #plt.matshow(thresh,cmap='gray')
    #plt.show()

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)
    opening2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
    #plt.imshow(opening)
    #plt.show()
    for i in range(opening.shape[0]):
        for j in range(opening.shape[1]):
            if opening[i][j]==0:
                im_arr[i][j]=(0,0,0)

    #plt.imshow(im_arr)
    #plt.show()
    #plt.matshow(opening)
    #plt.show()
    #plt.imshow(im_arr)
    #plt.show()


    #im = Image.fromarray(im_arr)
    #im.thumbnail((im.size[0] / ran, im.size[1] / ran), Image.ANTIALIAS)
    #im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
    #im_arr = im_arr.reshape((im.size[1], im.size[0], 3))



    #im = Image.fromarray(opening)
    #im.thumbnail((im.size[0] / ran, im.size[1] / ran), Image.ANTIALIAS)
    #opening = np.fromstring(im.tobytes(), dtype=np.uint8)
    #opening = opening.reshape((im.size[1], im.size[0]))


    img2=im_arr


    edges = filters.sobel(color.rgb2gray(img2))
    labels = segmentation.slic(img2, compactness=30, n_segments=N_SEGM)#30 2000
    #g = graph.rag_mean_color(img2, labels)#added

    g = graph.rag_boundary(labels, edges)#first

    #graph.show_rag(labels, g, img)
    #plt.title('Initial RAG')


    labels2 = graph.merge_hierarchical(labels, g, thresh=0.98, rag_copy=False,# 0.08
                                       in_place_merge=True,
                                       merge_func=merge_boundary,
                                       weight_func=weight_boundary)

    #final_labels = graph.cut_threshold(labels, g, 29)#added
    #final_label_rgb = color.label2rgb(final_labels, img2, colors=cols, kind='overlay')#added
    #labels2=final_label_rgb#added
    #plt.imshow(final_label_rgb)
    #plt.show()
    #graph.show_rag(labels, g, im)
    #plt.title('RAG after hierarchical merging')

    #plt.figure()


    #ret, opening = cv2.threshold(opening,0,255,cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+

    #out = color.label2rgb(labels2, img2, kind='avg')
    s=set()
    for row in labels2:
        for e in row:
           s.add(e)
    #print 'sss ',len(s)
    cols = list()
    c = 0
    cp = len(s)+5
    for r in range(0, 256, 1):
        for r2 in range(0, 256, 1):
            for r3 in range(0, 256, 1):
                cols.append((r, r2, r3))
                cp -= 1
                if cp == 0:
                    break

            if cp == 0:
                break
        if cp == 0:
            break
    # print 'cols', len(cols)

    shuffle(cols)


    img2 = np.zeros_like(img2)
    img2[:, :, 0] = opening2
    img2[:, :, 1] = opening2
    img2[:, :, 2] = opening2
    out = color.label2rgb(labels2, img2, colors=cols, kind='overlay', alpha=1)

    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            if img2[i][j][0]==0 and img2[i][j][1]==0 and img2[i][j][2]==0:#,0,0]:
                out[i][j][0]=0
                out[i][j][1]=0
                out[i][j][2]=0


    #print 'OUT'
    #plt.imshow(out)
    #plt.show()

    #plt.imshow(out)
    #plt.show()

    #xx = set()

    #for i in range(out.shape[0]):  # fx.shape[0]):
    #    for j in range(out.shape[1]):  # fx.shape[1]):

#            s = (out[i, j][0], out[i, j][1], out[i, j][2])

#            xx.add(s)

    out = np.uint8(out)
    im = Image.fromarray(out)


    #plt.imshow(out)
    #plt.show()
    return im






def process_files(type, min_p,min_s,value,thresh,adap,n_seg,dist_perc):


    TYPE=type#RAG#QUICK#RAG#WATER
    folder='ALL FELZ'
    fileSumm = open('/home/olusiak/ALL QUICK SIZE TYPE '+str(TYPE)+' THRESH '+str(thresh)+' ADAP '+str(adap)+'N SEG '+str(n_seg)+'DIST_P '+str(dist_perc)+'CHM '+str(value)+'_'+str(min_p)+'_'+str(min_s)+'.txt', 'w+')
    summ=Summarizer(min_p,min_s,value)
    files = [f for f in listdir(main_path) if isfile(join(main_path, f))]
    files.sort()
    print 'type',TYPE
    cf=0
    DIFF=0

    for file in files:
        cf+=1
        if cf==11:
            break
        #file2=main_path+file
        #file='134_13_004.png'#'1379_13_001.png'#1042_13_002.png'#'132_13_002.png'#'1379_13_001.png'#134_13_004.png'#1042_13_002.png'#''1381_13_003.png'
        #print file
        #file='134_13_004.png'#102_13_003.png'
        #file='1042_13_002.png'

        red_img=deconv_path+file+'-(Colour_2).jpg'
        blue_img = deconv_path + file + '-(Colour_1).jpg'

        file=file.replace('.png', '')
        dens_map=densities_path+file+'Markers_Counter Window -.png_density.png'


        #red_img=Image.open(red_img)
        if TYPE==RAG:
        #red_img=process(red_img)
            red_img = process(red_img,n_seg,thresh,adap)
            #summ.colorr(red_img)
            #red_img = summ.check_mat(red_img)
            #red_cnt2, red_cells2 = summ.count_with_median(red_img)
            #print 'red ', red_cnt2, red_cells2
            #plt.imshow(red_img)
            #plt.show()
            blue_img = process(blue_img,n_seg,thresh,adap)
            #red_img = summ.check_mat(blue_img)
            #red_cnt2, red_cells2 = summ.count_with_median(red_img)
            #print 'blu ', red_cnt2, red_cells2
            #print 'red'
            #plt.imshow(red_img)
            #plt.show()
            #print 'blu'
            #plt.imshow(blue_img)
            #plt.show()
        elif TYPE==WATER:
            #blue_img = process_water(blue_img)
            red_img = process_water(red_img,thresh,adap,dist_perc)
            #red_img = summ.check_mat(red_img)
            #red_cnt2, red_cells2 = summ.count_with_median(red_img)
            #print 'red ',red_cnt2,red_cells2
            #plt.imshow(red_img)
            #plt.show()

            blue_img = process_water(blue_img,thresh,adap,dist_perc)
            #plt.imshow(red_img)
            #plt.show()
            #plt.imshow(blue_img)
            #plt.show()
        elif TYPE==QUICK:
            red_img = process_quick(red_img)
            #summ.colorr(red_img)
            #plt.imshow(red_img)
            #plt.show()

            #red_img = summ.check_mat(red_img)
            #summ.colorr(red_img)
            #plt.imshow(red_img)
            #plt.show()

            blue_img = process_quick(blue_img)
        elif TYPE == FELZ:
            red_img = process_felzen(red_img)
            #red_img = summ.check_mat(red_img)
            # summ.colorr(red_img)
            #plt.imshow(red_img)
            blue_img = process_felzen(blue_img)
        elif TYPE == SUZUKI:
            red_img = process_suzuki(red_img,thresh,adap)
            #red_img = summ.check_mat(red_img)
            #plt.imshow(red_img)
            blue_img = process_suzuki(blue_img,thresh,adap)

        #red_img.save('/home/olusiak/Obrazy/' + folder + '/' + file + '_red_before.png')
        #blue_img.save('/home/olusiak/Obrazy/' + folder + '/' + file + '_blue_before.png')

        #red_img = summ.check_mat(red_img)
        map_red, red_gt, red_mat = summ.make_density_map(dens_map, True,red_img)
        red_img = summ.check_mat(red_img)
  #      plt.imshow(red_img)
  #      plt.show()
        #plt.imshow(red_img)
        #plt.show()
        tp_red2, fp_red = summ.count_tp(red_img, map_red, red_mat)
        red_cnt2, red_cells2 = summ.count_with_median(red_img)


        #red_img = summ.check_mat(red_img)

        red_img.save('/home/olusiak/Obrazy/'+folder+'/'+file+' ALL Q TYPE '+str(TYPE)+' THRESH '+str(thresh)+' ADAP '+str(adap)+'N SEG '+str(n_seg)+'DIST_P '+str(dist_perc)+'CHM '+str(value)+'_'+str(min_p)+'_'+str(min_s)+'_red.png')

        #tp_red = summ.count_tp(red_img, map_red)
        #red_cnt, red_cells = summ.count_with_median(red_img)
        #blue_img = summ.check_mat(blue_img)
        map_blue,blue_gt, blue_mat=summ.make_density_map(dens_map, False,blue_img)
        blue_img = summ.check_mat(blue_img)
        #plt.imshow(blue_img)
        #plt.show()
        tp_blue2,fp_blue=summ.count_tp(blue_img,map_blue,blue_mat)
        blue_cnt2, blue_cells2= summ.count_with_median(blue_img)

        blue_img = summ.check_mat(blue_img)
   #     plt.imshow(blue_img)
   #     plt.show()
        blue_img.save('/home/olusiak/Obrazy/' + folder + '/' + file + ' ALL Q TYPE ' + str(TYPE) + ' THRESH ' + str(
            thresh) + ' ADAP ' + str(adap) + 'N SEG ' + str(n_seg) + 'DIST_P ' + str(dist_perc) + 'CHM ' + str(
            value) + '_' + str(min_p) + '_' + str(min_s) + '_blue.png')

        #tp_blue = summ.count_tp(blue_img, map_blue)
        #blue_cnt, blue_cells = summ.count_with_median(blue_img)

        #x=(float(red_cells))/(float(blue_cells))
#        x2 = (float(red_cells2)) / (float(blue_cells2))
        print file,' blue: ',str(blue_cnt2),' wM '+str(blue_cells2),' GT: '+str(blue_gt)+', TP: ',str(tp_blue2)+', FP: ', str(fp_blue),', red: ',str(red_cnt2),' wM ',str(red_cells2),' gt: ',red_gt,' TP ',str(tp_red2), 'FP ', str(fp_red),'cm: ',float(red_cells2)/float(blue_cells2), ' gt: ',float(red_gt)/float(blue_gt)
        #s=9/0

        diff=((float(red_gt) / float(blue_gt)))-(float(red_cells2) / float(blue_cells2))
        fileSumm.write('\n'+file+ ';blue;'+ str(blue_cnt2)+ ';' + str(blue_cells2)+ ';' + str(blue_gt) + ';'+ str(
            tp_blue2) +';'+str(fp_blue)+ ';red;'+str(red_cnt2)+ ';'+ str(red_cells2)+ ';'+ str(red_gt)+ ';'+ str(
            tp_red2)+';'+str(fp_red)+ ';'+ str((float(red_cells2) / float(blue_cells2)))+ ';'+ str((float(red_gt) / float(blue_gt)))+';'+str((float(tp_blue2)/float(tp_blue2+fp_blue)))+';'+str((float(tp_red2)/float(tp_red2+fp_red)))+';'+str(diff))
        #D+=diff
        DIFF+=abs(diff)
        #fileSumm.write('\n'+str(diff))
    MEAN=float(DIFF/(cf-1))

    fileSumm.write('\n'+str(MEAN))
    fileSumm.close()
    return MEAN
        #blue_img = Image.open(blue_img)


        #
    #ihc = im#data.immunohistochemistry()
    #ihc_hdx = color.separate_stains(ihc, color.hdx_from_rgb)


min_percs = [0,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]#0.3, 0.5, 0.8]
min_sizes = [0,0.025,0.05]#, 0.075, 0.1, 0.125, 0.15]#, 0.015, 0.03]
vals=[0]
means=list()
for min_p in min_percs:
    for min_s in min_sizes:
        for val in vals:
            #STR= ('min_p ',min_p,min_s,val,RAG, min_p,min_s,val,30,False,4000,None)
            #print STR
            #m=process_files(RAG, min_p,min_s,val,30,False,4000,None)
            #means.append(str(m)+str(STR))

            #STR = ('min_p ', min_p, min_s, val, RAG, min_p, min_s, val, 30, False, 8000, None)
            #print STR
            #m = process_files(RAG, min_p, min_s, val, 30, False, 8000, None)
            #means.append(str(m) + str(STR))

            #STR = ('min_p ', min_p, min_s, val, RAG, min_p, min_s, val, 30, False, 4000, None)
            #print STR
            #m = process_files(RAG, min_p, min_s, val, 30, False, 4000, None)
            #means.append(str(m) + str(STR))

            #STR = ('min_p ', min_p, min_s, val,RAG, min_p, min_s, val, 30, False, 5000, None)
            #print STR
            #m=process_files(RAG, min_p, min_s, val, 30, False, 5000, None)
            #means.append(str(m)+str(STR))

            #STR = ('min_p ', min_p, min_s, val, RAG, min_p, min_s, val, 30, False, 3000, None)
            #print STR
            #m = process_files(RAG, min_p, min_s, val, 30, False, 3000, None)
            #means.append(str(m) + str(STR))

            #STR = ('min_p ', min_p, min_s, val, RAG, min_p, min_s, val, 35, False, 3000, None)
            #print STR
            #m = process_files(RAG, min_p, min_s, val, 35, False, 3000, None)
            #means.append(str(m) + str(STR))
            ###

            #STR = ('min_p ', min_p, min_s, val,RAG, min_p, min_s, val, 0, True, 2000, None)
            #print STR
            #m =process_files(RAG, min_p, min_s, val, 0, True, 2000, None)
            #means.append(str(m)+str(STR))
            #STR = ('min_p ', min_p, min_s, val, 20, False, 6000, None)
            #print STR

            #m =process_files(RAG, min_p, min_s, val, 20, False, 6000, None)
            #means.append(str(m)+str(STR))

            #STR = ('min_p ', min_p, min_s, val, 10, False, 6000, None)
            #print STR

            #m = process_files(RAG, min_p, min_s, val, 10, False, 6000, None)
            #means.append(str(m) + str(STR))

            #STR = ('min_p ', min_p, min_s, val, 25, False, 6000, None)
            #print STR
            #m =process_files(RAG, min_p, min_s, val, 25, False, 6000, None)
            #means.append(str(m)+str(STR))

            #STR = ('min_p ', min_p, min_s, val, 0, False, 6000, None)
            #print STR
            #m = process_files(RAG, min_p, min_s, val, 0, False, 6000, None)
            #means.append(str(m) + str(STR))
            #STR = ('min_p ', min_p, min_s, val, 0, True, 6000, None)
            #print STR
            #m =process_files(RAG, min_p, min_s, val, 0, True, 6000, None)
            #means.append(str(m)+str(STR))

            #STR = ('min_p ', min_p, min_s, val, 15, False, 8000, None)
            #print STR
            #m =process_files(RAG, min_p, min_s, val, 15, False, 8000, None)
            #means.append(str(m)+str(STR))
            #STR = ('min_p ', min_p, min_s, val, 35, False, 8000, None)
            #print STR
            #m =process_files(RAG, min_p, min_s, val, 35, False, 8000, None)
            #means.append(str(m)+str(STR))
            #STR = ('min_p ', min_p, min_s, val, 0, True, 8000, None)
            #print STR
            #m =process_files(RAG, min_p, min_s, val, 0, True, 8000, None)
            #means.append(str(m) +str(STR))

            #STR = ('min_p ', min_p, min_s, val, 15, False, None, 0.1)
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 15, False, None, 0.1)
            #means.append(str(m)+str(STR))
            #STR = ('min_p ', min_p, min_s, val, 30, False, None, 0.1)
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 30, False, None, 0.1)
            #means.append(str(m)+str(STR))




            #STR = ('min_p ', min_p, min_s, val, 30, False, None, 0.2)
            #print STR
            #m = process_files(WATER, min_p, min_s, val, 30, False, None, 0.2)
            #means.append(str(m) + str(STR))

            #STR = ('min_p ', min_p, min_s, val, 30, False, None, 0.05)
            #print STR
            #m = process_files(WATER, min_p, min_s, val, 30, False, None, 0.05)
            #means.append(str(m) + str(STR))

            #STR = ('min_p ', min_p, min_s, val, 35, False, None, 0.05)
            #print STR
            #m = process_files(WATER, min_p, min_s, val, 35, False, None, 0.05)
            #means.append(str(m) + str(STR))
            #STR = ('min_p ', min_p, min_s, val, 0, True, None, 0.1)
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 0, True, None, 0.1)
            #means.append(str(m)+str(STR))

            #STR = ('min_p ', min_p, min_s, val, 15, False, None, 0.05)
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 15, False, None, 0.05)
            #means.append(str(m)+str(STR))
            #STR = ('min_p ', min_p, min_s, val, 35, False, None, 0.05)
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 35, False, None, 0.05)
            #means.append(str(m)+str(STR))
            #STR = ('min_p ', min_p, min_s, val, 0, True, None, 0.05)
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 0, True, None, 0.05)
            #means.append(str(m)+str(STR))

            #STR = ('min_p ', min_p, min_s, val, 15, False, None, 0.2)
            #print STR
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 15, False, None, 0.2)
            #means.append(str(m) +str(STR))
            #STR = ('min_p ', min_p, min_s, val, 35, False, None, 0.2)
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 35, False, None, 0.2)
            #means.append(str(m)+str(STR))
            #STR = ('min_p ', min_p, min_s, val, 0, True, None, 0.2)
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 0, True, None, 0.2)
            #means.append(str(m)+str(STR))

            #STR = ('min_p ', min_p, min_s, val, 15, False, None, 0.4)
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 15, False, None, 0.4)
            #means.append(str(m)+str(STR))
            #STR = ('min_p ', min_p, min_s, val, 35, False, None, 0.4)
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 35, False, None, 0.4)
            #means.append(str(m)+str(STR))
            #STR = ('min_p ', min_p, min_s, val, 0, True, None, 0.4)
            #print STR
            #m =process_files(WATER, min_p, min_s, val, 0, True, None, 0.4)
            #means.append(str(m)+str(STR))

            #STR = ('min_p ', min_p, min_s, val, 15, False, None, None)
            #print STR
            #m =process_files(SUZUKI, min_p, min_s, val, 15, False, None, None)
            #means.append(str(m)+str(STR))
            #STR = ('min_p ', min_p, min_s, val, 35, False, None, None)
            #print STR
            #m =process_files(SUZUKI, min_p, min_s, val, 35, False, None, None)
            #means.append(str(m)+str(STR))
            #STR=( 'min_p ', min_p, min_s, val, 30, False, None, None)
            #print STR
            #m =process_files(SUZUKI, min_p, min_s, val, 30, False, None, None)
            #means.append(str(m)+str(STR))

#           STR = ('min_p ', min_p, min_s, val, 30, False, None, 0.1)
#           print STR
#           m = process_files(FELZ, min_p, min_s, val, 30, False, None, 0.1)
#           means.append(str(m) + str(STR))

           STR = ('min_p ', min_p, min_s, val, 30, False, None, 0.1)
           print STR
           m = process_files(QUICK, min_p, min_s, val, 30, False, None, 0.1)
           means.append(str(m) + str(STR))

for m in means:
    print 'M ',m

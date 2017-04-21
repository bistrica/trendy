from skimage import data, segmentation, filters, color
from skimage.future import graph
from matplotlib import pyplot as plt
from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2gray
import numpy
import sklearn
import numpy
import sklearn
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from random import shuffle
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.layers import Conv2DLayer, TransposedConv2DLayer
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano.sandbox.cuda

from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import keras

import os,sys
from PIL import Image
from os.path import isfile, join
from os import listdir
from skimage.color import rgb2gray
import theano.tensor as T
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import os
import gzip
import numpy as np
import theano
import lasagne
from lasagne import layers
from lasagne.layers import Conv2DLayer, TransposedConv2DLayer
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano.sandbox.cuda

from nolearn.lasagne import visualize
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import keras

import os,sys
from PIL import Image
from os.path import isfile, join
from os import listdir

from summarizer import Summarizer
from Water import process_water
from Water import process_quick
WATER=0
RAG=1
QUICK=2

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


def process(path):

    im = Image.open(path)#'/home/olusiak/Obrazy/rois/41136_001.png-2.jpg')
    ran=1

    im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((im.size[1], im.size[0], 3))

    gray = cv2.cvtColor(im_arr,cv2.COLOR_BGR2GRAY)


    data=dict()
    for i in gray.ravel():
        if data.has_key(i):
            data[i]+=1
        else:
            data[i]=1
    #if data.has_key(255):
    max=0
    id=None
    for k in data.keys():
        #print 'k ',k
        if data[k]>=max and k<240 and k>130:#50:#240:
            id=k
            max=data[k]
    #for k in data.keys():
    #    print 'k ',k,' ',data[k]
    print 'key: ',id,' - ',max
    id-=15
    ret, thresh = cv2.threshold(gray,id,255,cv2.ADAPTIVE_THRESH_MEAN_C+cv2.THRESH_BINARY_INV)#+cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+
    #plt.matshow(thresh)
    #plt.show()

    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)


    for i in range(opening.shape[0]):
        for j in range(opening.shape[1]):
            if opening[i][j]==0:
                im_arr[i][j]=(0,0,0)

    #plt.imshow(im_arr)
    #plt.show()
    plt.matshow(opening)
    plt.show()
    plt.imshow(im_arr)
    plt.show()


    im = Image.fromarray(im_arr)
    im.thumbnail((im.size[0] / ran, im.size[1] / ran), Image.ANTIALIAS)
    im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
    im_arr = im_arr.reshape((im.size[1], im.size[0], 3))



    #im = Image.fromarray(opening)
    #im.thumbnail((im.size[0] / ran, im.size[1] / ran), Image.ANTIALIAS)
    #opening = np.fromstring(im.tobytes(), dtype=np.uint8)
    #opening = opening.reshape((im.size[1], im.size[0]))


    img2=im_arr


    edges = filters.sobel(color.rgb2gray(img2))
    labels = segmentation.slic(img2, compactness=30, n_segments=2000)
    g = graph.rag_boundary(labels, edges)

    #graph.show_rag(labels, g, img)
    #plt.title('Initial RAG')

    labels2 = graph.merge_hierarchical(labels, g, thresh=0.08, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_boundary,
                                       weight_func=weight_boundary)

    #graph.show_rag(labels, g, im)
    #plt.title('RAG after hierarchical merging')

    plt.figure()


    #ret, opening = cv2.threshold(opening,0,255,cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+

    out = color.label2rgb(labels2, img2, kind='avg')
    #print 'op',opening


    plt.imshow(out)
    plt.show()
    #plt.matshow(opening)
    #plt.show()
    #plt.imshow(out)
    #plt.title('Final segmentation')
    #print opening.shape
    #plt.show()
    #print 'img2 ',img2

    #sets=set()
    #dic=dict()
    #nonblack=0
    for i in range(img2.shape[0]):
        for j in range(img2.shape[1]):
            #print '>',img2[i][j]
            if img2[i][j][0]==0 and img2[i][j][1]==0 and img2[i][j][2]==0:#,0,0]:
                out[i][j]=0

    im = Image.fromarray(out)


    #plt.imshow(out)
    #plt.show()
    return im

TYPE=QUICK#RAG#QUICK#RAG#WATER
folder='QUICK'
summ=Summarizer()
files = [f for f in listdir(main_path) if isfile(join(main_path, f))]
files.sort()
print 'type',TYPE
for file in files:
    #file='1381_13_003.png'
    #print file

    red_img=deconv_path+file+'-(Colour_2).jpg'
    blue_img = deconv_path + file + '-(Colour_1).jpg'

    file=file.replace('.png', '')
    dens_map=densities_path+file+'Markers_Counter Window -.png_density.png'

    #red_img=Image.open(red_img)
    if TYPE==RAG:
    #red_img=process(red_img)
        red_img = process(red_img)
        summ.colorr(red_img)
        red_img = summ.check_mat(red_img)
        plt.imshow(red_img)
        plt.show()
        blue_img = process(blue_img)
        #print 'red'
        #plt.imshow(red_img)
        #plt.show()
        #print 'blu'
        #plt.imshow(blue_img)
        #plt.show()
    elif TYPE==WATER:
        red_img = process_water(red_img)
        blue_img = process_water(blue_img)
    elif TYPE==QUICK:
        red_img = process_quick(red_img)
        #summ.colorr(red_img)
        plt.imshow(red_img)
        plt.show()

        red_img = summ.check_mat(red_img)
        summ.colorr(red_img)
        plt.imshow(red_img)
        plt.show()

        blue_img = process_quick(blue_img)


    plt.imshow(blue_img)
    plt.show()
    blue_img=summ.check_mat(blue_img)
    plt.imshow(blue_img)
    plt.show()
    plt.imshow(red_img)
    plt.show()
    red_img = summ.check_mat(red_img)
    plt.imshow(red_img)
    plt.show()
    #red_img.save('/home/olusiak/Obrazy/'+folder+'/'+file+'_red.png')
    map_red=summ.make_density_map(dens_map,True)
    tp_red = summ.count_tp(red_img, map_red)
    red_cnt, red_cells = summ.count_with_median(red_img)



    #blue_img.save('/home/olusiak/Obrazy/'+folder+'/' + file + '_blue.png')
    map_blue=summ.make_density_map(dens_map, False)
    tp_blue=summ.count_tp(blue_img,map_blue)
    blue_cnt, blue_cells= summ.count_with_median(blue_img)

    x=(float(red_cells))/(float(blue_cells))
    print file,': blue: ',blue_cnt, blue_cells, tp_blue,', red: ',red_cnt,red_cells,tp_red,' (',x,'), c: ', float(red_cnt)/float(blue_cnt)

    #blue_img = Image.open(blue_img)


    #
#ihc = im#data.immunohistochemistry()
#ihc_hdx = color.separate_stains(ihc, color.hdx_from_rgb)

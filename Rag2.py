from skimage import data, segmentation, filters, color
from skimage.future import graph
from matplotlib import pyplot as plt
from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2gray

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
    print '>',count_src , weight_src , count_dst , weight_dst, count
    return {
        'count': count,
        'weight': (count_src * weight_src + count_dst * weight_dst)/count
    }


def merge_boundary(graph, src, dst):
    """Call back called before merging 2 nodes.

    In this case we don't need to do any computation here.
    """
    pass
im = Image.open("{0}".format('/home/olusiak/Obrazy/rois/41136_001.png-2.jpg'))
ran=8

im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
im_arr = im_arr.reshape((im.size[1], im.size[0], 3))
#img = cv2.imread('/home/olusiak/Obrazy/rois/41136_001.png-2.jpg')#10978_13_001.png-1.jpg')# densities/143_13_001.png')#558_13_002.png')#fotoscale.jpg')143_13_001.png')#
gray = cv2.cvtColor(im_arr,cv2.COLOR_BGR2GRAY)
#im_arr = rgb2gray(im_arr)

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
    if data[k]>=max and k<240:#50:#240:
        id=k
        max=data[k]
print 'key: ',id,' - ',max
id-=15
ret, thresh = cv2.threshold(gray,id,255,cv2.ADAPTIVE_THRESH_MEAN_C+cv2.THRESH_BINARY_INV)#+cv2.THRESH_OTSU)#cv2.ADAPTIVE_THRESH_MEAN_C+


kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 3)



im = Image.fromarray(im_arr)
im.thumbnail((im.size[0] / ran, im.size[1] / ran), Image.ANTIALIAS)
im_arr = np.fromstring(im.tobytes(), dtype=np.uint8)
im_arr = im_arr.reshape((im.size[1], im.size[0], 3))

im = Image.fromarray(opening)
im.thumbnail((im.size[0] / ran, im.size[1] / ran), Image.ANTIALIAS)
opening = np.fromstring(im.tobytes(), dtype=np.uint8)
opening = opening.reshape((im.size[1], im.size[0]))


img=im_arr
#plt.title('Final segmentation')
#plt.imshow(opening)
#plt.show()
#img = data.coffee()
edges = filters.sobel(color.rgb2gray(img))
labels = segmentation.slic(img, compactness=30, n_segments=2000)
g = graph.rag_boundary(labels, edges)

#graph.show_rag(labels, g, img)
#plt.title('Initial RAG')

labels2 = graph.merge_hierarchical(labels, g, thresh=0.08, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_boundary,
                                   weight_func=weight_boundary)

#graph.show_rag(labels, g, img)
#plt.title('RAG after hierarchical merging')

#plt.figure()
out = color.label2rgb(labels2, img, kind='avg')
print 'op',opening
plt.matshow(opening)
plt.show()
#plt.imshow(out)
#plt.title('Final segmentation')
print opening.shape
#plt.show()
for i in range(opening.shape[0]):
    for j in range(opening.shape[1]):
        if opening[i][j]==255:
            out[i][j]=255
#print opening
#cv2.bitwise_and(out,out,opening,opening)
#plt.imshow(opening)
#plt.show()
plt.imshow(out)
plt.show()
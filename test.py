#!/usr/bin/env python
#coding=utf-8

import os
import time
import h5py
import glob
import importlib
import numpy as np
import tensorflow as tf
from utils import loadDatas as LD
import scipy.io as scio




def main(args):
    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    tf.reset_default_graph()

    dataDir = args.inDir
    resDir = args.resDir
    modelDir = args.mdDir
    mdName = args.mdName
    flist=args.flist

    md = importlib.import_module(mdName)

    if flist:
        with open(flist,'r') as f:
            flines = f.readlines()
            flines = [ele.strip()\
                    for ele in flines]

    if os.path.exists(resDir):
        files = os.listdir(resDir)
        for f in files:
            f = os.path.join(resDir,f)
            if os.path.isdir(f):
                os.rmdir(f)
            else:
                os.remove(f)
    else:
        os.makedirs(resDir)

    sess = tf.Session()
    in_image = tf.placeholder(tf.float32, [1,None,None,4])

    out_image = md.network(in_image)

    saver = tf.train.Saver(tf.global_variables())
    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
        model_file = tf.train.latest_checkpoint(modelDir)
        saver.restore(sess, model_file)
        print('Model restored from ', model_file)

        Test = glob.glob(dataDir\
                + '/*/*NOISY_RAW_010.MAT')
        Test.sort()
        for sample in Test:
            lb = sample.split('/')[-1].split('_')[0]
            if flist and (lb not in flines):
                continue

            fname = os.path.join(dataDir,sample)

            raw,pt = LD.readOneRaw(fname)

            st2 = time.time()
            output = sess.run(out_image,\
                    feed_dict={in_image: raw})
            st1 = time.time()

            print('forward cost: %.3f' % (st1-st2))
            output = np.squeeze(output, axis=0)
            output = np.minimum(np.maximum(output,0),1)

            print('save result as ".mat" file')
            sname = os.path.splitext(\
                    os.path.basename(fname))[0]
            sname = sname.replace('Noise','GT')
            sname = resDir+'/'+sname
            output = LD.unPackRaw(output,pt)
            output = np.transpose(output,(1,0))
            scio.savemat(sname, {'x': output})




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--mdDir',help='model dir')
    parser.add_argument('-n','--mdName',help='model name')
    parser.add_argument('-g','--gpu',help='gpu device index,\
            -1 for cpu',default='-1')
    parser.add_argument('-d','--inDir',help='test data dir',\
            default='./data/test1')
    parser.add_argument('-r','--resDir',help='result dir')
    parser.add_argument('-f','--flist',help='specify testing list',\
            default=0)
    args = parser.parse_args()
    main(args)

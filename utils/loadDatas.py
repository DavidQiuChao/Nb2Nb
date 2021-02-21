#!/usr/bin/env python 
#coding=utf-8

import os
import rawpy
import h5py
import glob
import logging
import numpy as np

logging.getLogger().setLevel(logging.INFO)
np.random.seed(23)



def initDataList(dataDir):
    dtList = glob.glob(dataDir\
            + '/*/*NOISY_RAW_010.MAT')
    ids = [os.path.basename(name)[0:4] for name in dtList]
    return ids,dtList


def packRaw(raw,pt):
    cfa = { 'GP':'bggr','IP':'rggb',\
            'S6':'grbg','N6':'bggr',\
            'G4':'bggr'}
    raw = np.expand_dims(raw,axis=3)
    B,H,W,C = raw.shape
    if pt in ['GP','N6','G4']:
        out = np.concatenate((
            raw[:,1:H:2,1:W:2,:],
            raw[:,1:H:2,0:W:2,:],
            raw[:,0:H:2,1:W:2,:],
            raw[:,0:H:2,0:W:2,:]),
            axis=3)
    elif pt == 'IP':
        out = np.concatenate((
            raw[:,0:H:2,0:W:2,:],
            raw[:,0:H:2,1:W:2,:],
            raw[:,1:H:2,0:W:2,:],
            raw[:,1:H:2,1:W:2,:]),
            axis=3)
    else:
        out = np.concatenate((
            raw[:,0:H:2,1:W:2,:],
            raw[:,0:H:2,0:W:2,:],
            raw[:,1:H:2,1:W:2,:],
            raw[:,1:H:2,0:W:2,:]),
            axis=3)
    return out


def unPackRaw(raw,pt):
    cfa = { 'GP':'bggr','IP':'rggb',\
            'S6':'grbg','N6':'bggr',\
            'G4':'bggr'}
    H,W,C = raw.shape
    out = np.zeros((2*H,2*W))
    if pt in ['GP','N6','G4']:
        out[1::2,1::2] = raw[...,0]
        out[1::2,0::2] = raw[...,1]
        out[0::2,1::2] = raw[...,2]
        out[0::2,0::2] = raw[...,3]
    elif pt == 'IP':
        out[0::2,0::2] = raw[...,0]
        out[0::2,1::2] = raw[...,1]
        out[1::2,0::2] = raw[...,2]
        out[1::2,1::2] = raw[...,3]
    else:
        out[0::2,1::2] = raw[...,0]
        out[0::2,0::2] = raw[...,1]
        out[1::2,1::2] = raw[...,2]
        out[1::2,0::2] = raw[...,3]
    return out


def readRawData(name):
    # read the raw data from a given path
    pt = name.split('/')[-2].split('_')[2]
    im = h5py.File(name)['x']
    im = np.expand_dims(im,0)
    _name = name[:-5]+'1.MAT'
    _im = h5py.File(_name)['x']
    _im = np.expand_dims(_im,0)
    im = np.concatenate([im,_im],0)
    im = packRaw(im,pt)
    return im


def readOneRaw(name):
    pt = name.split('/')[-2].split('_')[2]
    im = h5py.File(name)['x']
    im = np.expand_dims(im,0)
    im = packRaw(im,pt)
    return im,pt


def preLoadDataset(dtList):
    datas = []
    for dpath in dtList:
        logging.info('loading {}'.format(dpath))
        raw = readRawData(dpath)
        datas.append(raw)
    return datas


def onlineLoad(dpath):
    raw = readRawData(dpath)
    return raw


def randomProcess(data,ps):
    # random crop a patch size
    B,H,W,C = data.shape
    w = np.random.randint(0,W-ps)
    h = np.random.randint(0,H-ps)
    data = data[:,h:h+ps,w:w+ps,:]
    # do random flip and rotation
    if np.random.randint(2,size=1)[0]==1:
        data = np.flip(data,axis=1)
    if np.random.randint(2,size=1)[0]==1:
        data = np.flip(data,axis=2)
    if np.random.randint(2,size=1)[0]==1:
        data = np.transpose(data,(0,2,1,3))
    return data

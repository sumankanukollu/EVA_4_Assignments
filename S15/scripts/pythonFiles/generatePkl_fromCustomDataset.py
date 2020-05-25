import sys,os,pdb,shutil,time,random,copy,PIL,gc,pdb,pickle
import numpy as np

from pprint import pprint
from pathlib import Path
#import zipfile
from zipfile import ZipFile,ZIP_DEFLATED
from PIL import Image
from datetime import datetime
#from glob import glob

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i',   "--input",  help="Input Folder Path",       type=str)
parser.add_argument('-o',   "--output", help="Output Zip Folder Path",  type=str)
parser.add_argument('-s',   "--start",  help="BG start index",          type=int)
parser.add_argument('-e',   "--end",    help="BG end index",            type=int)
parser.add_argument('-op',  "--op",     help="Operations on directories",type=int)



parser.add_argument('-bf',     "--bg_fg_1",          default=0,  help="BG FG",            type=int)
parser.add_argument('-bfm',    "--bg_fg_mask_1",     default=0,  help="BG FG Mask",       type=int)
parser.add_argument('-bff',    "--bg_fgFlip_1",      default=0,  help="BG FG Flip",       type=int)
parser.add_argument('-bffm',   "--bg_fgFlip_mask_1", default=0,  help="BG FG Flip Mask",  type=int)

args = parser.parse_args()


    
def pickle_data_batch(zipfilename=args.input,pklFileName = args.output,rstart=args.start,rend=args.end):
    gc.enable()
    gc.get_threshold()
    # pickle dict - keys : batchlabel , fName, bgfg , bgfgM, bgfgF, bgfgFM
    dict_p = {}
    #f_bgfg,f_depth, f_bgfgM, bgfg, depth,bgfgM = [],[],[],[],[],[]
    bgfg, depth,bgfgM = [],[],[]
    dict_p['batchlabel'] = 'batch_range_%i'%(rend-rstart)
    
    #loaded_images,imgName = [],[]
    zipObj = ZipFile(zipfilename, 'r')
    files = zipObj.namelist()
    for i in files[rstart:rend]:
        fileinp = zipObj.open(i, mode='r')
        if i.startswith('bg_fg_1') or i.startswith('depth'):
            if fileinp.name.endswith('.jpg'):
                img = Image.open(fileinp)
                if img.size != (224,224):
                    img = img.resize(i,Image.ANTIALIAS)
                    img.save(i)
                #x = np.clip(np.asarray(img, dtype=float) / 255, 0, 1)
                img_np = (np.array(img))
                
                r = img_np[:,:,0].flatten()
                g = img_np[:,:,1].flatten()
                b = img_np[:,:,2].flatten()
                
                out = np.array(list(r) + list(g) + list(b),np.uint8)
                # Load dictionary object
                if i.startswith('bg_fg_1') :
                    #f_bgfg.append(i)
                    bgfg.append(out)
                elif i.startswith('depth') :
                    #f_depth.append(i)
                    depth.append(out)
                    
        elif i.startswith('bg_fg_mask_1') :
            #fileinp = zipObj.open(i, mode='r')
            if fileinp.name.endswith('.jpg'):
                img = Image.open(fileinp)
                #x = np.clip(np.asarray(img, dtype=float) / 255, 0, 1)
                img_np = (np.array(img))
                
                #pdb.set_trace()
                grey = img_np.flatten()
                
                out = np.array(list(grey),np.uint8)
                # Load dictionary object
                if i.startswith('bg_fg_mask_1') :
                    #f_bgfgM.append(i)
                    bgfgM.append(out)
                
                
    #dict_p['f_bgfg'] = f_bgfg
    dict_p['bgfg'] = bgfg
    
    #dict_p['f_depth'] = f_depth
    dict_p['depth'] = depth
    
    #dict_p['f_bgfgM'] = f_bgfgM
    dict_p['bgfgM']     = bgfgM

    #print(dict_p)
    # Create pickle file
    import bz2
    with bz2.BZ2File(pklFileName, 'w',compresslevel=9) as p:
        pickle.dump(dict_p,p,protocol=-1)
    #with open(pklFileName,'wb') as p:
    #    pickle.dump(dict_p,p,protocol=-1))
        
    zipObj.close()
          
                
def unpickle(file=args.output):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
    
if __name__ == '__main__':
    # cmd-1: Pickle create:
    #pdb.set_trace()
    pickle_data_batch()
    data = unpickle()
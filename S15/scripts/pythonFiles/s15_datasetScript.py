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

'''
if args.start:
    print("given start value is : {}".format(args.start))
if args.end:
	print("given end value is : {}".format(args.end))
	print('Expected number of Images are : {}'.format((args.end-args.start)*20*100))
'''

#sys.path.append(str(homepath))

if args.op==1:
	print('You selected an option to recreate the directories')
	try:
		if os.path.exists(os.path.join(homepath,'bg_fg')):
			shutil.rmtree(os.path.join(homepath,'bg_fg'))
			os.mkdir(os.path.join(homepath,'bg_fg'))
			
		if not os.path.exists(os.path.join(homepath,'bg_fg')):
			os.mkdir(os.path.join(homepath,'bg_fg'))
			
		if os.path.exists(os.path.join(homepath,'bg_fgFlip')):
			shutil.rmtree(os.path.join(homepath,'bg_fgFlip'))
			os.mkdir(os.path.join(homepath,'bg_fgFlip'))
		if not os.path.exists(os.path.join(homepath,'bg_fgFlip')):
			os.mkdir(os.path.join(homepath,'bg_fgFlip'))
			
		if os.path.exists(os.path.join(homepath,'bg_fg_mask')):
			shutil.rmtree(os.path.join(homepath,'bg_fg_mask'))
			os.mkdir(os.path.join(homepath,'bg_fg_mask'))
		if not os.path.exists(os.path.join(homepath,'bg_fg_mask')):
			os.mkdir(os.path.join(homepath,'bg_fg_mask'))
			
		if os.path.exists(os.path.join(homepath,'bg_fgFlip_mask')):
			shutil.rmtree(os.path.join(homepath,'bg_fgFlip_mask'))
			os.mkdir(os.path.join(homepath,'bg_fgFlip_mask'))
		if not os.path.exists(os.path.join(homepath,'bg_fgFlip_mask')):
			os.mkdir(os.path.join(homepath,'bg_fgFlip_mask'))

	except Exception as e:
		print(str(e))
		
	time.sleep(3)
#pdb.set_trace()

def overLayImagesInBatch(bgStart,bgEnd,bf=0,bfm=0,bff=0,bffm=0,input=None,outputZip=None):
    #homepath = Path(inputP)
    print('Output is generating at : {}'.format(outputZip))
    gc.enable()
    gc.garbage
    start_time = datetime.now()
    black = np.zeros((224,224))
    try:
        with ZipFile(outputZip,'a',ZIP_DEFLATED) as z:
            
            for i in range(bgStart,bgEnd): 
                start = 1 
                print('Using background : {}'.format(i))
                bgi =f'{homepath}/bg/{str(i)}_bg.jpg'
                bg 	= Image.open(f'{homepath}/bg/{str(i)}_bg.jpg')
                for j in  range(1,101): 
                    fg 		= Image.open(f'{homepath}/fg/{str(j)}_fg.jpg')
                    fgi 	= f'{homepath}/fg/{str(j)}_fg.jpg'
                    m 		= Image.open(f'{homepath}/mask/{str(j)}_fg_mask.jpg')
                    mi 		= f'{homepath}/mask/{str(j)}_fg_mask.jpg'
                    for k in range(1,21): 
                        r1 = random.randint(1, bg.size[0]-fg.size[0])
                        r2 = random.randint(1, bg.size[0]-fg.size[0])

                        bg_1_1, fg_t, m_t = copy.deepcopy(bg),copy.deepcopy(fg),copy.deepcopy(m)
                        bg_1_2, fg_f, m_f = copy.deepcopy(bg),copy.deepcopy(fg),copy.deepcopy(m)

                        fg_f = fg_f.transpose(Image.FLIP_LEFT_RIGHT) #flip fg image
                        m_f	 = m_f.transpose(Image.FLIP_LEFT_RIGHT) #flip mask
                        
                        
                        fname = os.path.basename(bgi).split('.')[0]+'_'+str(start)+'_fg.jpg'
                        if bfm == 1:
                            black_img1 	= Image.fromarray(black,mode='L')
                            black_img1.paste(m_t,(r1,r2), m_t.split()[0])
                            #fname = os.path.basename(bgi).split('.')[0]+'_'+str(start)+'_fg.jpg'
                            #black_img1.save(os.path.join(homepath,'bg_fg_mask',fname), optimize=True, quality=60)
                            black_img1.save('tmp_bg_fg_mask.jpg', optimize=True, quality=60)
                            z.write('tmp_bg_fg_mask.jpg',os.path.join('bg_fg_mask_1',fname))
                        if bffm ==1:
                            black_img2 = Image.fromarray(black,mode='L')
                            black_img2.paste(m_f,(r1,r2), m_f.split()[0])
                            #fname = os.path.basename(bgi).split('.')[0]+'_'+str(start)+'_fg.jpg'
                            #black_img2.save(os.path.join(homepath,'bg_fgFlip_mask',fname), optimize=True, quality=60)
                            black_img2.save('tmp_bg_fgFlip_mask.jpg', optimize=True, quality=60)
                            z.write('tmp_bg_fgFlip_mask.jpg',os.path.join('bg_fgFlip_mask_1',fname))

                        if bf == 1:
                            bg_1_1.paste(fg_t,(r1,r2),m_t.split()[0])
                            #fname = os.path.basename(bgi).split('.')[0]+'_'+str(start)+'_fg.jpg'
                            bg_1_1.save('tmp_bg_fg.jpg', optimize=True, quality=60)
                            z.write('tmp_bg_fg.jpg',os.path.join('bg_fg_1',fname))
                        if bff == 1:
                            bg_1_2.paste(fg_f,(r1,r2),m_f.split()[0])
                            #fname = os.path.basename(bgi).split('.')[0]+'_'+str(start)+'_fg.jpg'
                            #fname = os.path.basename(bgi).split('.')[0]+'_'+str(start)+'_fgFlip.jpg'
                            #bg_1_2.save(os.path.join(homepath,'bg_fgFlip',fname), optimize=True, quality=60)
                            bg_1_2.save('tmp_bg_fgFlip.jpg', optimize=True, quality=60)
                            z.write('tmp_bg_fgFlip.jpg',os.path.join('bg_fgFlip_1',fname))

                        [print(start) if start/500 in range(1000) else None]
                        start+=1
        with ZipFile(outputZip,'a') as z:
            labelsList = sorted(set(filter(lambda x:x.startswith('bg_fg_1/'),z.namelist())))
            with open('labels.txt','w') as f:
                for ele in labelsList:
                    f.write(os.path.basename(ele)+'\n')
            z.write('labels.txt','labels.txt')
    except Exception as e:
        print(str(e))
        print('Fg img : {} and BG Image : {}'.format(j,i))
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))
    
    
def createBinFileWithNdArrays(input=None,outputBinP=None):
    print('Input Directory is processing all jpg files : {}'.format(input))
    print('Output bin file is getting generated at : {}'.format(outputBinP))
    fileCount = 0
    if input is not None:
        input = Path(input)
        filenames = input.glob('*\*.jpg')
        #pdb.set_trace()
        #pprint(sorted(filenames))
        with open(outputBinP,'a') as f:
            #f = open(outputBinP,'a')
            for im in sorted(filenames):
                if im.is_file():
                    #print('Image name is : {} '.format(os.path.basename(im)))
                    tmp = os.path.basename(im).split('_')
                    label = float('{}.{}'.format(tmp[0],tmp[2]))
                    #print(label)
                    im_P = Image.open(im)
                    im_N = (np.array(im_P))

                    r = im_N[:,:,0].flatten()
                    g = im_N[:,:,1].flatten()
                    b = im_N[:,:,2].flatten()
                    label = [label]
                    out = np.array(list(label) + list(r) + list(g) + list(b),np.uint8)
                    #out = np.array(list(r) + list(g) + list(b),np.uint8)
                    fileCount +=1
                if fileCount/1000 in range(0,1000):
                    print('Files processed till now : %i'%fileCount)
                    out.tofile(f)
                    del out
                gc.enable()
                gc.get_threshold()
                gc.collect()
    print('Total number of files processed, exists in bin file is : %i'%fileCount)
                    
def readBinFileFromNdArray(noImgs,file='testOutput.bin'):
    dat = np.fromfile(file,dtype='uint8')
    dat.shape
    dat.size
    dat.resize(int(noImgs),3,224,224)
    im_rgb = dat.transpose(0,2,3,1)

    im_p = Image.fromarray(im_rgb[0],'RGB')
    im_p.show()

    im_p = Image.fromarray(im_rgb[1],'RGB')
    im_p.show()


    
def pickle_data_batch(zipfilename,rstart=args.start,rend=args.end):
    gc.enable()
    gc.get_threshold()
    # pickle dict - keys : batchlabel , fName, bgfg , bgfgM, bgfgF, bgfgFM
    dict_p = {}
    f_bgfg,f_bgfgF, f_bgfgM, bgfg_L, bgfgF_L,bgfgM_L = [],[],[],[],[],[]
    dict_p['batchlabel'] = 'batch_range_%i'%(rend-rstart)
    
    loaded_images,imgName = [],[]
    zipObj = ZipFile(zipfilename, 'r')
    files = zipObj.namelist()
    for i in files[rstart:rend]:
        fileinp = zipObj.open(i, mode='r')
        if i.startswith('bg_fg_1') or i.startswith('bg_fgFlip_1'):
            if fileinp.name.endswith('.jpg'):
                img = Image.open(fileinp)
                #x = np.clip(np.asarray(img, dtype=float) / 255, 0, 1)
                img_np = (np.array(img))
                
                r = img_np[:,:,0].flatten()
                g = img_np[:,:,1].flatten()
                b = img_np[:,:,2].flatten()
                
                out = np.array(list(r) + list(g) + list(b),np.uint8)
                # Load dictionary object
                if i.startswith('bg_fg_1') :
                    f_bgfg.append(i)
                    bgfg_L.append(out)
                elif i.startswith('bg_fgFlip_1') :
                    f_bgfgF.append(i)
                    bgfgF_L.append(out)
                    
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
                    f_bgfgM.append(i)
                    bgfgM_L.append(out)
                
                
    dict_p['f_bgfg'] = f_bgfg
    dict_p['bgfg'] = bgfg_L
    
    dict_p['f_bgfgF'] = f_bgfgF
    dict_p['bgfgF'] = bgfgF_L
    
    dict_p['f_bgfgM'] = f_bgfgM
    dict_p['bgfgM']     = bgfgM_L

    #print(dict_p)
    # Create pickle file
    with open(os.path.join(os.getcwd(),'data_batch_64_test'),'wb') as p:
        pickle.dump(dict_p,p)
        
    zipObj.close()
          
                
def unpickle(file=r'C:\Users\skanukollu\Documents\cnn_suman\customdataset_Notes\data_batch_64_test'):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
    
if __name__=='__main__':
    # Input path
    #inputP   = args.input
    print(args.start,args.end)
    inputP   = r'/content/drive/My Drive/EVA4/S15/dataset'
    homepath = Path(inputP)
    #pdb.set_trace()
    # Step-1 : Generate Overlay and mask images
    overLayImagesInBatch(bgStart= args.start, 
                        bgEnd   = args.end,
                        bf      = args.bg_fg_1,
                        bfm     = args.bg_fg_mask_1,
                        bff     = args.bg_fgFlip_1,
                        bffm    = args.bg_fgFlip_mask_1,
                        input   = homepath, 
                        #outputZip = os.path.join(inputP,'zipFiles',args.output)
                        outputZip = args.output
                    ) 
                    
    # Step-2 : Generate Depth maps:
    
    
    

# import stuff

import argparse
import os, glob

import numpy as np
import matplotlib.pyplot as plt

import openslide
import cv2
import h5py

import json
from datetime import datetime

from scipy.ndimage.morphology import binary_fill_holes, binary_opening, binary_dilation, binary_erosion
from skimage.color import rgb2gray
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu as otsu
from skimage.morphology import disk
from skimage.transform import resize
from sklearn.utils import shuffle
from skimage import feature

class Extractor:

    def __init__(self, fileName=None, fileExtension=None, filePath=None, kidneyIdentifier=0):        
        ''' initialize '''

        self.fileName = fileName
        self.filePath = filePath
        self.kidneyIdentifier = kidneyIdentifier
        # load WSI
        self.load_slide(fileExtension)

    def load_slide(self, fileExtension=None):
        ''' Read WSI in lowest resolution 
            and initialize all parameters
        '''

        self.file = os.path.join(self.filePath, self.fileName + fileExtension)
        self.slide = openslide.OpenSlide(self.file)    
        self.wsi = self.slide.read_region((0,0), 
                                self.slide.level_count-1, 
                                self.slide.level_dimensions[-1])

        # IMPORTANT: .svs self.files have a downsampling level of 4 while for .ndpi it is 2.
        self.slide_width, self.slide_height = self.slide.dimensions
        self.nrLevels = self.slide.level_count
        self.nrDimensions = self.slide.level_dimensions
        self.downscaleFactor = int(self.slide.level_downsamples[1])
        # scale to highest resolution
        self.scaleToHighestLevel = self.downscaleFactor**(self.slide.level_count-1) 
        # patch size
        self.patchSize = 500
        self.highestOffset = 50
        # hdf5File
        self.hdf5File = os.path.join(self.filePath,self.fileName + '.hdf5')

    def __repr__(self): # print
        return "Extractor! Using the file path {} with file name {}".format(self.filePath, self.fileName)

    def check_level(self, level=0):
        ''' make sure user inputs valid 'level' '''
        try: 
            assert 0 <= level <= self.slide.level_count-1, 'Unacceptable input: level'
            level = level 
        except: 
            print('level must be between 0 (highest) and ', self.slide.level_count-1, '(lowest).')
            raise
        
    def check_kidneyIdentifier(self, kidneyIdentifier=0):
        ''' make sure user inputs valid kidneyIdentifier '''
        try:
            assert 0 <= kidneyIdentifier <= self.nr_kidneys, 'Unacceptable kidney choice'
            self.kidneyIdentifier = kidneyIdentifier
        except:
            print('kidneyIdentifier must be between 0 (upper kidney) and ', self.nr_kidneys, '(lower kidney).')
            raise

    def check_nr_patches(self, nrPatches=None):
        ''' make sure user inputs valid number of patches '''
        try:
            isinstance(nrPatches, int)
        except ValueError: 
            print('Number of patches must be an integer value!')
            raise 

    def get_params(self, level=0, offset=0):
        ''' get parameters required to scale variable to desired level '''
        
        # scale to level specified
        scaleToLevel = self.downscaleFactor**((self.slide.level_count-1)-level) 
        # offset = 50 in level 0
        offset = int(np.ceil(offset/self.downscaleFactor**(level)))

        return scaleToLevel, offset

    def write_json(self, BB=[], poly=None, level=0, outputPath=None):
        ''' Open one json file for each WSI.
            Dump info from all glom on this WSI. 
            Close json.
        '''

        # write the coords for each WSI in a json file 
        json_data = {
            "labels": {
                "glom": [255, 0, 0]
            }, 
            "annotations": []
        }

        # idx = 0
        p = []
        
        # for row in rows:
        #     for col in cols:
        #         BB = [int(row), int(col), int(height), int(height)]
        for nrObj, bb in enumerate(BB):
                if poly is not None:
                    p = poly[nrObj]
                json_data["annotations"].append(
                    {
                        "bbox":bb,
                        "timestamp": datetime.today().isoformat('@'),
                        "id": datetime.today().isoformat('@'), # unique Id for annotation
                        "uid": 'unet', # user-ID
                        "label": 'glom',
                        "color": [255, 0, 0],
                        "remarks": '', #eg. strong/weak annotation
                        "poly": p
                    })
                # idx += 1

        with open(os.path.join(outputPath, self.fileName+'.json'), 'w') as outfile:
            json.dump(json_data, outfile)

    def visualize_extraction(self):
        pass

    def extract_kidney_mask(self, outputPath=None):
        ''' Extract kidney mask in smallest resolution ''' 

        # Gaussian blur
        wsi_blur = gaussian(np.uint8(self.wsi),sigma=1)
        # gray level
        kidneyMask = rgb2gray(wsi_blur)
        # binary by OTSU thresholding
        thresh = otsu(kidneyMask, nbins=256)
        kidneyMask = kidneyMask <= thresh
        # morphological operations
        kidneyMask = binary_fill_holes(kidneyMask)
        kidneyMask = binary_opening(kidneyMask, structure=np.ones((3,3)), iterations=3)

        # Get biggest area(s)
        labeledImg = label(kidneyMask)
        props = regionprops(labeledImg)

        all_areas = []
        for i, _ in enumerate(props):
            all_areas.append([props[i]['area'],i])
        all_areas.sort(reverse=True)

        # Deciding if there are one or two kidneys 
        '''
        First extract the biggest area. 
        If:
            second biggest area is at least as big as half the first biggest area, 
                this WSI shows two kidneys.
            Extract mask for two kidneys.
        Else:
            Extract only one mask.
        '''
        kidney_areas = []
        ind = []

        if len(all_areas) == 1:
            kidney_areas.extend([all_areas[0][0]])
            ind.extend([all_areas[0][1]])
        elif all_areas[1][0] >= int(all_areas[0][0]*.5): # bigger than 50% biggest area
            kidney_areas.extend([all_areas[0][0], all_areas[1][0]])
            ind.extend([all_areas[0][1], all_areas[1][1]])
        else: 
            kidney_areas.extend([all_areas[0][0]])
            ind.extend([all_areas[0][1]])

        # Delete all small blobs from kidney mask
        for i, _ in enumerate(props):
            if i not in ind:
                coord = props[i]['bbox']
                kidneyMask[coord[0]:coord[2], coord[1]:coord[3]] = 0  

        ''' If two kidneys, kidneyMask is a 3-channel image '''

        labeledImg = label(kidneyMask)
        props = regionprops(labeledImg)

        _, offset = self.get_params(self.slide.level_count-1, self.highestOffset)

        self.all_coord = []
        if len(np.unique(labeledImg)) == 3:
            self.nr_kidneys = 2
            tempMask = np.zeros((np.shape(labeledImg)[0],np.shape(labeledImg)[1],3))
            for i, region in enumerate(props):
                coord = region['bbox']
                # To make sure all kidney tissue is covered while extracting patches
                coord = (max(coord[0]-offset,0), max(coord[1]-offset,0),
                            min(coord[2]+offset, self.slide.level_dimensions[-1][1]), min(coord[3]+offset, self.slide.level_dimensions[-1][0])) 
                self.all_coord.append(coord)
                tempMask[:,:,i][coord[0]:coord[2], coord[1]:coord[3]] = kidneyMask[coord[0]:coord[2],
                                                                                    coord[1]:coord[3]]

            kidneyMask = tempMask    
        else:
            coord = props[0]['bbox']
            # To make sure all kidney tissue is covered while extracting patches
            coord = (max(coord[0]-offset,0), max(coord[1]-offset,0),
                            min(coord[2]+offset, self.slide.level_dimensions[-1][1]), min(coord[3]+offset, self.slide.level_dimensions[-1][0])) 
            self.all_coord = coord
            self.nr_kidneys = 1

        ''' Important for assigning kidneyIdentifier: 
            Upper/Left kidney = kidneyMask[:,:,0]
            Lower/Right kidney = kidneyMask[:,:,1] '''

        if self.nr_kidneys > 1:
            if self.all_coord[0][0] >= self.all_coord[1][0] or self.all_coord[0][2] >= self.all_coord[1][2]:
                # swap coords
                self.all_coord[0], self.all_coord[1] = self.all_coord[1], self.all_coord[0]
                # swap kidney images
                temp = np.zeros_like(kidneyMask[:,:,0])
                temp = kidneyMask[:,:,0].copy()
                kidneyMask[:,:,0] = kidneyMask[:,:,1]
                kidneyMask[:,:,1] = temp
        
        # make sure user inputs valid kidneyIdentifier
        self.check_kidneyIdentifier(self.kidneyIdentifier) # (self, kidneyIdentifier=self.kidneyIdentifier)
        # plt.imsave(os.path.join(outputPath, self.fileName + '_' + str(self.kidneyIdentifier) + '.jpg'), kidneyMask) 

        return kidneyMask, self.all_coord, self.nr_kidneys

    def get_relevant_kidney(self, level=5):
        ''' Returns selected kidney segmented in desired level.
            Also returns the scaled coordinated (BB) for the selected kidney.
        '''

        self.extract_kidney_mask()
        # make sure user inputs valid 'level'
        self.check_level(level) #(self, level=level)

        if self.nr_kidneys == 1:
            self.kidneyIdentifier = 0 
            coord = self.all_coord
        else:
            # kidneyIdentifier = kidneyIdentifier
            coord = self.all_coord[self.kidneyIdentifier]

        scaleToLevel, offset = self.get_params(level, self.highestOffset)

        newCoord = tuple([c*scaleToLevel for c in coord])

        if not level:
            ''' do not display kidney image for highest level- too much memory demanding  '''
            kidney = []
        else:
            # kidney = self.slide.read_region(((coord[1]-offset)*self.scaleToHighestLevel,
            #                                 (coord[0]-offset)*self.scaleToHighestLevel),
            #                                 level,
            #                                 (newCoord[3]-newCoord[1]+offset*2*scaleToLevel,
            #                                 newCoord[2]-newCoord[0]+2*offset*scaleToLevel))

            kidney = self.slide.read_region((coord[1]*self.scaleToHighestLevel-self.highestOffset,
                                            coord[0]*self.scaleToHighestLevel-self.highestOffset),
                                            level,
                                            (newCoord[3]-newCoord[1]+2*offset,
                                            newCoord[2]-newCoord[0]+2*offset))
        # coord is first row, then col
        return kidney, newCoord 

    def get_annotations(self, BB=False):
        ''' read annotations from HDF5 file 
            Note: If BB = False, we get perfect contour masks
            Important: necessary to get full mask first to make sure we get all the labeled glomeruli within a given area
        '''
        global hf, data
        hf = h5py.File(self.hdf5File, 'r')
        data = hf.get('annotations')

        _ , highestCoords = self.get_relevant_kidney(0) 

        rects = []
        masks = []
        self.nrGlom = 0
        self.fullMask = np.zeros((self.slide_height, self.slide_width))

        for item in data:

            annotation = data.__getitem__(item)
            # currentLabel = annotation.attrs.__getitem__('label')
            rect = np.array(annotation.__getitem__('boundingRect'), 'int')

            if highestCoords[0] <= rect[1]+rect[3] <= highestCoords[2] and \
                highestCoords[1] <= rect[0]+rect[2]<= highestCoords[3]: 
                # extracting glom masks from only given kidney
                self.nrGlom += 1
                if BB:
                    self.fullMask[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1] = 1
                else:
                    mask = np.array(annotation.__getitem__('mask')) / 255
                    self.fullMask[rect[1]:rect[1]+rect[3]+1, rect[0]:rect[0]+rect[2]+1] = mask

    def make_3channel_mask(self, img):
        # all masks are saved as 3 channel uint8 images (.png):
        # 1. Channel one: edge
        # 2. Channel two: object as 1, bg as 0
        # 3. Channel three: bg as 1, object as 0
        img2  = img<0.5
        mask = np.zeros((self.patchSize, self.patchSize, 3))
        mask[:,:,0] = binary_dilation(feature.canny(img2), structure=disk(1), iterations=1)
        mask[:,:,1] = 1-img2
        mask[:,:,1][np.where(mask[:,:,0] + mask[:,:,1] == 2)] = 0
        mask[:,:,2] = 1 - mask[:,:,0] - mask[:,:,1]

        return mask

    def get_training_data(self, level=1, withinKidney=True, BB=False, nrPatches=None,  outputPath=None):
        ''' 1. Get patches with glomeruli centered + corresponding masks
            2. Get random patches with or without glomeruli + corresponding masks
            3. Default: Patches are extracted in the second highest resolution => level=1 
        '''

        # intialize
        self.get_annotations()

        # make sure user inputs valid number of patches
        self.check_nr_patches(nrPatches=nrPatches)

        # make sure user inputs valid 'level' number
        self.check_level(level=level)

        # make two directories for glom-positive and random patches
        glomDir = os.path.join(outputPath, 'glom_patches')
        if not os.path.exists(glomDir):
            os.makedirs(glomDir, exist_ok=True)
        randomDir = os.path.join(outputPath, 'random_patches')
        if not os.path.exists(randomDir):
            os.makedirs(randomDir, exist_ok=True)

        # 2. glomeruli containing patches: 
        #---------------------------------
        # downscaled patches and masks

        # extract all glom patches if nrPatches is NOT specified
        maxNrPatches = self.nrGlom   
        if nrPatches is None:
            nrPatches = maxNrPatches

        count = 0 # counter for number of glomeruli patches to be extracted
        patch_coord_row = []
        patch_coord_col = []
        poly = []
        all_poly = []
        all_bb = []

        if self.fileName[:4] == '2023':
            a=1 
        itemNr = 0

        _ , highestCoords = self.get_relevant_kidney(0) 
        for item in data:

            itemNr += 1
            if itemNr == 55:
                a = 1
            annotation = data.__getitem__(item)
            rect = np.array(annotation.__getitem__('boundingRect'), 'int')

            if highestCoords[0] <= rect[1]+rect[3] <= highestCoords[2] and \
                highestCoords[1] <= rect[0]+rect[2]<= highestCoords[3]:

                count += 1
                
                start_x = rect[0]-int((self.downscaleFactor**level*self.patchSize-rect[2])/2)
                start_y = rect[1]-int((self.downscaleFactor**level*self.patchSize-rect[3])/2)
                # extract glomeruli containing patches; glomeruli centered
                patch = np.array(self.slide.read_region( (start_x, start_y), level, (self.patchSize,self.patchSize) ), 'uint8')[:,:,[0,1,2]]
                patch_coord_row.append(start_y)
                patch_coord_col.append(start_x)
                plt.imsave(os.path.join(glomDir, str(count) + '_patch' + '.png'), patch)  
                patchMask = self.fullMask[start_y: start_y+self.downscaleFactor**level*self.patchSize, start_x: start_x+self.downscaleFactor**level*self.patchSize]

                # contours, _ = cv2.findContours(np.uint8(patchMask),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                # for contour in contours:
                #     bb = cv2.boundingRect(contour)
                #     all_bb.append(list(bb))
                #     for coords in contour:
                #         poly.append([coords[0][0], coords[0][1]])
                #     all_poly.append(list(poly))

                if np.shape(patchMask) != (1000, 1000):
                    a=1
                patchMask = resize(patchMask, (self.patchSize,self.patchSize))
                patchMask = self.make_3channel_mask(patchMask)
                plt.imsave(os.path.join(glomDir, str(count) + '_mask' + '.png'), patchMask) 
                # if nrGlom > nrPatches: break
                if count == nrPatches: #TODO: check why nrPatches is not = self.nrGlom
                    break
            else: 
                continue # only executed if the inner loop did NOT break

        # # write the coords for each WSI in a json file # TODO: stop at len(number of random patches)
        # self.write_json(rows=patch_coord_row, cols=patch_coord_col, height=self.patchSize, poly=poly, level=0, outputPath=glomDir)

        # 3. random patches: 
        #---------------------------------

        patch_coord_row = []
        patch_coord_col = []
        overlap = int(np.ceil(0.1*self.patchSize)) # overlap = 10% of patch size 

        if not withinKidney:
            # from all over the WSI
            patch_coord_row = np.arange(0, self.slide.level_dimensions[0][1]-self.patchSize*self.downscaleFactor,
                                        (self.patchSize-overlap)*self.downscaleFactor)
            patch_coord_col = np.arange(0, self.slide.level_dimensions[0][0]-self.patchSize*self.downscaleFactor, 
                                        (self.patchSize-overlap)*self.downscaleFactor)
        else:
            # from within the given kidney BB
            patch_coord_col = np.arange(highestCoords[1], highestCoords[3]-self.patchSize*self.downscaleFactor, 
                                        (self.patchSize-overlap)*self.downscaleFactor)  
            patch_coord_row = np.arange(highestCoords[0], highestCoords[2]-self.patchSize*self.downscaleFactor, 
                                        (self.patchSize-overlap)*self.downscaleFactor)  

        countR = 0 # counter for number of random patches
        # to get random patches each time, shuffle 
        patch_coord_row = shuffle(patch_coord_row[:-1])
        patch_coord_col = shuffle(patch_coord_col[:-1])

        nrRandomPatches = np.ceil(nrPatches/2)

        # maxNrPatches = (len(patch_coord_row) -1) * (len(patch_coord_col) -1)    
        # if nrPatches is None:
        #     nrPatches = maxNrPatches/2
        # else:
        #     nrPatches = nrPatches/2 # nrRandomPatches = nrPatches/2 with glomeruli

        for row in patch_coord_row: #[:-1]:
            for col in patch_coord_col: #[:-1]:
                countR += 1
                # extract patches 
                patch = np.array(self.slide.read_region((col,row), level, (self.patchSize,self.patchSize)), 'uint8')[:,:,[0,1,2]]
                plt.imsave(os.path.join(randomDir, str(countR) + '_rpatch' + '.png'), patch)   
                # extract corresponding masks
                patchMask = self.fullMask[row: row+self.downscaleFactor**level*self.patchSize, col: col+self.downscaleFactor**level*self.patchSize]
                if np.shape(patchMask) != (1000, 1000):
                    a=1
                patchMask = resize(patchMask, (self.patchSize, self.patchSize))
                patchMask = self.make_3channel_mask(patchMask)
                plt.imsave(os.path.join(randomDir, str(countR) + '_rmask' + '.png'), patchMask) 
                if countR == nrRandomPatches:
                    break
            else: 
                continue # only executed if the inner loop did NOT break
            break # only executed if the inner loop DID break

    def get_only_patches(self, level=1, withinKidney=True, outputPath=None):
        ''' 1. Grid like patch extraction at fixed intervals (patchSize = 492 here)
            2. Default: Patches are extracted in the second highest resolution => level=1 
        '''

        # make sure user inputs valid 'level' number
        self.check_level(level=level)

        patchSize = 492

        patchesDir = os.path.join(outputPath, 'all_patches')
        if not os.path.exists(patchesDir):
            os.makedirs(patchesDir, exist_ok=True)

        _ , highestCoords = self.get_relevant_kidney(0) 

        overlap = 0# int(np.ceil(0.1*self.patchSize)) # overlap = 10% of patch size 

        if not withinKidney:
            # from all over the WSI
            patch_coord_row = np.arange(0, self.slide.level_dimensions[0][1]-patchSize*self.downscaleFactor,
                                        (patchSize-overlap)*self.downscaleFactor)
            patch_coord_col = np.arange(0, self.slide.level_dimensions[0][0]-patchSize*self.downscaleFactor, 
                                        (patchSize-overlap)*self.downscaleFactor)
        else:
            # from within the given kidney BB
            patch_coord_col = np.arange(highestCoords[1], highestCoords[3]-patchSize*self.downscaleFactor, 
                                        (patchSize-overlap)*self.downscaleFactor)  
            patch_coord_row = np.arange(highestCoords[0], highestCoords[2]-patchSize*self.downscaleFactor, 
                                        (patchSize-overlap)*self.downscaleFactor)  

        count = 0 # counter for number of patches
        distance = patchSize - overlap
        patchBB = []

        for row in patch_coord_row[:-1]:
            for col in patch_coord_col[:-1]:
                count += 1
                # extract patches 
                patch = np.array(self.slide.read_region((col,row), level, (patchSize,patchSize)), 'uint8')[:,:,[0,1,2]]
                patchBB.append(list((int(row), int(col), int(distance), int(distance))))
                plt.imsave(os.path.join(patchesDir, str(count) + '_patch' + '.png'), patch)   

        # write the coords for each WSI in a json file 
        self.write_json(BB=patchBB, poly=None, level=0, outputPath=outputPath)

    def get_medulla_cortex(self):

        # Extract cortex-medulla
        hf = h5py.File(self.hdf5File, 'r')
        data = hf.get('annotations')

        _ , highestCoords = self.get_relevant_kidney(0) 

        if self.nrLevels >= 3:
            level = 3 # check if this level is available or not for the given WSI # TODO: check if this is correct
        else:
            level = self.nrLevels # else take the lowest available level

        BB = False
        rects = []
        masks = []

        # fullMask = np.zeros((slide.level_dimensions[0]))
        halfMask = np.zeros((self.slide.level_dimensions[level]))
        for item in data:

            annotation = data.__getitem__(item)
            # currentLabel = annotation.attrs.__getitem__('label')
            rect = np.array(annotation.__getitem__('boundingRect'), 'int')

            if highestCoords[0] <= rect[1]+rect[3] <= highestCoords[2] and \
                highestCoords[1] <= rect[0]+rect[2]<= highestCoords[3]:
                
                resizedRect = [int(r/self.downscaleFactor**level) for r in rect]
                mask = np.array(annotation.__getitem__('mask')) / 255
                smallerMask = np.resize(mask, (int(rect[3]/self.downscaleFactor**level),int(rect[2]/self.downscaleFactor**level)))
                halfMask[resizedRect[1]:resizedRect[1]+resizedRect[3],
                            resizedRect[0]:resizedRect[0]+resizedRect[2]] = smallerMask

        # extract BB 
        scaleToLevel, offset = self.get_params(level, self.highestOffset)
        # offset = offset*scaleToLevel
        kidneyMask, rescaledCoords = self.get_relevant_kidney(level)
        halfMask = halfMask[rescaledCoords[0]-offset:rescaledCoords[2]+offset, rescaledCoords[1]-offset:rescaledCoords[3]+offset]

        struct = disk(3)
        iter = 55
        newMask = binary_erosion(binary_dilation(halfMask, structure=struct, iterations=iter), structure=struct, iterations=iter)
        plt.imshow(newMask)
        plt.show()

        contours, _ = cv2.findContours(np.uint8(newMask),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contourImg = cv2.drawContours(np.uint8(newMask), contours, -1, (128,255,0), -1)
        contourImgOrig = contourImg.copy()

        plt.figure()
        plt.imshow(contourImg)
        plt.show()


        hull = cv2.convexHull(contours[0])
        hullImg = cv2.drawContours(np.uint8(contourImg), [hull], 0, (255,255,0), -1)
        plt.figure()
        plt.imshow(hullImg)
        plt.show()


        hullImg = hullImg>125
        contourImgOrig = contourImgOrig>125
        diffImg = hullImg^contourImgOrig
        if np.sum(diffImg) < np.sum(hullImg)/5:
            diffImg = np.copy(hullImg)
            medullaIdentifier = False
        else:
            diffImg = binary_erosion(diffImg, structure=disk(3), iterations=5)
            medullaIdentifier = True

        # Keep only the biggest area or the one with the centroid
        labeledImg = label(diffImg)
        unique, counts = np.unique(labeledImg, return_counts=True)
        list_seg = list(zip(unique, counts))[1:] # ignore label=0=background
        largest = max(list_seg, key=lambda x:x[1])[0]
        largest_blob = (labeledImg == largest).astype('uint8')

        plt.imshow(largest_blob)
        plt.show()

        # get kidney
        kidneyMask = np.uint8(kidneyMask)
        kidneyMask = kidneyMask[:,:,0:3]

        # get cortex
        cortex = np.zeros_like(kidneyMask)
        # plt.imshow(thisCortex)
        cortex[:,:,0] = kidneyMask[:,:,0]*largest_blob
        cortex[:,:,1] = kidneyMask[:,:,1]*largest_blob
        cortex[:,:,2] = kidneyMask[:,:,2]*largest_blob
        # plt.imshow(cortex)
        # plt.imsave('data/imgs2/' + '_cortex' + '.png', cortex) 


        if medullaIdentifier:
            # get kidney mask
            thisMedulla = cv2.resize(kidneyMask, (largest_blob.shape[1], largest_blob.shape[0]))

            # kidney hull
            filledThisMedulla = np.copy(thisMedulla[:,:,0])
            contoursMedulla, _ = cv2.findContours(np.uint8(filledThisMedulla),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contourImgMedulla = cv2.drawContours(np.uint8(filledThisMedulla), contoursMedulla, -1, (128,255,0), -1)
            hullMedulla = cv2.convexHull(contoursMedulla[0])
            hullImgMedulla = cv2.drawContours(np.uint8(contourImgMedulla), [hullMedulla], 0, (255,255,0), -1)

            # get medulla
            diffImg = cv2.bitwise_and(hullImgMedulla, (1-largest_blob))
            medulla = np.zeros_like(kidneyMask)
            medulla[:,:,0] = kidneyMask[:,:,0]*(diffImg)
            medulla[:,:,1] = kidneyMask[:,:,1]*(diffImg)
            medulla[:,:,2] = kidneyMask[:,:,2]*(diffImg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Laxmi's Patch Extractor")
    parser.add_argument('--output_root_dir', type=str, default='/work/scratch/gupta/Patches/Dennis_Data_1', help='Output path for storing the extracted patches')
    parser.add_argument('--input_root_dir', type=str, default='/images/ACTIVE/2015-04_Boor/DataSetsHisto/Histo2ProgressiveDiseses/Histo2UUO', help='Path for where to find the whole slide images')
    parser.add_argument('--within_kidney', action='store_true', default=False)
    args = parser.parse_args()

    if args.output_root_dir is None:
        args.output_root_dir = args.input_root_dir + '_Patches'

    for file_in_path in glob.glob(args.input_root_dir + '/*'):
        if file_in_path.endswith(('.ndpi', '.svs')):

                fileName, fileExtension = os.path.splitext(file_in_path)
                filePath , fileName = os.path.split(fileName)
                if '2017' or '2007' or '2027' or '2150' or '2179' or '2146' or '2175' in fileName[:4]:
                    pass                
                _, folderName = os.path.split(filePath)


                ex = Extractor(filePath=filePath, fileName=fileName, fileExtension=fileExtension, kidneyIdentifier=1) 

                outputPath = os.path.join(args.output_root_dir, folderName, fileName)
                if not os.path.exists(outputPath):
                    os.makedirs(outputPath, exist_ok=True)

                #ex.get_training_data(outputPath=outputPath)
                #print(ex)
                ex.get_only_patches(withinKidney=False, outputPath=outputPath)
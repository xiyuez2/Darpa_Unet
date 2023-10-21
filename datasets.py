import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import PIL
import os 
import torch.nn.functional as F
import torchvision.transforms.functional as aug_f
import random
import cv2
import json 
from itertools import chain
from patchify import patchify, unpatchify

training_path = "/media/jiahua/FILE/uiuc/NCSA/processed/training"
validation_path = "/media/jiahua/FILE/uiuc/NCSA/processed/validation"

def cut_map(image,patch_dims = (256,256),patch_overlap=32):
    # patch parameter
    patch_dims = (256,256)
    patch_overlap = 32
    patch_step = patch_dims[1]-patch_overlap
    # cut boundary of the map 
    shift_x = (image.shape[0]-patch_dims[0])%patch_step
    shift_y = (image.shape[1]-patch_dims[1])%patch_step
    shift_x_left = shift_x//2
    shift_x_right = shift_x - shift_x_left
    shift_y_left = shift_y//2
    shift_y_right = shift_y - shift_y_left
    # recoder the shift_coord
    shift_coord =  [shift_x_left, shift_x_right, shift_y_left, shift_y_right]
    
    map_im_cut = image[shift_x_left:image.shape[0]-shift_x_right, shift_y_left:image.shape[1]-shift_y_right,:]
    cut_shape = np.shape(map_im_cut)
    map_patchs = patchify(map_im_cut, (*patch_dims,3), patch_step)
    patch_shape = np.shape(map_patchs)
    map_patchs = map_patchs.reshape((-1,patch_dims[0],patch_dims[1],3))
    return map_patchs, shift_coord, cut_shape, patch_shape

class MAPData(data.Dataset):
    '''
    return:
        map_img: map image (3,resize_size,resize_size)
        legend_img: legend image (3,resize_size,resize_size)
        seg_img: segmentation image (3,resize_size,resize_size)
    '''
    def __init__(self, data_path=training_path,type="poly",range=None,resize_size = 256 , train = True):
        self.resize_size = resize_size
        self.train = train
        # if train:
        self.data_transforms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        # transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue=0.5)                        
        ])
        # else:
        #     self.data_transforms = transforms.Compose([
        #     transforms.Resize((resize_size, resize_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225]),
        #     ])
        self.mask_transforms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor()
        ])
        self.root = data_path
        self.type = type
        map_path = os.listdir(os.path.join(self.root,self.type,"map_patches"))
        legend_path = ['_'.join(x.split('_')[0:-2])+'.png' for x in map_path]
        
        if range is not None:
            map_path = map_path[range[0]:range[1]]
            legend_path = legend_path[range[0]:range[1]]
        
        self.map_path = [os.path.join(self.root,self.type,"map_patches",x) for x in map_path]
        self.legend_path = [os.path.join(self.root,self.type,"legend",x) for x in legend_path]
        self.seg_path = [os.path.join(self.root,self.type,"seg_patches",x) for x in map_path]
        
    def __getitem__(self, index):
        map_img = Image.open(self.map_path[index])
        legend_img = Image.open(self.legend_path[index])
        seg_img = Image.open(self.seg_path[index])
        seg_img = torch.tensor(np.array(seg_img).astype(float))
        seg_img = F.interpolate(seg_img.unsqueeze(0).unsqueeze(0), size=(self.resize_size,self.resize_size), mode='bilinear', align_corners=False).squeeze(0)
        seg_img = seg_img > 0.5
        seg_img = seg_img.type(torch.int64)
        # seg_img = seg_img.type(torch.float)

        map_img = self.data_transforms(map_img)
        legend_img = self.data_transforms(legend_img)# .unsqueeze(0)

    
        return {
            "map_img": map_img,    # 3 H W
            "legend_img": legend_img, # 1 3 H W - > 3 H W 
            "GT": seg_img[0], #.type(torch.LongTensor),    # 1 H W
            # "sup_msk": legend_seg, # 1 H W
            # "cls": 1,
            # 'qry_names': ["datasetid"+str(index)],
            'qry_ori_size': torch.tensor((256,256))
        }
    
    def __len__(self):
        return len(self.map_path)

    def sample_tasks(self):
        pass
    def reset_sampler(self):
        pass

class eval_MAPData(data.Dataset):
    '''
    return:
        map_img: map image (3,resize_size,resize_size)
        legend_img: legend image (3,resize_size,resize_size)
        seg_img: segmentation image (3,resize_size,resize_size)
    '''
    def __init__(self, data_path=training_path,type="poly",range=None,resize_size = 256 , mapName = ''):
        self.resize_size = resize_size
        self.train = False # train

        self.data_transforms = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        self.mask_transforms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor()
        ])
        self.root = data_path
        self.type = type 
        # read in map data:
        mapPath = os.path.join(data_path, mapName)
        jsonPath = os.path.join(data_path, mapName[0:-4]+'.json')
        gtPath = data_path + "/../validation_rasters/" + mapName[:-4] #_Xjgb_poly.tif
        
        self.map = cv2.imread(mapPath)

        with open(jsonPath, 'r') as f:
            jsonData = json.load(f)
        
        # only keep polygon label
        # read in legend img
        legend_name = []
        legend_img = []
        legend_gt = []

        print(jsonData)
        for label_dict in jsonData['shapes']:
            if not label_dict['label'].endswith('_poly'):
                continue
            legend_name.append(label_dict['label'])
            gt_file = gtPath + "_" + label_dict['label'] + ".tif"
            gt_im = cv2.imread(gt_file)
            legend_gt.append(gt_im[:,:,0])
            
            point_coord = label_dict['points']
            flatten_list = list(chain.from_iterable(point_coord))
            if point_coord[0][0] >= point_coord[1][0] or point_coord[0][1] >= point_coord[1][1] or (len(flatten_list)!=4):
                x_coord = [x[0] for x in point_coord]
                y_coord = [x[1] for x in point_coord]
                x_low, y_low, x_hi, y_hi = int(min(x_coord)), int(min(y_coord)), int(max(x_coord)), int(max(y_coord))

            else: x_low, y_low, x_hi, y_hi = [int(x) for x in flatten_list]
            legend_coor =  [(x_low, y_low), (x_hi, y_hi)]
            shift_pixel  = 4
            im_crop = self.map[y_low+shift_pixel:y_hi-shift_pixel, x_low+shift_pixel:x_hi-shift_pixel] # need to resize
            im_crop_resize = cv2.resize(im_crop, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
            legend_img.append(im_crop_resize.transpose(0,1,2))

        self.legend_name = legend_name # [:2]
        self.legend_img = legend_img # [:2]
        self.legend_gt = legend_gt # [:2]
        self.map_patches, self.shift_coord, self.cut_shape,self.patch_shape = cut_map(self.map)
        # print("debug: ", np.shape(legend_gt),np.min(legend_gt),np.max(legend_gt))

        # self.w =
        print("when init dataset:", np.shape(legend_name), np.shape(legend_img), np.shape(legend_gt),np.shape(self.map_patches))
        

    def __getitem__(self, index):
        # return {}
        # map_img = Image.open(self.map_path[index])
        # legend_img = Image.open(self.legend_path[index])
        # seg_img = Image.open(self.seg_path[index])
        # map_shape = np.shape(self.map)
        patch_idx = index %  len(self.map_patches)
        
        legend_idx = index // len(self.map_patches)
        
        map_img = self.map_patches[patch_idx]
        legend_img = self.legend_img[legend_idx]
        seg_img = np.zeros((256,256)) #self.legend_gt[patch_idx,legend_idx]

        seg_img = torch.tensor(np.array(seg_img).astype(float))
        seg_img = F.interpolate(seg_img.unsqueeze(0).unsqueeze(0), size=(self.resize_size,self.resize_size), mode='bilinear', align_corners=False).squeeze(0)
        seg_img = seg_img > 0.5
        seg_img = seg_img.type(torch.int64)
        # seg_img = seg_img.type(torch.float)

        # seg_img = (seg_img > 0.0001).type(torch.float)
        # print("debug:", np.shape(map_img),np.shape(legend_img))
        map_img = self.data_transforms(Image.fromarray(map_img))
        legend_img = self.data_transforms(Image.fromarray(legend_img))# .unsqueeze(0)

        legend_seg = seg_img.clone()
        legend_seg = legend_seg * 0 + 1
        # print("mean of msk",torch.mean(seg_img.type(torch.LongTensor).type(torch.float)))
        
        # print("debug:", map_img.size(),legend_img.size(),seg_img.size(),legend_seg.size())
        return {
            "map_img": map_img,    # 3 H W
            "legend_img": legend_img, # 1 3 H W -> 3 H W
            "GT": seg_img[0], #.type(torch.LongTensor),    # 1 H W
            # "sup_msk": legend_seg, # 1 H W
            # "cls": 1,
            'qry_names': ["datasetid"+str(index)],
            'ori_size': torch.tensor((256,256))
        }
    
    def metrics(self,preds):
        
        preds = preds > 0.5
        shape = np.shape(preds)
        print(shape)
        preds = preds.reshape(len(self.legend_name),self.patch_shape[0],self.patch_shape[1],1,shape[-1],shape[-1],1)
        print(np.shape(preds))
        for i in range(len(preds)):
            cur_legend_name = self.legend_name[i]
            print("for legend:", cur_legend_name)
            cur_gt = self.legend_gt[i]

            unpatch_predicted = unpatchify(preds[i], (self.cut_shape[0], self.cut_shape[1], 1))
            pad_unpatch_predicted = np.pad(unpatch_predicted, [(self.shift_coord[0], self.shift_coord[1]), (self.shift_coord[2], self.shift_coord[3]), (0,0)], mode='constant')
            pad_unpatch_predicted = pad_unpatch_predicted.astype(int)
            cur_masked_img = map_mask(self.map, pad_unpatch_predicted)

            intersection = np.logical_and(cur_masked_img.flatten(), cur_gt.flatten()).sum()
            union = cur_masked_img.sum() + cur_gt.sum() 
            dice = (2.0 * intersection) / union
            print("dice:", dice)
            viz_seg = np.repeat(cur_masked_img[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            viz_gt = np.repeat(cur_gt[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            cv2.imwrite("eval_viz/"+cur_legend_name+"_pred.tif", viz_seg) 
            cv2.imwrite("eval_viz/"+cur_legend_name+"_gt.tif", viz_gt) 


    def __len__(self):
        return len(self.map_patches) * len(self.legend_name)

    def sample_tasks(self):
        pass
    def reset_sampler(self):
        pass

def map_mask(imarray, pad_unpatch_predicted_threshold):
    
    gray = cv2.cvtColor(imarray, cv2.COLOR_BGR2GRAY)  # greyscale image
    # Detect Background Color
    pix_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    background_pix_value = np.argmax(pix_hist, axis=None)

    # Flood fill borders
    height, width = gray.shape[:2]
    corners = [[0,0],[0,height-1],[width-1, 0],[width-1, height-1]]
    for c in corners:
        cv2.floodFill(gray, None, (c[0],c[1]), 255)

    # AdaptiveThreshold to remove noise
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Edge Detection
    thresh_blur = cv2.GaussianBlur(thresh, (11, 11), 0)
    canny = cv2.Canny(thresh_blur, 0, 200)
    canny_dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Keeping only the largest detected contour.
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    wid, hight = pad_unpatch_predicted_threshold.shape[0], pad_unpatch_predicted_threshold.shape[1]
    mask = np.zeros([wid, hight])
    mask = cv2.fillPoly(mask, pts=[contour], color=(1,1,1)).astype(int)
    masked_img = cv2.bitwise_and(pad_unpatch_predicted_threshold, mask)
    
    return masked_img


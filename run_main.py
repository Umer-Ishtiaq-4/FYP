import os
import numpy as np
import random
import sys
import time
import datetime
import pprint
from easydict import EasyDict as edict
import torch
import torchvision.transforms as transforms
# datasets
from torch.utils import data 
import pickle
import scipy.sparse as sp
import json
import cv2


edt = edict()
# MY
edt.data_dir = 'D:\Text-to-3D_House_Model\layout'

edt.gpu = 0
edt.CUDA = True
edt.bNum = 3
edt.wProd = False
edt.batch = 64
edt.img_size= 512
edt.channel = 3
edt.imgsizes = [edt.batch,edt.batch*2,edt.batch*4]
# edt.sSize = True
edt.train = True


def Display_Config():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(edt.gpu)

    # print('\n== Config Dictionary ==')
    # pprint.pprint(edt)
    # print("\n")

    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    torch.manual_seed(seed=seed)

    if edt.CUDA:
        torch.cuda.manual_seed_all(seed=seed)

def get_idx_list(my_list, item):
    idx = None
    for i, data_item in enumerate(my_list):
        if data_item == item:
            idx = i
            break
    return idx

def training_loader():
    imsize = edt.batch * (2 ** (edt.bNum-1))
    output_dir  = '/content/Output_Dir'
    image_transform = transforms.Compose([])
    
    dataset_train = Dataset(edt.data_dir, True, edt.batch)
    print("")
    print("Dataset_train[i] == ",len(dataset_train[0]))

    assert dataset_train

    dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=edt.batch,
            drop_last=True, shuffle=True, collate_fn=dataset_train.collate_fn,
            num_workers= 6)
    print("Train_Loader FUN == ",len(dataloader_train))
    return dataloader_train



class Dataset(data.Dataset):
    def __init__(self, data_dir, train_set, batch):

        
        # normalizing each chanel of input values are mean & std_Dev
        self.transform = transforms.Compose([])
        # print(self.transform
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
        
        batch = batch * 8
        # print("Batch",batch)

        
        self.room_types = ['livingroom', 'bedroom', 'corridor', 'kitchen', 
                             'washroom', 'study', 'closet', 'storage', 'balcony']
        self.room_postions = ['NW', 'N', 'NE', 'W', 'C', 'E', 'SW', 'S', 'SE']
        self.total_room_types = len(self.room_types)
        self.total_room_positions = len(self.room_postions)
        
        sem_dir = os.listdir(os.path.join(data_dir, 'semantic-expression'))
        self.total_files = sorted([os.path.splitext(file)[0] for file in sem_dir])

        self.train_set = train_set
        self.wObj = edt.wProd  # False

        if edt.train and self.train_set:
            print("\n\nLoading Training Dataset")
            # Training ID's
            filepath = os.path.join(data_dir, 'train_id.pickle')
            print("== Training ID's File Dir ==>",filepath)
            training_ids = pickle.load(open(filepath, "rb"),encoding="ISO-8859-1")
            print("Total Training ID's ==>",len(training_ids))
            print("== Training Id's Sorted ==\n")
            # print("Train Id =====>\n",train_id)
            training_ids = sorted(training_ids)
    
            # Loading files according to data
            self.training_files = sorted([self.total_files[idx] for idx in training_ids])
            # print(self.training_files)
            # print(len(self.training_files))
            
            
            self.graphs = self.create_Matrix(self.training_files)
            # print("Graph ==> ",self.graphs[0])
         
            # Bounding Box Info
            self.boundingboxes = self.get_boundingbox(self.training_files)
            # print(self.boundingboxes[0])
            self.tensor_matrix = self.tensor_layout_info(self.training_files)
            print(self.tensor_matrix[0])
            # print(self.tensor_matrix[10])
            # print(len(self.tensor_matrix))
            # sys.exit()
            # print("Obj_Vectors ==> ",self.objs_vectors[0])
            
            # build iterator
            self.iterator = self.training_iter

            

            
            print("=== Training Pre Loaded ===\n")


    def create_Matrix(self, filenames):
            current_dir = os.path.join(edt.data_dir, 'semantic-expression')
            graphs = []
            # i = 0
            for filename in filenames:
                path = current_dir + '/' + filename + '.txt'
                # adjacency matrix
                graph = self.gen_Graph(path)
                graph = self.normalizing_Graph(graph)
                graphs.append(graph)
                # i = i + 1
            return graphs

    def gen_Graph(self, path):
        with open(path, 'r') as f:
            transcrpt = json.load(f)
        # pprint.pprint(transcrpt)

        room_info = transcrpt['rooms']
        room_links = transcrpt['links']
        # pprint.pprint(room_info)

        # total rooms in the layout
        num_rooms = sum(room_info[room]['room num'] for room in room_info)
        matrix = np.zeros((num_rooms, num_rooms))

        # Creating rooms names in the adjacency matrix
        room_num = {}
        indiv_num  = 0
        for room in room_info:
            num_room = room_info[room]['room num']
            for i in range(num_room):
                room_name = room + str(i+1)
                room_num[room_name] = indiv_num
                indiv_num += 1
            
        for indiv_room_link in room_links:
            room1_name, room2_name = indiv_room_link['room pair']
            room1 = room_num[room1_name]
            room2 = room_num[room2_name]
            # update matrix 
            matrix[room1][room2] = 1.0

        return matrix

    # normalize the input graph and output with the same size of input one
    def normalizing_Graph(self, graph):
        mat = graph

        # convert to sparse format
        mat = sp.csr_matrix(mat, dtype = np.float32) 

        mat = mat.tolil()
        # build symmetric adjacency matrix
        i, j = np.triu_indices(mat.shape[0])
        mat[j, i] = mat[i, j]

        # normalization
        identity = sp.eye(mat.shape[0], format='csr')
        result = identity + mat
        
        mat = self.row_normalization(result)
        mat = torch.FloatTensor(np.array(mat.todense()))
        return mat

    def get_boundingbox(self, filenames):
        current_dir = os.path.join(edt.data_dir, 'semantic-expression')
        bboxes = []
        # i = 0
        for file in filenames:
            path = current_dir + '/' + file + '.txt'
            # if(i < 10):
            #     print(path)
            bbox = self.get_box_info(path)
            bboxes.append(bbox)
            # i = i + 1
            # break
            # print(bbox)
        return bboxes

    def get_box_info(self, path):
        width = height = edt.img_size
        with open(path, 'r') as f:
            transcript = json.load(f)

        room_data = transcript['rooms']

        total_rooms = sum(room_data[room]['room num'] for room in room_data)

        new_labels = torch.full((total_rooms,), -1, dtype=torch.long)
        new_bounding_boxes = torch.tensor([[0, 0, 1, 1]]).repeat(total_rooms, 1).float()
        # print(total_rooms)
        # print(new_labels)
        # print(new_bounding_boxes)

        # pprint.pprint(room_data)
        room_types = list(room_data.keys())
        
        room_count = 0
        for idx,room_type in enumerate(room_types):
            # print(count)
            room_class_index = get_idx_list(self.room_types,room_type)
            # room count is number of rooms of that particular type
            room_count_type = room_data[room_type]['room num']
            for i in range(room_count_type):
                
                # print(room_type)
                # normalizing with the total img width and height
                x0 = room_data[room_type]['boundingbox'][i]['min']['x']/width
                y0 = room_data[room_type]['boundingbox'][i]['min']['y'] /height
                x1 = room_data[room_type]['boundingbox'][i]['max']['x']/width
                y1 = room_data[room_type]['boundingbox'][i]['max']['y']/height
                new_bounding_boxes[room_count] = torch.FloatTensor([x0, y0, x1, y1])
                new_labels[room_count] = room_class_index
                room_count = room_count + 1
        return new_bounding_boxes, new_labels

   
    def tensor_layout_info(self, filenames):
        current_dir = os.path.join(edt.data_dir, 'semantic-expression')
        layout_mats = []
        for file in filenames:
            path = current_dir + '/' + file + '.txt'
            layout_mats.append(self.get_tensor_matrix(path))
        return layout_mats

    # build the vectors of objects in an image
    def get_tensor_matrix(self, path):
        with open(path, 'r') as f:
            transcript = json.load(f)

            room_data = transcript['rooms']
            total_rooms = sum(room_data[room]['room num'] for room in room_data)
            # Indiv layout room info tensor
            room_tensor = torch.full((total_rooms,), -1, dtype=torch.long)
            
            len_of_obj = self.total_room_types + self.total_room_positions
            tensor_matrix = torch.tensor(np.zeros((total_rooms, len_of_obj + 1)), dtype=torch.float32)
            # pprint.pprint(room_data)
            room_types = room_data.keys()
            room_count_type = 0
            for idx,room_type in enumerate(room_types):
                # the position in the room classes
                room_class_index = get_idx_list(self.room_types,room_type)
                room_sizes = room_data[room_type]['size']
                room_positions = room_data[room_type]['position']
                room_count = room_data[room_type]['room num']
                # print(room_class_index)
                # print(room_positions)
                # print(room_sizes)
                # print(room_count)
                for i in range(room_count):
                    # update tensor matrix
                    tensor_matrix[room_count_type][room_class_index] = 1.0
                    room_position = room_positions[i]
                    room_position_index = get_idx_list(self.room_postions,room_position)
                    tensor_matrix[room_count_type][self.total_room_types+1+room_position_index] = 1.0
                    tensor_matrix[room_count_type][len(self.room_types)] = room_sizes[i]
                    room_tensor[room_count_type] = room_class_index
                    room_count_type = room_count_type + 1
                    
        return tensor_matrix, room_tensor
    
    

    def prepare_tp_file(self,dir,img_type,id,f_type):
        return dir + '/' + img_type + '/'+ id + f_type


    
    
    def training_iter(self, index):
        label_directory = 'label'
        mask_directory = 'mask'
        file_format = '.png'
        
        training_file = self.get_training_data(train_idx=index)
        graph_data = self.get_training_data(graph_idx=index)
        bounding_box = self.get_training_data(boundingbox_idx=index)
        tensor_matrix = self.get_training_data(tensor_matrix_idx=index)

        label_image_path = self.prepare_tp_file(edt.data_dir, label_directory, training_file, file_format)
        processed_label_images = self.process_images(label_image_path, edt.imgsizes, self.transform, normalize_func=self.normalize)

        mask_image_path = self.prepare_tp_file(edt.data_dir, mask_directory, training_file, file_format)
        processed_mask_images = self.process_images(mask_image_path, edt.imgsizes, self.transform, normalize_func=self.normalize)

        # get a different index
        incorrect_index = self.generate_random_index(index)
        
        
        
        incorrect_id = self.get_training_data(train_idx=incorrect_index)

        incorrect_label_image_path = self.prepare_tp_file(edt.data_dir, label_directory, incorrect_id, file_format)
        processed_incorrect_label_images = self.process_images(incorrect_label_image_path, edt.imgsizes, self.transform, normalize_func=self.normalize)

        incorrect_mask_image_path = self.prepare_tp_file(edt.data_dir, mask_directory, incorrect_id, file_format)
        processed_incorrect_mask_images = self.process_images(incorrect_mask_image_path, edt.imgsizes, self.transform, normalize_func=self.normalize)

        return processed_label_images, processed_mask_images, processed_incorrect_label_images, processed_incorrect_mask_images, graph_data, bounding_box, tensor_matrix, training_file





    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        if edt.train and self.train_set:
            return len(self.training_files)
        
    def generate_random_index(self, correct_index):
        incorrect_index = (correct_index + random.randint(1, len(self.training_files) - 1)) % len(self.training_files)
        return incorrect_index

    

    def get_training_data(self,train_idx=None,graph_idx=None,boundingbox_idx=None,tensor_matrix_idx=None):
        
        if train_idx != None:
            return self.training_files[train_idx]
        elif graph_idx != None:
            return self.graphs[graph_idx]
        elif boundingbox_idx != None:
            return self.boundingboxes[boundingbox_idx]
        elif tensor_matrix_idx != None:
            return self.tensor_matrix[tensor_matrix_idx]


    def row_normalization(self, mx):
        rowsum = mx.sum(1)
        rowsum[rowsum == 0] = 1
        r_mat_inv = 1/rowsum
        mx = mx.multiply(r_mat_inv)
        return mx
    
    def process_images(self, file_path, dimensions, transform=None, normalize_func=None):
        image = cv2.imread(file_path)
        if transform:
            image = transform(image)
        processed_images = []
        for i in range(edt.bNum):
            resized_image = cv2.resize(image, dsize=(dimensions[i], dimensions[i]), interpolation=cv2.INTER_CUBIC)
            if normalize_func:
                resized_image = normalize_func(resized_image)
            processed_images.append(resized_image)
        return processed_images

    def collate_fn(self, batch):
        data = {
            'images': [],
            'masks': [],
            'wrong_images': [],
            'wrong_masks': [],
            'graphs': [],
            'bboxes': [],
            'vectors': [],
            'keys': []
        }
        for b in batch:
            if len(b) == 8:
                data['wrong_images'].append(b[2][-1])
                data['wrong_masks'].append(b[3][-1])
            elif len(b) == 6:
                pass
            data['images'].append(b[0][-1])
            data['masks'].append(b[1][-1])
            data['graphs'].append(b[2])
            data['bboxes'].append(b[3])
            data['vectors'].append(b[4])
            data['keys'].append(b[5])
        data['images'] = torch.stack(data['images'], dim=0)
        return (data['images'], data['masks'], data['wrong_images'], data['wrong_masks'], data['graphs'], data['bboxes'], data['vectors'], data['keys'])




    



if __name__ == "__main__":
    
    Display_Config()
    if edt.train:
        print("TES")
        train_Loader = training_loader()
    
    print("Train_Loader == ",len(train_Loader))
    print("CUDA Available -> ",torch.cuda.is_available())
    print("WORKING")
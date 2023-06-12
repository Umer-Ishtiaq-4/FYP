import os
import numpy as np
import random
import sys
import time
import datetime
import pprint
import argparse
from easydict import EasyDict as edict
import torch
import torchvision.transforms as transforms
# datasets
from torch.utils import data 
import pickle
import scipy.sparse as sp
import json
import networkx as nx
from PIL import Image
import cv2


edt = edict()
cfg = edt
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
# __C.WORKERS = 6
edt.train = True


def Display_Config():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(edt.gpu)

    # print('\n== Config Dictionary ==')
    # pprint.pprint(cfg)
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

    print("Dataset_train[i] == ",len(dataset_train[0]))

    assert dataset_train

    dataloader_train = torch.utils.data.DataLoader(
            dataset_train, batch_size=edt.batch,
            drop_last=True, shuffle=True, collate_fn=dataset_train.collate_fn,
            num_workers= 6)
    return dataloader_train



class Dataset(data.Dataset):
    def __init__(self, data_dir, train_set, batch):

        
        # normalizing each chanel of input values are mean & std_Dev
        self.transform = transforms.Compose([])
        # print(self.transform
        self.normalizes= transforms.Compose([
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
            print("Loading Training Dataset")
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
            # print(self.training_files[0])
            # print(len(self.training_files))
            # sys.exit()
            
            self.graphs = self.load_graphs(data_dir, self.training_files)
            print(self.graphs[4])
            sys.exit()
            sys.exit()
            
            
            self.boundingboxes = self.load_bboxes(data_dir, batch,self.training_files)
            # print("bBOX ==> ",self.bboxes[0])

            
            self.tensor_matrix = self.build_tensor_matrix(data_dir, self.training_files)
            
            # print("Obj_Vectors ==> ",self.objs_vectors[0])
            
            # build iterator
            self.iterator = self.training_iter

            

            
            print("=== Training Pre Loaded ===\n")


    # def create_Matrix(self, data_dir, filenames):
    #         current_dir = os.path.join(data_dir, 'semantic-expression')
    #         graphs = []
    #         # i = 0
    #         for filename in filenames:
    #             path = current_dir + '/' + filename + '.txt'
    #             # get the adjacent of the rooms
    #             graph = self.gen_Graph(path)
    #             # if(i == 0):
    #             #     print("Adj Matrix \n",graph)
    #             # probabilistic the adjacency of a room to every room
    #             graph = self.normalizing_Graph(graph)
    #             graphs.append(graph)
    #             # i = i + 1
    #         return graphs
    def load_filenames(self, data_dir):
        filenames = []
        current_dir = os.path.join(data_dir, 'semantic-expression')
        for file in os.listdir(current_dir):
            filenames.append(os.path.splitext(file)[0])
        return sorted(filenames)

    def load_graphs(self, data_dir, filenames):
        current_dir = os.path.join(data_dir, 'semantic-expression')
        graphs = []
        for filename in filenames:
            path = os.path.join(current_dir, '{}.txt'.format(filename))
            # get the adjacent of the rooms
            graph = self.build_graph(path)
            # probabilistic the adjacency of a room to every room
            graph = self.normalize_graph(graph)
            graphs.append(graph)
        return graphs

    # build the node and edge according each text description
    # path: e.g., path = './semantic-expression/0.txt'
    def build_graph(self, path):
        with open(path, 'rb') as f:
            desc = f.read()
            desc = eval(desc)
            desc_rooms = desc['rooms']
            desc_links = desc['links']
            # count the number of nodes (rooms)
            count_nodes = 0
            for room in desc_rooms.keys():
                num_room = desc_rooms[room]['room num']
                count_nodes += num_room

            adj = np.zeros((count_nodes, count_nodes))

            # handle feature
            counter = 0  # counter of room number
            roomnames = []  # the name of each room, e.g., bedroom1
            # build all room in roomnames
            for room in desc_rooms.keys():
                # the position in the room classes
                idx = self.room_types.index(room)
                num_room = desc_rooms[room]['room num']
                for i in range(num_room):
                    # feature[counter][idx] = 1.0
                    roomnames.append('{}{}'.format(room, i+1))
                    counter += 1
            # handle adjacent
            for desc_link in desc_links:
                # decide the position in adj
                link = desc_link['room pair']
                x = roomnames.index(link[0])
                y = roomnames.index(link[1])
                adj[x][y] = 1.0
        return adj

    # normalize the input graph and output with the same size of input one
    def normalize_graph(self, graph):
        # feature, adj = graph[0], graph[1]
        adj = graph

        # convert to space format
        adj = sp.coo_matrix(adj, dtype = np.float32)

        # build symmetric adjacency matrix
        # the item "adj.multiply(adj.T > adj)" may be useless
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # normalization
        # diagonal line will be 1
        adj = self.normalize(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        return adj

    # normalization for graph (feature and adj)
    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    # def gen_Graph(self, path):
    #     with open(path, 'r') as f:
    #         transcrpt = json.load(f)
    #     # pprint.pprint(transcrpt)

    #     room_info = transcrpt['rooms']
    #     room_links = transcrpt['links']
    #     # pprint.pprint(room_info)

    #     # Build the adjacency matrix
    #     num_rooms = sum(room_info[room]['room num'] for room in room_info)
    #     matrix = np.zeros((num_rooms, num_rooms))

    #     # Create a mapping of room names to indices in the adjacency matrix
    #     room_num = {}
        
    #     # every room with separate number 
    #     indiv_num  = 0
    #     for room in room_info:
    #         num_room = room_info[room]['room num']
    #         for i in range(num_room):
    #             room_name = room + str(i+1)
    #             room_num[room_name] = indiv_num
    #             indiv_num += 1
        
    #     for indiv_room_link in room_links:
    #         room1_name, room2_name = indiv_room_link['room pair']
    #         # print(room1_name,room2_name)
    #         room1 = room_num[room1_name]
    #         room2 = room_num[room2_name]
    #         # update matrix 
    #         matrix[room1][room2] = float(1)

    #     return matrix

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


    def load_bboxes(self, data_dir, batch, filenames):
        current_dir = os.path.join(data_dir, 'semantic-expression')
        bboxes = []
        # i = 0
        for file in filenames:
            path = current_dir + '/' + file + '.txt'
            # if(i < 10):
            #     print(path)
            bbox = self.build_bboxes(path,batch)
            bboxes.append(bbox)
            # i = i + 1
            # break
        return bboxes

    def build_bboxes(self, path,batch):
        width = height = edt.img_size
        with open(path, 'r') as f:
            transcript = json.load(f)

        room_data = transcript['rooms']

        total_rooms = sum(room_data[room]['room num'] for room in room_data)

        new_labels = torch.full((total_rooms,), -1, dtype=torch.long)
        new_bounding_boxes = torch.tensor([[0, 0, 1, 1]]).repeat(total_rooms, 1).float()


        # pprint.pprint(room_data)
        room_types = list(room_data.keys())
        

        for idx,room_type in enumerate(room_types):
            room_class_index = get_idx_list(self.room_types,room_type)
            # room count is number of rooms of that particular type
            room_count = room_data[room_type]['room num']
            # print(room_count)
            for i in range(room_count):
                # print(room_type)
                # normalizing with the total img width and height
                bouding_box = room_data[room_type]['boundingbox']
                x0 = bouding_box[i]['min']['x']
                y0 = bouding_box[i]['min']['y']
                x1 = bouding_box[i]['max']['x']
                y1 = bouding_box[i]['max']['y']
                new_bounding_boxes[idx] = torch.FloatTensor([x0/width, y0/height, x1/width, y1/height])
                new_labels[idx] = room_class_index


        return new_bounding_boxes, new_labels

   
    def build_tensor_matrix(self, data_dir, filenames):
        current_dir = os.path.join(data_dir, 'semantic-expression')
        objs_vectors = []
        for file in filenames:
            path = current_dir + '/' + file + '.txt'
            objs_vector = self.get_tensor_matrix(path)
            objs_vectors.append(objs_vector)
        return objs_vectors

    # build the vectors of objects in an image
    def get_tensor_matrix(self, path):
        with open(path, 'r') as f:
            transcript = json.load(f)

            room_data = transcript['rooms']
            total_rooms = sum(room_data[room]['room num'] for room in room_data)

            # dimension of each objects (row)
            # print(total_rooms)
            # Indiv layout room info tensor
            room_tensor = torch.full((total_rooms,), -1, dtype=torch.long)
            
            len_of_obj = self.total_room_types + self.total_room_positions
            tensor_matrix = torch.tensor(np.zeros((total_rooms, len_of_obj + 1)), dtype=torch.float32)
        
            # pprint.pprint(room_data)
            room_types = room_data.keys()
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
                    tensor_matrix[idx][room_class_index] = 1.0
                    room_position = room_positions[i]
                    room_position_index = get_idx_list(self.room_postions,room_position)
                    tensor_matrix[idx][self.total_room_types+1+room_position_index] = 1.0
                    tensor_matrix[idx][len(self.room_types)] = room_sizes[i]
                    room_tensor[idx] = room_class_index

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
        train_Loader = training_loader()
    
    print("Train_Loader == ",len(train_Loader))
    print("CUDA Available -> ",torch.cuda.is_available())
    print("WORKING")
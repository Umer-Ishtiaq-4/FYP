import torch
import pprint
# from model_graph import GCN, BBOX_NET
import cv2
from torch import tensor
import numpy as np
import os
from PIL import Image, ImageFont, ImageDraw
import scipy.sparse as sp

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


room_classes = ['livingroom', 'bedroom', 'corridor', 'kitchen', 
                             'washroom', 'study', 'closet', 'storage', 'balcony']
position_classes = ['NW', 'N', 'NE', 'W', 'C', 'E', 'SW', 'S', 'SE']



class GCN(nn.Module):
    def __init__(self, nfeat, nhid, output_dim):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if device == 'cuda':
            self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            if device == 'cuda':
                self.bias = Parameter(torch.cuda.FloatTensor(out_features))
            else:
                self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # support = torch.mm(input, self.weight)
        support = torch.matmul(input, self.weight)
        # output = torch.spmm(adj, support)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class BBOX_NET(nn.Module):
    def __init__(self, dim_list, activation='relu', batch_norm='none',
                  dropout=0, final_nonlinearity=True):
        super(BBOX_NET, self).__init__()

        self.mlp = self.build_mlp(dim_list=dim_list, 
            activation=activation, batch_norm=batch_norm,
            dropout=dropout, final_nonlinearity=final_nonlinearity)

    def build_mlp(self, dim_list, activation='relu', batch_norm='none',
                  dropout=0, final_nonlinearity=True):
      layers = []
      for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
          if batch_norm == 'batch':
            layers.append(nn.BatchNorm1d(dim_out))
          if activation == 'relu':
            layers.append(nn.ReLU())
          elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
          layers.append(nn.Dropout(p=dropout))
      return nn.Sequential(*layers)

   
    def forward(self, objs_vector, graph_objs_vector):
            # element-wise add
            x = torch.add(objs_vector, graph_objs_vector)
            output = self.mlp(x)
            return output
 


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
    layers = []
    for i in range(len(dim_list) - 1):
        dim_in, dim_out = dim_list[i], dim_list[i + 1]
        layers.append(nn.Linear(dim_in, dim_out))
        final_layer = (i == len(dim_list) - 2)
        if not final_layer or final_nonlinearity:
          if batch_norm == 'batch':
            layers.append(nn.BatchNorm1d(dim_out))
          if activation == 'relu':
            layers.append(nn.ReLU())
          elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def define_models():
        objs_vector_dim = 19
        # build gcn model
        input_graph_dim = objs_vector_dim
        hidden_graph_dim = 64
        output_graph_dim = objs_vector_dim
        gcn = GCN(nfeat=input_graph_dim, 
                  nhid=hidden_graph_dim, output_dim=output_graph_dim)
        # build box_net model
        gconv_dim = objs_vector_dim
        gconv_hidden_dim = 512
        box_net_dim = 4
        mlp_normalization = 'none'
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        box_net = BBOX_NET(box_net_layers, batch_norm=mlp_normalization)
        return gcn, box_net



def get_gcn_bbox_loaded():
    # set the device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f"Device set to: {device}")

    # Load the state dictionary from the .pth file using map_location
    state_dict_gcn = torch.load('gcn_best.pth', map_location=device)
    # build gcn model
    gcn, box_net = define_models()
    # load the state dictionary into the model
    gcn.load_state_dict(state_dict_gcn)

    state_dict_box = torch.load('box_net_best.pth', map_location=device)
    box_net.load_state_dict(state_dict_box)

    # move the models to the device
    gcn.to(device)
    box_net.to(device)
    return gcn,box_net

gcn, box_net = get_gcn_bbox_loaded()
# print(gcn)
# print(box_net)


# All the graph building logic
def count_rooms(room_name,room_list):
    count = 0
    for room in room_list:
        if room[:-1] == room_name:
            count += 1
    return count

def remove_duplicates(room_list):
    unique_rooms = []
    for room in room_list:
        if room[:-1] not in unique_rooms:
            unique_rooms.append(room[:-1])
    return unique_rooms


def build_graph(desc):
        try:
            desc_rooms = desc['rooms']
            desc_links = desc['links']
            
            count_nodes = len(desc_rooms)
            adj = np.zeros((count_nodes, count_nodes))

            number_of_speicific_rooms = {}
            # get individual rooms
            indiv_room_list = remove_duplicates(desc_rooms)
            for room in indiv_room_list:
              number_of_speicific_rooms[room] = count_rooms(room,desc_rooms)
            # print(number_of_speicific_rooms)
            # handle feature
            counter = 0  # counter of room number
            roomnames = []  # the name of each room, e.g., bedroom1
            # build all room in roomnames

            for room in indiv_room_list:
                # the position in the room classes
                idx = room_classes.index(room)
                # num_room = desc_rooms[room]['room num']
                num_room = number_of_speicific_rooms[room]
                for i in range(num_room):
                    # feature[counter][idx] = 1.0
                    roomnames.append('{}{}'.format(room, i+1))
                    counter += 1
            # handle adjacent
            for desc_link in desc_links:
                # decide the position in adj
                link = desc_link
                x = roomnames.index(link[0])
                y = roomnames.index(link[1])
                adj[x][y] = 1.0
            return adj
        except:
            raise("The rooms you have mentioned in text is not in the list")


def normalize_graph(graph):
        # feature, adj = graph[0], graph[1]
        adj = graph

        # convert to space format
        adj = sp.coo_matrix(adj, dtype = np.float32)

        # build symmetric adjacency matrix
        # the item "adj.multiply(adj.T > adj)" may be useless
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # normalization
        # diagonal line will be 1
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = torch.FloatTensor(np.array(adj.todense()))
        return adj

# normalization for graph (feature and adj)
def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

# All the object vector logic
def get_all_rooms(desc,room):
    rooms = desc['rooms']
    specific_rooms_list = []
    for indiv_room in rooms:
      if(indiv_room[:-1] == room):
        specific_rooms_list.append(indiv_room)
    return specific_rooms_list
def build_objs_vector(desc):
            desc_rooms = desc['rooms']

            O = len(desc_rooms)
            # dimension of each objects (row)
            D = len(room_classes) + 1 + len(position_classes)
            
            objs = torch.LongTensor(O).fill_(-1)
            objs_vector = torch.FloatTensor([[0.0]]).repeat(O, D)

            # +++++
            number_of_speicific_rooms = {}
            indiv_room_list = remove_duplicates(desc_rooms)
            for room in indiv_room_list:
              number_of_speicific_rooms[room] = count_rooms(room,desc_rooms)

            # handle vector of objects (type, size, position)
            counter = 0  # counter of room number
            indiv_room_list = remove_duplicates(desc_rooms)
            for room in indiv_room_list:
                # the position in the room classes
                idx_type = room_classes.index(room)
                num_room = number_of_speicific_rooms[room]
                rooms_sizes = []
                all_room_rooms = get_all_rooms(desc,room)
                for i,room in enumerate(all_room_rooms):
                    # type in vector
                    objs_vector[counter][idx_type] = 1.0
                    
                    # position in vector
                    position = desc['sizes'][room][1]
                    
        
                    idx_position = position_classes.index(position)
                    size = desc['sizes'][room][0]
                    objs_vector[counter][len(room_classes)+1+idx_position] = 1.0
                        # size in vector
                    objs_vector[counter][len(room_classes)] = size
                  

                    # record type
                    objs[counter] = idx_type
                    counter += 1
            return objs_vector, objs

def get_graph_objs(info_dict):
  graph = build_graph(info_dict)
  graph = normalize_graph(graph)
  # print(graph)

  objs_vector = build_objs_vector(info_dict)
  # print(objs_vector)
  return graph, objs_vector

# graph, objs_vector = get_graph_objs(info)

# print(graph.to(device))
# print(objs_vector[0].to(device))


def draw_floor_plan(image, curr_box, label, original_size, new_size):
    wall_thickness = 2
    wall_symbol = 2.0
    x1, y1, x2, y2 = curr_box[0], curr_box[1], curr_box[2], curr_box[3]
    h, w = original_size
    h_new, w_new = new_size
    x1 = int(x1.item() * w)
    y1 = int(y1.item() * h)
    x2 = int(x2.item() * w)
    y2 = int(y2.item() * h)
    x1_new = int(x1 * w_new / w)
    y1_new = int(y1 * h_new / h)
    x2_new = int(x2 * w_new / w)
    y2_new = int(y2 * h_new / h)
    image[:, y1_new:y2_new, x1_new:x2_new] = label / 13.0
    image[:, y1_new-wall_thickness:y1_new+wall_thickness, x1_new:x2_new] = wall_symbol
    image[:, y2_new-wall_thickness:y2_new+wall_thickness, x1_new:x2_new] = wall_symbol
    image[:, y1_new:y2_new, x1_new-wall_thickness:x1_new+wall_thickness] = wall_symbol
    image[:, y1_new:y2_new, x2_new-wall_thickness:x2_new+wall_thickness] = wall_symbol
    return image



def save_multiple_bboxes(bboxes_tensor, labels, filename, room_names):
    bg_color = 9.0 / 13
    original_size = (256, 256)
    new_size = (384, 384)  # Increase the image width to 384 pixels
    image = torch.ones((3, *new_size)) * bg_color

    # Draw each bounding box on the white background with the corresponding label
    for bbox, label in zip(bboxes_tensor, labels):
        image = draw_floor_plan(image, bbox, label, original_size, new_size)

    # Convert the tensor to a numpy array and scale it to a range of 0 to 13
    ndarr = image.mul_(13).add_(0.5).clamp_(0, 14).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    # Define the color palette
    palette = [255, 255, 255] * 256
    # print("Palette Shape: {}".format(len(palette)))
    custom_colors = [
        [84, 139, 84], # green
        [0, 100, 0], # dark green
        [0, 0, 128], # navy blue
        [85, 26, 139], # purple
        [255, 0, 255], # magenta
        [165, 42, 42], # brown
        [139, 134, 130], # gray
        [205, 198, 115], # olive green
        [139, 58, 58], # dark red
        [255, 255, 255], # white
        [0, 0, 0], # black
        [30, 144, 255], # dodger blue
        [135, 206, 235], # sky blue
        [255, 255, 0], # yellow
        [0, 0, 0],
    ]
    # print("Len CC: {}".format(len(custom_colors)))
    for i, color in enumerate(custom_colors):
        # print("Color: ", color)
        palette[i * 3:i * 3 + 3] = color
    # Create an image object and apply the color palette
    im = Image.fromarray(ndarr).convert('L')
    im.putpalette(palette)

    # Draw room names and corresponding colors on the image
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('arial.ttf', 12)  # You may need to replace 'arial.ttf' with the path to a font file

    x, y = 5, 5  # Starting position of the legend
    for i, room_name in enumerate(room_names):
        draw.rectangle([(x, y), (x + 10, y + 10)], fill=i)
        draw.text((x + 15, y - 2), room_name, fill=(0, 0, 0), font=font)
        y += 20  # Move the next legend item down by 20 pixels
    # Save the image
    return im, filename
    

def file_check(filename):
#   print(filename)
  filebase, fileext = os.path.splitext(filename)
  if os.path.isfile((filebase + fileext)):
    while(os.path.isfile(filebase + fileext)):
    #   print(f'{filebase} exists!')
      trunc_filebase = filebase[:-1]
      count = int(filebase[-1])
      count = count + 1
      filebase = trunc_filebase + str(count)
      
    return filebase+fileext
  else:
    #   print(f'{filebase} does not exist!')
      return filename

# print(file_check('Sav1.png'))

def Generate(post_processed_input):
    room_names = post_processed_input['rooms']
    graph, objs_vector = get_graph_objs(post_processed_input)
    graph = graph.to(device)
    objs = objs_vector[0].to(device)
    gcn.eval()
    box_net.eval()
    with torch.no_grad():
        graph_objs_vector_test = gcn(objs, graph)
        boxes_pred = box_net(objs, graph_objs_vector_test)
        # print("Pred type",type(boxes_pred))
        # print("Pred len",len(boxes_pred))

        file1 = 'Sav1.png'
        file1 = file_check(file1)
        num_bboxes = boxes_pred.shape[0]
        labels = list(range(num_bboxes))
        im, filename = save_multiple_bboxes(boxes_pred, labels, file1, room_names)
        

        # print("Image Saved")

        return im

# Generate(post_processed_input)




from __future__ import print_function
import os
import time
import save_images as save
import torch
import json
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

from main import edt

plt.switch_backend('agg')




class Graph_Net(nn.Module):
    def __init__(self, nfeat, nhid, output_dim):
        super(Graph_Net, self).__init__()
        self.gc1 = Graph_Conv(nfeat, nhid)
        self.gc2 = Graph_Conv(nhid, output_dim)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = self.gc2(x, adj)
        return x


class Graph_Conv(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Graph_Conv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.cuda.FloatTensor(in_features, out_features))
        
        if bias:
            if edt.CUDA:
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


class BouningBox_Net(nn.Module):
    def __init__(self, dim_list, activation='relu', batch_norm='none',
                  dropout=0, final_nonlinearity=True):
        super(BouningBox_Net, self).__init__()

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



def Set_Optimizer(model, lr, weight_decay):
    optimizer_model = optim.Adam(model.parameters(),
                            lr=lr, 
                            weight_decay=weight_decay,
                            betas=(0.75, 0.999))
    return optimizer_model


def save_model(model, epoch, model_dir, model_name, best=False):
    if best:
        torch.save(model.state_dict(), '%s/%s_best.pth' % (model_dir, model_name))
        # cfg.TRAIN.GCN = 'SAVED'
    else:
        torch.save(model.state_dict(), '%s/%s_%d.pth' % (model_dir, model_name, epoch))
        # cfg.TRAIN.GCN = 'SAVED'

def save_img_results(imgs_tcpu, real_box, boxes_pred, count, image_dir):
    num = 64
    # range [0, 1]
    real_img = imgs_tcpu[-1][0:num]
    # save floor plan images
    save.save_floor_plan(
        real_img, real_box, '%s/count_%09d_real_floor_plan.png' % (image_dir, count),
        normalize=True)
    save.save_image(
        real_img, '%s/count_%09d_real_samples.png' % (image_dir, count),
        normalize=True)
    # save bounding box images
    save.save_bbox(
        real_img, real_box, '%s/count_%09d_real_bbox.png' % (image_dir, count),
        normalize=True)
    save.save_bbox(
        real_img, boxes_pred, '%s/count_%09d_fake_bbox.png' % (image_dir, count),
        normalize=True)
    save.save_floor_plan(
        real_img, boxes_pred, '%s/count_%09d_fake_floor_plan.png' % (image_dir, count),
        normalize=True)

def create_dir(path):    
    if not os.path.exists(path):
        # Create directory if it doesn't exist
        os.makedirs(path)
        print("Directory created successfully!")
    else:
        print("Directory already exists!")


class Trainer(object):
    def __init__(self, output_dir, dataloader_train):
        # build save data dir
        print("Layout Trainer ====>",output_dir)
        self.output_dir = output_dir
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        
        create_dir(self.model_dir)
        create_dir(self.image_dir)    
        self.device = torch.device('cuda: 0')

        self.batch_size =  edt.batch
        if dataloader_train!=None:
            self.dataloader_train = dataloader_train
            self.num_batches = len(self.dataloader_train)
        self.dataloader_test = dataset_test
        self.best_loss = float('inf')
        self.best_epoch = 0

    def prepare_data(self, data, test= False):
        if not test:
            label_imgs, _, wrong_label_imgs, _, graph, bbox, objs_vector, key = data
            vgraph, vbbox, vobjs_vector = [], [], []
            if edt.CUDA:
                for i in range(len(graph)):
                    # vgraph.append((graph[i][0].to(self.device), graph[i][1].to(self.device)))
                    vgraph.append(graph[i].to(self.device))
                    vbbox.append((bbox[i][0].to(self.device), bbox[i][1].to(self.device)))
                    vobjs_vector.append((objs_vector[i][0].to(self.device), objs_vector[i][1].to(self.device)))
            return label_imgs, vgraph, vbbox, vobjs_vector, key
        else:
            label_imgs, _, graph, bbox, objs_vector, key = data

            # real_vimgs = []
            vgraph, vbbox, vobjs_vector = [], [], []
            if cfg.CUDA:
                for i in range(len(graph)):
                    # vgraph.append((graph[i][0].to(self.device), graph[i][1].to(self.device)))
                    vgraph.append(graph[i].to(self.device))
                    vbbox.append((bbox[i][0].to(self.device), bbox[i][1].to(self.device)))
                    vobjs_vector.append((objs_vector[i][0].to(self.device), objs_vector[i][1].to(self.device)))
            return label_imgs, vgraph, vbbox, vobjs_vector, key



    def get_models(self):   
        objs_vector_dim = 19
        # gcn model
        input_graph_dim = objs_vector_dim
        hidden_graph_dim = 64
        output_graph_dim = objs_vector_dim
        gn = Graph_Net(nfeat=input_graph_dim, 
                  nhid=hidden_graph_dim, output_dim=output_graph_dim)
        # box_net model
        gconv_dim = objs_vector_dim
        gconv_hidden_dim = 512
        box_net_dim = 4
        mlp_normalization = 'none'
        box_net_layers = [gconv_dim, gconv_hidden_dim, box_net_dim]
        box_net = BouningBox_Net(box_net_layers, batch_norm=mlp_normalization)
        return gn, box_net


    def train(self):
        # plot
        self.training_epoch = []
        self.testing_epoch = []
        self.training_loss = []
        self.testing_loss = []

        self.gcn, self.box_net = self.get_models()
        # optimization method
        self.optimizer_gcn = Set_Optimizer(
            self.gcn, edt.gcn_lr, edt.l2_reg)
        self.optimizer_bbox = Set_Optimizer(
            self.box_net, edt.bbox_lr, edt.l2_reg)

        # criterion function
        self.criterion_bbox = nn.MSELoss()
        if edt.CUDA:
            self.criterion_bbox.to(self.device)
            self.gcn.to(self.device)
            self.box_net.to(self.device)
        
        start_epoch = 0
        self.max_epoch = 10
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()
            print("\n\n\nEpoch =======>",epoch)
            print("Out of ====>",self.max_epoch)
            for step, data in enumerate(self.dataloader_train, 0):
                self.imgs_tcpu, self.graph, self.real_box, self.objs_vector, self.key = self.prepare_data(data)
                self.box_net.train()
                print("Step ===>",step)
                # for each image
                for i in range(len(self.real_box)):
                    graph_objs_vector = self.gcn(self.objs_vector[i][0], self.graph[i])
                    boxes_pred = self.box_net(self.objs_vector[i][0], graph_objs_vector)
                    # optimization
                    if i == 0:
                        loss_bbox = self.criterion_bbox(boxes_pred, self.real_box[i][0])
                    else:
                        loss_bbox += self.criterion_bbox(boxes_pred, self.real_box[i][0])
                loss_bbox = loss_bbox / len(self.real_box)
                loss_total = edt.box_pred_imp * loss_bbox
                self.training_epoch.append(epoch)
                self.training_loss.append(loss_total.item())

                self.optimizer_gcn.zero_grad()
                self.optimizer_bbox.zero_grad()
                loss_total.backward()
                self.optimizer_gcn.step()
                self.optimizer_bbox.step()

            # save the best models
            print('comparing total loss...')
            if loss_total.item() < self.best_loss:
                self.best_loss = loss_total.item()
                self.best_epoch = epoch
                print('saving best models...')
                print("Model Dir =====>",self.model_dir)
                save_model(model=self.gcn, epoch=epoch, model_dir=self.model_dir,
                           model_name='gcn', best=True)
                save_model(model=self.box_net, epoch=epoch, model_dir=self.model_dir,
                           model_name='box_net', best=True)
            print("current_epoch[{}] current_loss[{}] best_epoch[{}] best_loss[{}]".format(
    epoch, loss_total.item(), self.best_epoch, self.best_loss))
            self.training_epoch.append(epoch)
            self.training_loss.append(loss_total)
            
            # print("Len ===> self.training_epoch", len(self.training_epoch))
            # print("Len ===> self.training_loss", len(self.training_loss))
            
            if epoch % 5 == 0 or epoch == 0:
                self.gcn.eval()
                self.box_net.eval()
                boxes_pred_collection = []
                for i in range(len(self.real_box)):
                    graph_objs_vector = self.gcn(self.objs_vector[i][0], self.graph[i])
                    # bounding box prediction
                    boxes_pred_save = self.box_net(self.objs_vector[i][0], graph_objs_vector)
                    boxes_pred_collection.append((boxes_pred_save, self.real_box[i][1]))
                save_img_results(self.imgs_tcpu, self.real_box, boxes_pred_collection, epoch, self.image_dir)

            self.training_epoch.append(epoch)
            self.training_loss.append(loss_total)
            # plot
            plt.figure(0)
            self.training_epoch = [e for e in self.training_epoch]
            self.training_loss = [e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e for e in self.training_loss]
            plt.plot(self.training_epoch, self.training_loss, color="r", linestyle="-", linewidth=1, label="training")
            self.testing_loss = [e.detach().cpu().numpy() if (isinstance(e, torch.Tensor) and e.device.type == 'cuda') else e for e in self.testing_loss]
            plt.plot(self.testing_epoch, self.testing_loss, color="b", linestyle="-", linewidth=1, label="testing")

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.legend(loc='best')
            plt.savefig(os.path.join(self.output_dir, "loss.png"))
            plt.close(0)

            
            print('Checkpoint')
            if epoch % 5 == 0:
                save_model(model=self.gcn, epoch=epoch, model_dir=self.model_dir,
                           model_name='gcn', best=False)
                save_model(mSSodel=self.box_net, epoch=epoch, model_dir=self.model_dir,
                           model_name='box_net', best=False)


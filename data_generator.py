import torch
import os
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import sys
sys.path.append(os.path.join(CUR_DIR,'../../RDF'))
# from panda_layer.panda_layer import PandaLayer
# from bf_sdf import BPSDF
from splineCurve3D import splineCurve3D, TDCR
from torchmin import minimize
import time
import math
import copy
import matplotlib.pyplot as plt
import yaml

PI = math.pi

class DataGenerator():
    def __init__(self,device):
        # panda model
        # self.panda = PandaLayer(device)
        # self.bp_sdf = BPSDF(8,-1.0,1.0,self.panda,device)
        # self.model = torch.load(os.path.join(CUR_DIR,'../../RDF/models/BP_8.pt'))
        k = 3 # degree
        self.n = 12 # number of control points
        breaks = torch.linspace(0, 1, k+self.n+1-2*k)
        t = torch.cat((torch.zeros(k), breaks, torch.ones(k)))# knots
        self.L = 0.2
        sites = torch.cat((torch.tensor([0]), breaks[:-1] + (breaks[1:] - breaks[:-1])/2, torch.tensor([1])))
        # sites = torch.tensor([0,1])
        self.curve = splineCurve3D(t, k, self.L, sites, device)
        self.q_max = 40 # curvature limit
        self.q_min = -40
        # device
        self.device = device
        self.arc_pt_num = 12

        # data generation
        self.workspace = [[-0.05,-0.05,0.055],[0.05,0.05,0.15]]
        self.theta = torch.linspace(0,PI*0.8,self.arc_pt_num,device=device)
        self.arcl = torch.linspace(0.3,1,self.arc_pt_num,device=device)
        # self.workspace = [[0.0,0.0,0.0],[0.0,0.0,0.2]]
        self.n_disrete = 2         # total number of x: n_discrete**3
        self.batchsize = 200       # batch size of q
        # self.pose = torch.eye(4).unsqueeze(0).to(self.device).expand(self.batchsize,4,4).float()
        self.epsilon = 5e-3         # distance threshold to filter data

        # Robot
        with open("config.yaml", 'r') as stream:
            config = yaml.safe_load(stream)
        self.robot = TDCR(config['spline'], config['robot'], device)

    def compute_sdf(self,x,q,return_index = False, type = None):
        # x : (Nx,3)
        # q : (Nq,3n)
        # return_index : if True, return the index of link that is closest to x
        # return d : (Nq)
        # return idx : (Nq) optional

        if type == 'QS':
            qtau = q
            self.robot.solve_Cosserat_model(qtau)
            p = self.robot.get_position()
        else:
            p = self.curve.get_position(config=q) # Nq, n, 3
        # d2 = torch.sum(torch.square(x[None,:,None,:] - p[:,None,...]), dim=3) # Nq x Nx x n
        d = torch.linalg.vector_norm(x[None,:,None,:] - p[:,None,...], dim=3) # Nq x Nx x n
        if not return_index:
            d = d.reshape((q.shape[0],-1)).min(dim=1)[0]
            return d
        else:
            d_temp = d.min(dim=1)[0] # Nq x n
            # print(d)
            d,idx = d_temp.min(dim=1) # Nq
            idx = self.curve.get_basis_idx(site_idx=idx)
            return d,idx
        # pose = torch.eye(4).unsqueeze(0).to(self.device).expand(len(q),4,4).float()
        # if not return_index:
        #     d,_ = self.bp_sdf.get_whole_body_sdf_batch(x,pose, q,self.model,use_derivative =False)
        #     d = d.min(dim=1)[0]
        #     return d
        # else:
        #     d,_,idx = self.bp_sdf.get_whole_body_sdf_batch(x,pose, q,self.model,use_derivative =False,return_index = True)
        #     # idx: Nq x Nx
        #     d,pts_idx = d.min(dim=1)
        #     idx = idx[torch.arange(len(idx)),pts_idx]
        #     return d,idx

    def given_x_find_q(self,x,q = None, batchsize = None,return_mask = False,epsilon = 5e-3, type = None):
        # x : (N,3)
        # scale x to workspace
        if not batchsize:
            batchsize = self.batchsize

        def cost_function(q):
            #  find q that d(x,q) = 0
            # q : B,3n
            # x : N,3
            d = self.compute_sdf(x,q,return_index = False,type=type)
            cost = torch.sum(d**2)
            return cost

        t0 = time.time()
        # optimizer for data generation
        if type == 'QS':
            qtau =  torch.rand(batchsize,2).to(self.device)*(self.robot.qub[0]-self.robot.qlb[0])+self.robot.qlb[0]
            q = qtau
        else:     
            if q is None:
                q = torch.rand(batchsize,self.n*2).to(self.device)*(self.q_max-self.q_min)+self.q_min

        res = minimize(
            cost_function, 
            q, 
            method='l-bfgs', 
            options=dict(line_search='strong-wolfe'),
            max_iter=200,
            disp=0
            )
        
        d,idx = self.compute_sdf(x,res.x,return_index=True,type=type)
        d,idx = d.squeeze(),idx.squeeze()
        mask = torch.abs(d) < epsilon
        # q_valid,d,idx = res.x[mask],d[mask],idx[mask]
        boundary_mask = ((res.x > self.q_min) & (res.x < self.q_max)).all(dim=1)
        final_mask = mask & boundary_mask
        final_q,idx = res.x[final_mask],idx[final_mask]

        if type == 'QS':
            q = self.robot.c
            # Remove the zero dimension ([:, 2, :])
            q = q[:, :2, :]  # Shape: (N, 2, n)
            # Permute dimensions back (N, 2, n) -> (N, n, 2)
            q = torch.permute(q, (0, 2, 1))  # Shape: (N, n, 2)
            q = torch.reshape(q, (self.N, 2 * self.n))  # Shape: (N, 2n)
            boundary_mask = ((q > self.q_min) & (q < self.q_max)).all(dim=1)
            final_mask = mask & boundary_mask
            final_q,idx = q[final_mask],idx[final_mask]

        # q0 = q0[mask][boundary_mask]
        # print(res)
        # print(d)
        # print(mask)
        # print(boundary_mask)
        # pp = self.curve.T[:,:,0:3,3].cpu().detach().numpy()
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.scatter(0.0,0.0,0.1)
        # for i in range(pp.shape[0]):
        #     ax.plot(pp[i,:,0], pp[i,:,1], pp[i,:,2])
        # ax.axis('equal')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.show()

        print('number of q_valid: \t{} \t time cost:{}'.format(len(final_q),time.time()-t0))
        if return_mask:
            return final_mask,final_q,idx
        else:
            return final_q,idx

    def distance_q(self,x,q):
        # x : (Nx,3)
        # q : (Np,3n)
        # return d : (Np) distance between q and x in C space. d = min_{q*}{L2(q-q*)}. sdf(x,q*)=0

        # compute d
        Np = q.shape[0]
        q_template,link_idx = self.given_x_find_q(x,epsilon=self.epsilon)
        print(q_template.shape)

        if link_idx.min() == 0:
            return torch.zeros(Np).to(self.device)
        else:
            # link_idx[link_idx==7] = 6
            # link_idx[link_idx==8] = 7
            d = torch.inf*torch.ones(Np,self.n*3).to(self.device)
            for i in range(link_idx.min(),link_idx.max()+1):
                mask = (link_idx==i)
                d_norm = torch.norm(q[:,:i].unsqueeze(1)- q_template[mask][:,:i].unsqueeze(0),dim=-1)
                d[:,i-1] = torch.min(d_norm,dim=-1)[0]
        d = torch.min(d,dim=-1)[0]

        # compute sign of d
        # d_ts = self.compute_sdf(x,q)
        # mask =  (d_ts < 0)
        # d[mask] = -d[mask]
        return d 

    def projection(self,x,q):
        q.requires_grad = True
        d = self.distance_q(x,q)
        grad = torch.autograd.grad(d,q,torch.ones_like(d),create_graph=True)[0]
        q_new = q - grad*d.unsqueeze(-1)
        return q_new

    def generate_offline_data(self,datasave_path, type = None):
        
        # x = torch.linspace(self.workspace[0][0],self.workspace[1][0],self.n_disrete).to(self.device)
        # y = torch.linspace(self.workspace[0][1],self.workspace[1][1],self.n_disrete).to(self.device)
        # z = torch.linspace(self.workspace[0][2],self.workspace[1][2],self.n_disrete).to(self.device)
        # x,y,z = torch.meshgrid(x,y,z)
        tt = torch.mul(self.arcl[:,None],self.theta[None,1:])
        r = self.L/self.theta[1:]
        x = torch.cat((torch.zeros((len(self.arcl),1),device=self.device),torch.mul(r[None,:],1-torch.cos(tt))),dim=1)
        y = torch.zeros_like(x,device=self.device)
        z = torch.cat((self.arcl[:,None]*self.L,torch.mul(r[None,:],torch.sin(tt))),dim=1)
        pts = torch.stack([x,y,z],dim=-1).reshape(-1,3)
        # pts = torch.tensor([[0.0,0.0,0.1]],device=self.device)
        # pp = pts.cpu().detach().numpy()
        # ax = plt.figure().add_subplot(projection='3d')
        # ax.scatter(pp[:,0], pp[:,1], pp[:,2])
        # # for i in range(pp.shape[0]):
        # #     ax.plot(pp[i,:,0], pp[i,:,1], pp[i,:,2])
        # ax.axis('equal')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.show()
        data = {}
        for i,p in enumerate(pts):
            q,idx = self.given_x_find_q(p.unsqueeze(0),epsilon=self.epsilon, type=type) 
            print('q',q.shape)
            data[i] ={
                'x':    p.detach().cpu().numpy(),
                'q':    q.detach().cpu().numpy(),
                'idx':  idx.detach().cpu().numpy(),
            }
            print(f'point {i} finished, number of q: {len(q)}')
        np.save(datasave_path,data)

def analysis_data(x):
    # Compute the squared Euclidean distance between each row
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    diff = diff.pow(2).sum(-1)

    # Set the diagonal elements to a large value to exclude self-distance
    diag_indices = torch.arange(x.shape[0])
    diff[diag_indices, diag_indices] = float('inf')
    
    # Compute the Euclidean distance by taking the square root
    diff = diff.sqrt()
    min_dist = torch.min(diff,dim=1)[0]
    print(f'distance\tmax:{min_dist.max()}\tmin:{min_dist.min()}\taverage:{min_dist.mean()}')


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.set_default_device(device) # does not work with torch.vander
    torch.autograd.set_detect_anomaly(True)

    gen = DataGenerator(device)
    # x = torch.tensor([[0.5,0.5,0.5]]).to(device)
    # gen.single_point_generation(x)

    # datasave_path = os.path.join(CUR_DIR,'data_xx.npy')
    # gen.generate_offline_data(datasave_path,type=None)

    datasave_path = os.path.join(CUR_DIR,'data_QS_121.npy')
    gen.generate_offline_data(datasave_path,type='QS')
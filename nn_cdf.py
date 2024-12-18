# 7D panda robot

import numpy as np
import os
import sys
import torch
import math
import time
CUR_PATH = os.path.dirname(os.path.realpath(__file__))
from mlp import MLPRegression
sys.path.append(os.path.join(CUR_PATH,'../../RDF'))
# from panda_layer.panda_layer import PandaLayer
# import bf_sdf
from splineCurve3D import splineCurve3D
import matplotlib.pyplot as plt

PI = math.pi
# torch.manual_seed(10)
np.random.seed(10)
# torch.autograd.set_detect_anomaly(True)

class CDF:
    def __init__(self,device,datafile_prefix=None) -> None:
        # device
        self.device = device  

        self.batch_x = 10
        self.batch_q = 100
        self.max_q_per_link = 100

        # panda robot
        # self.panda = PandaLayer(device)
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

        self.datafile_prefix = datafile_prefix
        self.raw_data = np.load(os.path.join(CUR_PATH,'data_'+self.datafile_prefix+'.npy'),allow_pickle=True).item()
        self.data_path = os.path.join(CUR_PATH,'data_'+self.datafile_prefix+'.pt') 
        # uncomment these lines to process the generated data and train your own CDF
        self.process_data(self.raw_data,self.data_path)
        self.data = self.load_data(self.data_path)
        self.len_data = len(self.data['k'])

    def process_data(self,data, data_path):
        import pytorch3d.ops 
        keys = list(data.keys())  # Create a copy of the keys
        processed_data = {}
        for k in keys:
            if len(data[k]['q']) == 0:
                data.pop(k)
                continue
            q = torch.from_numpy(data[k]['q']).float().to(self.device)
            q_idx = torch.from_numpy(data[k]['idx']).float().to(self.device)
            # q_idx[q_idx==7] = 6
            # q_idx[q_idx==8] = 7
            q_lib = torch.inf*torch.ones(self.max_q_per_link,2*self.n,2*self.n).to(self.device)
            for i in range(2*self.n):
                mask = (q_idx==i)
                if len(q[mask])>self.max_q_per_link:
                    fps_q = pytorch3d.ops.sample_farthest_points(q[mask].unsqueeze(0),K=self.max_q_per_link)[0]
                    q_lib[:,:,i-1] = fps_q.squeeze()
                    # q_lib[:,:,i-1] = (q[mask])[:self.max_q_per_link]
                    # print(q_lib[:,:,i]) 
                elif len(q[mask])>0:
                    q_lib[:len(q[mask]),:,i-1] = q[mask]
            processed_data[k] = {
                'x':torch.from_numpy(data[k]['x']).float().to(self.device),
                'q':q_lib,
            }
        final_data = {
            'x': torch.cat([processed_data[k]['x'].unsqueeze(0) for k in processed_data.keys()],dim=0),
            'q': torch.cat([processed_data[k]['q'].unsqueeze(0) for k in processed_data.keys()],dim=0),
            'k':torch.tensor([k for k in processed_data.keys()]).to(self.device)
        }

        torch.save(final_data,data_path)
        return data
    
    def load_data(self,path):
        data = torch.load(path)
        return data

    def select_data(self):
        # x_batch:(batch_x,3)
        # q_batch:(batch_q,7)
        # d:(batch_x,batch_q)
        
        x = self.data['x']
        q = self.data['q']

        idx = torch.randint(0,len(x),(self.batch_x,)) 
        # idx = torch.tensor([4000])
        x_batch,q_lib = x[idx],q[idx]
        # print(x_batch)
        q_batch = self.sample_q()   
        # d,grad = self.decode_distance(x_batch,q_batch,q_lib)
        d,grad = self.decode_distance(q_batch,q_lib)
        return x_batch,q_batch,d,grad

    def decode_distance(self,q_batch,q_lib):
        # batch_q:(batch_q,7)
        # q_lib:(batch_x,self.max_q_per_link,7,7)

        batch_x = q_lib.shape[0]
        batch_q = q_batch.shape[0]
        d_tensor = torch.ones(batch_x,batch_q,2*self.n).to(self.device)*torch.inf
        grad_tensor  = torch.zeros(batch_x,batch_q,2*self.n,2*self.n).to(self.device)
        for i in range(2*self.n):
            q_lib_temp = q_lib[:,:,:i+1,i].reshape(batch_x*self.max_q_per_link,-1).unsqueeze(0).expand(batch_q,-1,-1)
            q_batch_temp = q_batch[:,:i+1].unsqueeze(1).expand(-1,batch_x*self.max_q_per_link,-1)
            d_norm = torch.norm((q_batch_temp - q_lib_temp),dim=-1).reshape(batch_q,batch_x,self.max_q_per_link)

            d_norm_min,d_norm_min_idx = d_norm.min(dim=-1)
            grad = torch.autograd.grad(d_norm_min.reshape(-1),q_batch_temp,torch.ones_like(d_norm_min.reshape(-1)),retain_graph=True)[0]
            grad_min_q = grad.reshape(batch_q,batch_x,self.max_q_per_link,-1).gather(2,d_norm_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,i+1))[:,:,0,:]
            grad_tensor[:,:,:i+1,i] = grad_min_q.transpose(0,1)
            d_tensor[:,:,i] = d_norm_min.transpose(0,1)

        d,d_min_idx = d_tensor.min(dim=-1)
        grad_final = grad_tensor.gather(3,d_min_idx.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,2*self.n,2*self.n))[:,:,:,0]
        return d, grad_final
    
    def sample_q(self,batch_q = None):
        if batch_q is None:
            batch_q = self.batch_q
        q_sampled = self.q_min + torch.rand(batch_q,2*self.n).to(self.device)*(self.q_max-self.q_min)
        q_sampled.requires_grad = True
        return q_sampled
    
    def projection(self,q,d,grad):
        q_new = q - grad*d.unsqueeze(-1)
        return q_new

    def train_nn(self,epoches=500,model=None):

        # model
        # input: [x,q] (B,3+7)

        # model = MLPRegression(input_dims=3+2*self.n, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5000,
                                                        threshold=0.05, threshold_mode='rel',
                                                        cooldown=0, min_lr=0, eps=1e-04, verbose=True)
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        COSLOSS = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        model_dict = {}

        # List to store losses for plotting
        losses = []
        iter_list = []
        # Set up interactive plot
        plt.ion()
        fig, ax = plt.subplots()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss over Epochs')
        line, = ax.plot([], [], label='Training Loss')
        ax.legend()

        plt.show(block=False)

        for iter in range(epoches):
            model.train()
            with torch.cuda.amp.autocast():
                x_batch,q_batch,d,gt_grad = self.select_data()

                x_inputs = x_batch.unsqueeze(1).expand(-1,self.batch_q,-1).reshape(-1,3)
                q_inputs = q_batch.unsqueeze(0).expand(self.batch_x,-1,-1).reshape(-1,2*self.n)

                inputs = torch.cat([x_inputs,q_inputs],dim=-1)
                outputs = d.reshape(-1,1)
                gt_grad = gt_grad.reshape(-1,2*self.n)

                weights = torch.ones_like(outputs).to(device)
                # weights = (1/outputs).clamp(0,1)

                d_pred = model.forward(inputs)
                d_grad_pred = torch.autograd.grad(d_pred, q_inputs, torch.ones_like(d_pred), retain_graph=True,create_graph=True)[0]

                # Compute the Eikonal loss
                eikonal_loss = torch.abs(d_grad_pred.norm(2, dim=-1) - 1).mean()

                # Compute the tension loss
                # print(d_grad_pred.shape)
                # print(q_inputs.shape)
                # print(d_pred.shape)
                dd_grad_pred = torch.autograd.grad(d_grad_pred, q_inputs, torch.ones_like(d_grad_pred), retain_graph=True,create_graph=True)[0]

                # gradient loss
                gradient_loss = (1 - COSLOSS(d_grad_pred, gt_grad)).mean()
                # tension loss
                tension_loss = dd_grad_pred.square().sum(dim=-1).mean()
                # Compute the MSE loss
                d_loss = ((d_pred-outputs)**2*weights).mean()

                # Combine the two losses with appropriate weights
                w0 = 5.0
                w1 = 50 # 0.01
                w2 = 50 # 0.01
                w3 = 50 # 0.1
                loss = w0 * d_loss + w1 * eikonal_loss + w2 * tension_loss + w3 * gradient_loss

                # # Print the losses for monitoring

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step(loss)
                if iter % 10 == 0 and iter>10:
                    print(f"Epoch:{iter}\tMSE Loss: {d_loss.item():.3f}\tEikonal Loss: {eikonal_loss.item():.3f}\tTension Loss: {tension_loss.item():.3f}\tGradient Loss: {gradient_loss.item():.3f}\tTotal loss:{loss.item():.3f}\tTime: {time.strftime('%H:%M:%S', time.gmtime())}")
                    model_dict[iter] = model.state_dict()
                    torch.save(model_dict, os.path.join(CUR_PATH,'model_dict_'+self.datafile_prefix+'.pt'))

                    # Update plot dynamically
                    losses.append(loss.item())
                    iter_list.append(iter)

                    line.set_xdata(iter_list)
                    line.set_ydata(losses)
                    ax.relim()
                    ax.autoscale_view()
                    fig.canvas.flush_events()
                # plt.draw()
                # plt.pause(0.01)  # Pause to allow plot to update


        plt.ioff()
        # plt.show()

        return model
    
    def inference(self,x,q,model):
        model.eval()
        x,q = x.to(self.device),q.to(self.device)
        # q.requires_grad = True
        # print(x.shape)
        x_cat = x.unsqueeze(1).expand(-1,len(q),-1).reshape(-1,3)
        q_cat = q.unsqueeze(0).expand(len(x),-1,-1).reshape(-1,2*self.n)
        inputs = torch.cat([x_cat,q_cat],dim=-1)
        cdf_pred = model.forward(inputs)
        return cdf_pred
    
    def inference_d_wrt_q(self,x,q,model,return_grad = True):
        cdf_pred = self.inference(x,q,model)
        d = cdf_pred.reshape(len(x),len(q)).min(dim=0)[0]
        if return_grad:
            grad = torch.autograd.grad(d,q,torch.ones_like(d),retain_graph=True,create_graph=True)[0]
            # dgrad = torch.autograd.grad(grad,q,torch.ones_like(grad),retain_graph=True,create_graph=True)[0]
            return d,grad
        else:
            return d

    def eval_nn(self,model,num_iter = 1):
        eval_time = False
        eval_acc = True
        if eval_time:
            x = torch.rand(100,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
            q = self.sample_q(batch_q=100)
            time_cost_list = []
            for i in range(100):
                t0 = time.time()
                d = self.inference_d_wrt_q(x,q,model,return_grad = False)
                t1 = time.time()
                grad = torch.autograd.grad(d,q,torch.ones_like(d),retain_graph=True,create_graph=True)[0]
                q_proj = self.projection(q,d,grad)
                t2 = time.time()
             
                if i >0:
                    time_cost_list.append([t1-t0,t2-t1])
            mean_time_cost = np.mean(time_cost_list,axis=0)
            print(f'inference time cost:{mean_time_cost[0]}\t projection time cost: {mean_time_cost[1]}')

        if eval_acc:
            # bp_sdf model
            # bp_sdf = bf_sdf.BPSDF(8,-1.0,1.0,self.panda,device)
            # bp_sdf_model = torch.load(os.path.join(CUR_PATH,'../../RDF/models/BP_8.pt'))

            res = []
            for i in range (1000):
                # x = torch.rand(1,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
                arcl = 0.3 + 0.7*torch.rand(1).to(device)
                theta = PI*0.8*torch.rand(1).to(device)
                r = self.L/theta
                x = torch.tensor([[r*(1-torch.cos(theta*arcl)),0.0,r*torch.sin(theta*arcl)]],device=self.device)
                q = self.sample_q(batch_q=1)
                p0 = self.curve.get_position(config=q)
                pp0 = self.curve.T[:,:,0:3,3].cpu().detach().numpy()
                for _ in range (num_iter):
                    d,grad = self.inference_d_wrt_q(x,q,model,return_grad = True)
                    q = self.projection(q,d,grad)
                q,grad = q.detach(),grad.detach()   # release memory
                # pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(self.device).float()
                # sdf,_ = bp_sdf.get_whole_body_sdf_batch(x, pose, q, bp_sdf_model,use_derivative=False)
                p = self.curve.get_position(config=q) # Nq, n, 3
                sdf = torch.linalg.vector_norm(x[None,:,None,:] - p[:,None,...], dim=3) # Nq x Nx x n
                sdf,_ = torch.min(sdf.reshape((q.shape[0],-1)), dim=1)
                
                # print('sdf.shape', sdf.shape)

                xx = x.cpu().detach().numpy()
                pp = self.curve.T[:,:,0:3,3].cpu().detach().numpy()
                ax = plt.figure().add_subplot(projection='3d')
                r = 5e-3  # Radius of the ball
                center = (xx[:,0], xx[:,1], xx[:,2])  # Center of the ball (x, y, z)
                # Create a meshgrid for the sphere
                phi, theta = np.mgrid[0:np.pi:100j, 0:2*np.pi:100j]
                x_plot = r * np.sin(phi) * np.cos(theta) + center[0]
                y_plot = r * np.sin(phi) * np.sin(theta) + center[1]
                z_plot = r * np.cos(phi) + center[2]
                # Plotting the ball
                # ax.scatter(xx[:,0],xx[:,1],xx[:,2],c='r',marker='o',size=10)
                p_plot = p.cpu().detach().numpy()
                p0_plot = p0.cpu().detach().numpy()
                for i in range(p.shape[0]):
                    ax.plot(p_plot[i,:,0], p_plot[i,:,1], p_plot[i,:,2],label='projected')
                    ax.plot(p0_plot[i,:,0], p0_plot[i,:,1], p0_plot[i,:,2],label='original')
                ax.legend()
                ax.plot_surface(x_plot, y_plot, z_plot, color='r', alpha=0.6, label='point')
                # ax.legend()
                ax.axis('equal')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                plt.show()
                
                error = sdf.reshape(-1).abs()
                MAE = error.mean()
                RMSE = torch.sqrt(torch.mean(error**2))
                SR = (error<0.03).sum().item()/len(error)
                res.append([MAE.item(),RMSE.item(),SR])
                print(f'iter {i} finished, MAE:{MAE}\tRMSE:{RMSE}\tSR:{SR}')
            res = np.array(res)
            print(f'MAE:{res[:,0].mean()}\tRMSE:{res[:,1].mean()}\tSR:{res[:,2].mean()}')
            print(f'MAE:{res[:,0].std()}\tRMSE:{res[:,1].std()}\tSR:{res[:,2].std()}')

    # def eval_nn_noise(self,model,num_iter = 3):
    #         bp_sdf = bf_sdf.BPSDF(8,-1.0,1.0,self.panda,device)
    #         bp_sdf_model = torch.load(os.path.join(CUR_PATH,'../../RDF/models/BP_8.pt'))

    #         res = []
    #         for i in range (1000):
    #             x = torch.rand(1,3).to(device)-torch.tensor([[0.5,0.5,0]]).to(device)
    #             noise = torch.normal(0,0.03,(1,3)).to(device)
    #             x_noise = x + noise
    #             q = self.sample_q(batch_q=1000)
    #             for _ in range (num_iter):
    #                 d,grad = self.inference_d_wrt_q(x_noise,q,model,return_grad = True)
    #                 q = self.projection(q,d,grad)
    #             q,grad = q.detach(),grad.detach()   # release memory
    #             pose = torch.eye(4).unsqueeze(0).expand(len(q),-1,-1).to(self.device).float()
    #             sdf,_ = bp_sdf.get_whole_body_sdf_batch(x, pose, q, bp_sdf_model,use_derivative=False)
                
    #             error = sdf.reshape(-1).abs()
    #             MAE = error.mean()
    #             RMSE = torch.sqrt(torch.mean(error**2))
    #             SR = (error<0.03).sum().item()/len(error)
    #             res.append([MAE.item(),RMSE.item(),SR])
    #             print(f'iter {i} finished, MAE:{MAE}\tRMSE:{RMSE}\tSR:{SR}')
    #         res = np.array(res)
    #         print(f'MAE:{res[:,0].mean()}\tRMSE:{res[:,1].mean()}\tSR:{res[:,2].mean()}')
    #         print(f'MAE:{res[:,0].std()}\tRMSE:{res[:,1].std()}\tSR:{res[:,2].std()}')


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    datafile_prefix='121(2)'
    cdf = CDF(device,datafile_prefix=datafile_prefix)

    # model = MLPRegression(input_dims=3+2*12, output_dims=1, mlp_layers=[1024, 512, 256, 128, 128],skips=[], act_fn=torch.nn.ReLU, nerf=True)
    model = MLPRegression(input_dims=3+2*12, output_dims=1, mlp_layers=[2048, 1024, 1024, 512, 512, 128],skips=[2,4], act_fn=torch.nn.ReLU, nerf=True)

    # cdf.train_nn(epoches=50000,model=model)
    
    model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict_'+datafile_prefix+'.pt'))[49900])
    # model.load_state_dict(torch.load(os.path.join(CUR_PATH,'model_dict.pt'))[49900])
    model.to(device)
    cdf.eval_nn(model,num_iter = 5)
import torch
import numpy as np
from scipy.interpolate import BSpline
# import matplotlib.pyplot as plt

def expSE3(Psi,device):
    # exponential map se3 to SE3, refer to Muller's review
    # Psi: N x 6 np array
    N = Psi.shape[0]
    # print(Psi._version)
    z11 = torch.zeros((N,1,1), device=device) # N x 1 x 1
    Psiw = Psi[:,0:3]#.clone() # N x 3
    Psiv = Psi[:,3:6]#.clone() # N x 3
    # print(Psiv._version)
    # theta = torch.sqrt(torch.sum(Psiw**2, axis=1))[:,None] # N x 1
    theta = torch.linalg.vector_norm(Psiw, dim=1, keepdim=True) # N x 1
    # print(theta)
    zeroidx = torch.flatten(theta != 0)
    n = torch.zeros_like(Psiw, device=device)
    n[zeroidx,:] = torch.divide(Psiw[zeroidx,:], theta[zeroidx,:]) # N x 3

    n_hat = torch.cat((torch.cat((z11, -n[:,None,None,2], n[:,None,None,1]),2),
                    torch.cat((n[:,None,None,2], z11, -n[:,None,None,0]),2),
                    torch.cat((-n[:,None,None,1], n[:,None,None,0], z11),2)),1) # N x 3 x 3
    n_hat2 = torch.matmul(n_hat, n_hat) # N x 3 x 3
    alpha = torch.zeros_like(theta, device=device)
    alpha[zeroidx,:] = torch.divide(torch.sin(theta[zeroidx,:]), theta[zeroidx,:]) # N x 1
    beta = torch.zeros_like(theta, device=device)
    beta[zeroidx,:] = torch.divide(1-torch.cos(theta[zeroidx,:]), theta[zeroidx,:]**2) # N x 1
    expPsiw = torch.eye(3,device=device)[None,:,:] + alpha[:,:,None]*n_hat*theta[:,:,None] + beta[:,:,None]*n_hat2*(theta**2)[:,:,None] # (2.6) N x 3 x 3
    dexpPsiw = torch.eye(3,device=device)[None,:,:] + beta[:,:,None]*n_hat*theta[:,:,None] + (1-alpha)[:,:,None]*n_hat2 # (2.13) N x 3 x 3
    expPsiv = torch.matmul(dexpPsiw, Psiv[:,:,None]) # (2.27) N x 3 x 1
    return torch.cat((torch.cat((expPsiw, expPsiv),2), torch.cat((torch.zeros((N,1,3),device=device), torch.ones((N,1,1),device=device)),2)),1) # N x 4 x 4

class splineCurve3D:
    def __init__(self, t, k, L, sites, device):
        # c: n x batch_size x 3
        # super().__init__(t, c, k, axis=0)
        self.device = device
        self.t = t
        self.c = None
        self.k = k
        self.L = L
        # self.n, self.N, _ = c.shape
        self.n = len(t) - k - 1
        self.N = 0
        self.s = sites # collocation sites
        self.h = sites[1:] - sites[:-1]
        self.tq = torch.tensor([1/2-torch.sqrt(torch.tensor(15))/10, 1/2, 1/2+torch.sqrt(torch.tensor(15))/10]) # Legendre zeros as quadrature points
        q = torch.reshape(sites[:-1,None] + self.h[:,None]*self.tq[None,:], (3*len(self.h),))
        V = torch.fliplr(torch.vander(self.tq-1/2)) # 3 x 3
        self.invV = torch.linalg.inv(V).to(self.device)
        self.T = None
        self.Tq = None
        # self.T = torch.zeros((self.N,len(self.s),4,4), device=self.device)#.to(self.device)
        # self.T[:,0,:,:] = torch.eye(4,device=self.device)[None,:,:]
        # self.dT = []

        # collocation
        self.S = torch.zeros((self.n,len(sites)), device=device)
        self.Q = torch.zeros((self.n,(len(sites)-1)*3), device=device)
        self.D = torch.zeros_like(self.S,device=device)
        for i in range(self.n):
            b = BSpline.basis_element(t[i:i+k+2])
            db = b.derivative(nu=1)
            active_s = torch.logical_and(sites>=t[i], sites<=t[i+k+1])
            active_q = torch.logical_and(q>=t[i], q<=t[i+k+1])
            self.S[i,active_s] = torch.from_numpy(b(sites[active_s])).float().to(self.device)
            self.Q[i,active_q] = torch.from_numpy(b(q[active_q])).float().to(self.device)
            self.D[i,active_s] = torch.from_numpy(db(sites[active_s])).float().to(self.device)
        self.S[-1,-1] = 1
        # self.uc = torch.permute(c,(1,2,0))@self.S[None,:,:] # N x 3 x len(sites)
        self.uc = None
        self.t = self.t.to(device)

    def Magnus_expansion(self,Xq,h):
        # Xq: N x 3 x 6
        z1 = torch.zeros((self.N,3,1,1),device=self.device)
        z3 = torch.zeros((self.N,3,3,3),device=self.device)
        B = self.invV[None,:,:]@Xq*h # N x 3 x 6 self-adjoint basis
        w_hat = torch.cat((torch.cat((z1, -B[:,:,None,None,2], B[:,:,None,None,1]), 3),
            torch.cat((B[:,:,None,None,2], z1, -B[:,:,None,None,0]), 3),
            torch.cat((-B[:,:,None,None,1], B[:,:,None,None,0], z1), 3)), 2) # N x v x 3 x 3
        v_hat = torch.cat((torch.cat((z1, -B[:,:,None,None,5], B[:,:,None,None,4]), 3),
            torch.cat((B[:,:,None,None,5], z1, -B[:,:,None,None,3]), 3),
            torch.cat((-B[:,:,None,None,4], B[:,:,None,None,3], z1), 3)), 2) # N x v x 3 x 3
        adB = torch.cat((torch.cat((w_hat, z3),3), torch.cat((v_hat, w_hat),3)), 2) # N x v x 6 x 6

        B12 = adB[:,0,:,:]@B[:,1,:,None] # N x 6 x 1
        B23 = adB[:,1,:,:]@B[:,2,:,None]
        B13 = adB[:,0,:,:]@B[:,2,:,None]
        B113 = adB[:,0,:,:]@B13
        B212 = adB[:,1,:,:]@B12
        B112 = adB[:,0,:,:]@B12
        B1112 = adB[:,0,:,:]@B112

        Psi = B[:,0,:] + B[:,2,:]/12 + torch.squeeze(B12/12 - B23/240 + B113/360 - B212/240 - B1112/720) # N x 6
        
        return Psi

    def integrate_SE3(self,quadrature_pose=False):
        # print('integrate_se3')
        # self.T = torch.zeros((self.N,len(self.s),4,4), device=self.device)#.to(self.device)
        # self.T[:,0,:,:] = torch.eye(4,device=self.device)[None,:,:]
        self.T = torch.tile(torch.eye(4,device=self.device),(self.N,len(self.s),1,1))
        # print(self.T._version)
        # self.T = torch.tile(torch.eye(4,device=self.device),(self.N,1,1,1))
        # self.dT = np.empty((self.n,self.N,3,len(self.s),4,4))
        h = self.h*self.L
        uq = self.c @ self.Q[None,:,:] # N x 3 x vnk, strain at quadrature points
        # Xq = torch.zeros((self.N,3,6,len(self.s)-1), device=self.device)
        # Psi = torch.zeros((self.N,6,len(self.s)-1), device=self.device)
        # expPsit = torch.zeros((self.N,4,4,len(self.s)-1), device=self.device)
        
        # zz = np.zeros((self.n,self.N,3,1,1))
        if quadrature_pose:
            self.Tq = torch.zeros((self.N,3*(len(self.s)-1),4,4),device=self.device)
            v = torch.cat((torch.zeros((self.N,2),device=self.device),torch.ones((self.N,1),device=self.device)),dim=1)

        for k in range(len(self.s)-1): # forward kinematics propagation, calculate SE3 transformation and derivatives
            # snapshots at quadrature point
            Xq = torch.cat((torch.permute(uq[:,:,k*3:k*3+3], (0,2,1)), torch.tile(torch.tensor([0, 0, 1],device=self.device),(self.N,3,1))), 2) # N x v x 6

            # Calculate Magnus expansion
            Psi = self.Magnus_expansion(Xq,h[k]) # N x 6

            # exponential map se3 to SE3
            # expPsit[...,k] = expSE3(Psi[...,k],self.device)

            # Integrate on SE3
            # a = self.T[:,k,:,:].clone()
            # b = expPsit[...,k].clone()
            self.T[:,k+1,:,:] = torch.matmul(self.T[:,k,:,:].clone(),expSE3(Psi.clone(),self.device)) #torch.rand((4,4),device=self.device)[None,:,:]
            # self.T = torch.cat((self.T,(self.T[:,k,:,:]@expPsit[...,k])[:,None,:,:]),dim=1)

            if quadrature_pose:
                # Psiq1 = (Xq(1,:)' + [Uc(:,k);0;0;1])/2*GV.hsp(k)*(1/2-sqrt(15)/10);
                # Psiq2 = (Xq(2,:)' + Xq(1,:)')/2*GV.hsp(k)*sqrt(15)/10;
                # Psiq3 = (Xq(3,:)' + [Uc(:,k+1);0;0;1])/2*GV.hsp(k)*(1/2-sqrt(15)/10);
                Psiq = torch.cat(((Xq[:,1,:] + torch.cat((self.uc[:,:,k],v),dim=1))/2*self.tq[0]*h[k], # N x 6
                                (Xq[:,2,:] + Xq[:,1,:])/2*(1/2-self.tq[0])*h[k],
                                -(Xq[:,3,:] + torch.cat((self.uc[:,:,k+1],v),dim=1))/2*self.tq[0]*h[k]), dim=0)
                expPsiq = expSE3(Psiq)
                self.Tq[:,k*3,:,:] = self.T[:,k,:,:]*expPsiq[:self.N,:,:]
                self.Tq[:,k*3+1,:,:] = self.Tq[:,k*3-2,:,:]*expPsiq[self.N:2*self.N,:,:]
                self.Tq[:,k*3+2,:,:] = self.T[:,k+1,:,:]*expPsiq[2*self.N:,:,:]

    def get_position(self, config=None, sites=np.linspace(0,1,19)):
        # config: N x (nx2), no torsion for now
        if config is None and self.c is None:
            print('Coefficients not given.')
            return
        if config is not None:
            self.N = config.shape[0]
            self.c = torch.cat((torch.permute(torch.reshape(config, (self.N, self.n, 2)),(0,2,1)),torch.zeros((self.N,1,self.n),device=self.device)), dim=1) # N x 3 x n
            self.uc = self.c @ self.S[None,:,:] # N x 3 x len(sites)
            # self.T = None
            self.integrate_SE3()
        # if self.T is None:
        #     self.integrate_SE3()
        p = torch.zeros((self.N, len(sites), 3), device=self.device)
        seg = torch.cat((torch.tensor([0]), self.s[:-1] + self.h/2)) # segment the length such that collocation points are centered in intervals
        for k in range(len(sites)):
            idx = torch.nonzero(sites[k]>=seg)[-1][0] # index of reference collocation point
            ds = (sites[k] - self.s[idx])*self.L
            Psii = ds*torch.hstack((self.uc[:,:,idx], torch.zeros((self.N,2),device=self.device), torch.ones((self.N,1),device=self.device))) # N x 6
            # print('get_position')
            # expPsi = expSE3(Psi,self.device) # N x 4 x 4
            # print(Psi)
            # Tk = self.T[:,idx,:,:]@expSE3(Psi,self.device) # N x 4 x 4
            p[:,k,:]  = (self.T[:,idx,:,:]@expSE3(Psii,self.device))[:,0:3,3]
            # p[:,k,:]  = (self.T[:,idx,:,:])[:,0:3,3]
        return p
    
    def get_basis_idx(self, site_idx, sites=torch.linspace(0,1,19)):
        sites = sites.to(self.device)
        s = sites[site_idx]
        k_idx = torch.logical_and(torch.gt(s[:,None],self.t[None,:-1]),torch.le(s[:,None],self.t[None,1:])).nonzero()
        idx = torch.zeros(len(s),dtype=torch.long,device=self.device)
        idx[k_idx[:,0]] = k_idx[:,1]
        return idx

def zero_force(p):
    # p: ... x 3
    return 0*p

class TDCR(splineCurve3D):
    def __init__(self, t, k, L, sites, device, fe=zero_force):
        super().__init__(t, k, L, sites, device)
        self.fe = fe
        # read config file

    # def solve_Cosserat_model(self):

        # function [T,dpdc,lambda,coeffs,res,exitflag,dedxi,dedq,dpdL,dnbdc,dnbbdL] = collocation_spline_TDK(qa,fe,Fe,Me,init_guess,GV)
        # Implementation based on cubic B-spline
        # https://www.math.ucdavis.edu/~bremer/classes/fall2018/MAT128a/lecture15.pdf
        # Nov 2024, Yifan Wang
        # Kirchhoff tendon-driven robot

        # % tau = zeros(GV.num_tendons,1);
        # % if qa(1) >= 0, tau(1) = qa(1); else, tau(3) = -qa(1); end
        # % if qa(2) >= 0, tau(2) = qa(2); else, tau(4) = -qa(2); end
        # tau = [qa(1:2);-qa(1:2)]; % push-pull
        # % hsp0 = GV.hsp; D0 = GV.D;
        # GV.hsp = GV.hsp*qa(3); % 1xn
        # GV.D = GV.D/qa(3);
        # GV.dUdotdc = GV.dUdotdc/qa(3);
        # %     counter = 0;
        # global T dT Tq dTq lambda;
        # T = zeros(4,4,GV.n+1); T(:,:,1) = eye(4);
        # dT = zeros(4,4,3*(GV.dof),GV.n+1);
        # Tq = zeros(4,4,GV.n*3);
        # dTq = zeros(4,4,3*(GV.dof),GV.n*3); % 4x4x3dofxvn
        # % [res,dedxi] = collocation_error(init_guess); exitflag = true; xic = init_guess;

        # [coeffs,res,exitflag,output,dedxi] = fsolve(@(coeffs) collocation_sp_error(coeffs), init_guess, GV.options_sp);


    def collocation_error(self):
        #  c: N x 3 x n spline coefficients stacked

        # forward kinematics
        self.integrate_SE3(quadrature_pose=True)

        # Setup tendon linear system
        gx = torch.zeros((self.N,3,len(self.s)),device=self.device)

        # quadrature for integration of distributed external force
        fe = self.fe(self.Tq[:,:,0:2,3]) # N x 3(n-1) x 3
        intfe = torch.cat((self.L*self.h[None,:,None]*(5*(fe[:,0::3,:] + fe[:,2::3,:]) + 8*fe[:,1::3,:])/18,\
                            torch.zeros((self.N,1,3),device=self.device)),dim=1)
        Intfe = torch.cumsum(intfe.flip(1),dim=1)

        # u = self.uc[...,i] # N x 3 x len(sites)
        
        # a = torch.zeros((self.N,len(self.s),3),device=self.device)
        # b = torch.zeros_like(a,device=self.device)
        # A = torch.zeros((self.N,len(self.s),3,3),device=self.device)
        # #G = zeros(3,3)
        # H = torch.zeros_like(A,device=self.device)
        # nbt = torch.zeros_like(a,device=self.device)
        e3 = torch.tensor([0,0,1],device=self.device)
        I3 = torch.eye(3,device=self.device)
        meL = torch.zeros(self.N,3)

        # zz = torch.zeros((self.N,1,1,len(self.s)))
        # u_hat = torch.cat((torch.cat((zz,-self.uc[:,2,None,:],self.uc[:,1,None,:]),dim=2),
        #                    torch.cat((self.uc[:,2,None,:],zz,-self.uc[:,0,None,:]),dim=2),
        #                    torch.cat((-self.uc[:,1,None,:],self.uc[:,0,None,:],zz),dim=2)),dim=1)

        for i in range(len(self.s),-1,-1): # for each collocation point, backward
            
            # if i == len(self.s):
            #     meL = R'@Me
            a = torch.zeros((self.N,3),device=self.device)
            b = torch.zeros_like(a,device=self.device)
            A = torch.zeros((self.N,3,3),device=self.device)
            #G = zeros(3,3)
            H = torch.zeros_like(A,device=self.device)
            nbt = torch.zeros_like(a,device=self.device)

            for j in range(self.n_tendons): # these are all "local" variables
                pb_si = torch.cross(self.uc[:,:,i],self.r[j,:].expand(self.N,3),dim=1) + e3[None,:] # N x 3
                pb_s_norm = torch.linalg.vector_norm(pb_si,dim=1)
                Fb_j = -self.tau[:,j]*pb_si/pb_s_norm # N x 3
                nbt -= Fb_j

                A_j = -(pb_si[:,:,None]@pb_si[:,None,:] - torch.square(pb_s_norm[:,None,None])*I3[None,...]) * (self.tau[:,j]/torch.pow(pb_s_norm,3)) # N x 3 x 3
                G_j = -A_j @ self.r_hat[j,:,:]
                a_j = A_j @ torch.cross(self.uc[:,:,i],pb_si,dim=1)

                a = a + a_j
                b = b + torch.cross(self.r[None,j,:], a_j, dim=1)
                A = A + A_j
                #G = G + G_j;
                H = H + self.r_hat[j,:,:]@G_j

                if i == len(self.s)-1: # boundary condition
                    meL = meL + torch.cross(self.r[None,j,:], Fb_j, dim=1)

            K = H + self.Kbt[None,:,:] # Nx3x3
            nb = -nbt + self.T[:,i,...].transpose(-2,-1)@Intfe[:,len(self.s)-i-1,:] # Nx3 not local
            mb = self.Kbt[None,:,:]@self.uc[:,:,i] # Nx3

            # Calculate ODE terms
            gx[:,:,i] = -torch.linalg.solve(K, torch.cross(self.uc[:,:,i],mb,dim=1) + torch.cross(e3.expand_as(nb),nb) + b) # Nx3

        # lambda = [mb;nb]
        
        # Assemble collocation and boundary errors
        bL = torch.linalg.solve(self.Kbt[None,:,:],meL) # N x 3

        E = torch.cat((self.c@self.D[None,:,:], self.uc[:,:,-1,None]), dim=2) - torch.cat((gx, bL[:,None,:]), dim=2) # N x 3 x len(s)+1
        return torch.sum(torch.square(E)) # careful here


# def main():
#     k = 3 # degree
#     n = 12 # number of control points
#     breaks = torch.linspace(0, 1, k+n+1-2*k)
#     t = torch.cat((torch.zeros(k), breaks, torch.ones(k)))# knots
#     c1 = torch.vstack((torch.ones(n)*20, torch.zeros(n), torch.zeros(n)))
#     c2 = torch.vstack((torch.zeros(n), torch.ones(n)*20, torch.zeros(n)))
#     c = torch.stack((c1.T,c2.T), axis=1) # n x N x 3
#     L = 0.02
#     sites = torch.cat((torch.tensor([0]), breaks[:-1] + (breaks[1:] - breaks[:-1])/2, torch.tensor([1])))
#     spl = splineCurve3D(t, c, k, L, sites)
#     spl.integrate_SE3()
#     p = spl.get_position(sites=sites) # N x n x 3
#     # print(spl([0.2, 0.3])) # N x 3 x n_query
#     # print(p)
#     # print(spl.Q)
#     # print(spl.uc)
#     # print(spl.T[0,-1,:,:])
#     # print(spl.T[1,-1,:,:])

#     # b0 = BSpline.basis_element(t[0:0+k+2])
#     # bn = BSpline.basis_element(t[n-1:n-1+k+2])
#     # x0 = np.linspace(t[0], t[0+k+1], 100)
#     # xn = np.linspace(t[n-1], t[n-1+k+1], 100)
#     # fig, ax = plt.subplots()
#     # ax.plot(x0, b0(x0), 'g', lw=3)
#     # ax.plot(xn, bn(xn), 'r', lw=3)
#     # ax.grid(True)
#     # plt.show()

#     ax = plt.figure().add_subplot(projection='3d')
#     for i in range(p.shape[0]):
#         ax.plot(p[i,:,0].numpy(), p[i,:,1].numpy(), p[i,:,2].numpy())
#     # i=1
#     # ax.plot(p[i,:,0].numpy(), p[i,:,1].numpy(), p[i,:,2].numpy())
#     ax.axis('equal')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.show()

# main()
import torch
from torchmin import minimize

# def rosen(x):
#     return torch.sum(100*(x[..., 1:] - x[..., :-1]**2)**2 
#                      + (1 - x[..., :-1])**2)

# # initial point
# x0 = torch.tensor([1., 8.])

# # Select from the following methods:
# #  ['bfgs', 'l-bfgs', 'cg', 'newton-cg', 'newton-exact', 
# #   'trust-ncg', 'trust-krylov', 'trust-exact', 'dogleg']

# # BFGS
# result = minimize(rosen, x0, method='bfgs')
# print(result)

# # Newton Conjugate Gradient
# result = minimize(rosen, x0, method='newton-cg')
# print(result)

# # Newton Exact
# result = minimize(rosen, x0, method='newton-exact')
# print(result)

# a = torch.tensor([0,3])
# b = torch.tensor([1,4])
# print(torch.gt(a[:,None],b[None,:]))
# print(torch.nonzero(torch.tensor([[0,0],[1,0]])))

# for i in range(1,8):
#     print(i)

# c = torch.tensor([0,1,2,3,4,5,6,7,8,9])
# print(c[5:8])

# m = torch.randn(6, 5, 4, 3)
# p = torch.randn(4)
# # print(p.expand_as(m).size())
# print(p.expand(3,4))

for i in range(11,-1,-1):
    print(i)
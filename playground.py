import torch

a = torch.tensor([0,3])
b = torch.tensor([1,4])
print(torch.gt(a[:,None],b[None,:]))
print(torch.nonzero(torch.tensor([[0,0],[1,0]])))

for i in range(1,8):
    print(i)

c = torch.tensor([0,1,2,3,4,5,6,7,8,9])
print(c[5:8])

m = torch.randn(6, 5, 4, 3)
p = torch.randn(4)
# print(p.expand_as(m).size())
print(p.expand(3,4))
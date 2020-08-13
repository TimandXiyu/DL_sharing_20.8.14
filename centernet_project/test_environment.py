#显示是否有CUDA环境
import 	torch
import  time
print(torch.__version__)
print(torch.cuda.is_available())

#新建a,b矩阵,内部数字随意,矩阵
b = torch.randn(1000, 2000)
a = torch.randn(10000, 1000)
#在CPU上面进行矩阵计算
t0 = time.time()
c = torch.matmul(a, b)
t1 = time.time()
#输出a.devic,以及计算一下t1-t0的时间
print(a.device, t1 - t0, c.norm(2))
#变成CUDA
device = torch.device('cuda')
#将a,b给cuda
a = a.to(device)
b = b.to(device)
#第一次计算一下二者时间
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))
#第二次计算一下二者时间(第一次时间会比第二次久,因为CUDA第一次运行需要一些时间
t0 = time.time()
c = torch.matmul(a, b)
t2 = time.time()
print(a.device, t2 - t0, c.norm(2))

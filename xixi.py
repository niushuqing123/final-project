import taichi as ti
import numpy as np
from functools import reduce

# from sph_base import SPHBase

# ti.init(arch=ti.cpu)
# Use GPU for higher peformance if available
ti.init(arch=ti.gpu, device_memory_GB=4, packed=True)


# 因为邻居搜索的网格不会做，所以尺寸数据只好也沿用助教的写法
# res = (720,720)
res = (512,512)
dim = 2
assert dim > 1
screen_to_world_ratio = 50
bound = np.array(res) / screen_to_world_ratio
print(bound)

# Material
material_boundary = 0


particle_radius = 0.05  # particle radius
particle_diameter = 2 * particle_radius
support_radius = particle_radius * 4.0  # support radius
m_V = 0.8 * particle_diameter ** dim
particle_max_num = 2 ** 15
particle_max_num_per_cell = 100
particle_max_num_neighbor = 200
particle_num = ti.field(int, shape=())


# gravity = -98.0  # 重力
viscosity = 0.05  # 黏性
density_0 = 1000.0  # 参照密度
mass = m_V * density_0

dt =2e-4


exponent = 7.0
stiffness = 50.0

# 粒子信息
x = ti.Vector.field(dim, dtype=float)
v = ti.Vector.field(dim, dtype=float)
d_velocity = ti.Vector.field(dim, dtype=float)
density = ti.field(dtype=float)
pressure = ti.field(dtype=float)
material = ti.field(dtype=int)
color = ti.field(dtype=int)


particle_neighbors = ti.field(int)
particle_neighbors_num = ti.field(int)


particles_node = ti.root.dense(ti.i, particle_max_num)
particles_node.place(x,v,d_velocity, density, pressure, material, color,particle_neighbors_num)

        



# Grid related properties
grid_size = support_radius
grid_num = np.ceil(np.array(res) / grid_size).astype(int)
print(grid_num)

grid_particles_num = ti.field(int)
grid_particles = ti.field(int)
padding = grid_size

particle_node = particles_node.dense(ti.j, particle_max_num_neighbor)
particle_node.place(particle_neighbors)
index = ti.ij if dim == 2 else ti.ijk
grid_node = ti.root.dense(index, grid_num)
grid_node.place(grid_particles_num)


cell_index = ti.k if dim == 2 else ti.l
cell_node = grid_node.dense(cell_index, particle_max_num_per_cell)
cell_node.place(grid_particles)


    # ========================================
    # 
    

# boundary particle



# 圆形，粒子信息
circular_max_num=1000
circular_num= ti.field(int, shape=())

circular_node = ti.root.dense(ti.i, circular_max_num)
c_x=ti.Vector.field(dim, dtype=float)
c_v=ti.Vector.field(dim, dtype=float)
c_f=ti.Vector.field(dim, dtype=float)
c_r=ti.field(float)
c_m=ti.field(float)

circular_node.place(c_x,c_v,c_f,c_r,c_m)


Young_modulus=2000000

# 弹簧数据结构
rest_length = ti.field(dtype=float, shape=(circular_max_num, circular_max_num))
fixed = ti.field(dtype=ti.i32, shape=circular_max_num)

Young_modulus_spring=921000
dashpot_damping=300#弹簧减震器


离墙距离=0.2#粒子边界不能距离实际边界太近，否则无效，可能是网格问题？


@ti.func
def cubic_kernel( r_norm):
    res = ti.cast(0.0, ti.f32)
    h = support_radius
    # value of cubic spline smoothing kernel
    k = 1.0
    if dim == 1:
        k = 4 / 3
    elif dim == 2:
        k = 40 / 7 / np.pi
    elif dim == 3:
        k = 8 / np.pi
    k /= h ** dim
    q = r_norm / h
    if q <= 1.0:
        if q <= 0.5:
            q2 = q * q
            q3 = q2 * q
            res = k * (6.0 * q3 - 6.0 * q2 + 1)
        else:
            res = k * 2 * ti.pow(1 - q, 3.0)
    # res是一个被强制转换为ti.f32的值
    return res

@ti.func
def cubic_kernel_derivative( r):
    h = support_radius
    # derivative of cubic spline smoothing kernel
    k = 1.0
    if dim == 1:
        k = 4 / 3
    elif dim == 2:
        k = 40 / 7 / np.pi
    elif dim == 3:
        k = 8 / np.pi
    k = 6. * k / h ** dim

    r_norm = r.norm()
    q = r_norm / h
    res = ti.Vector([0.0 for _ in range(dim)])
    if r_norm > 1e-5 and q <= 1.0:
        grad_q = r / (r_norm * h)
        if q <= 0.5:
            res = k * q * (3.0 * q - 2.0) * grad_q
        else:
            factor = 1.0 - q
            res = k * (-factor * factor) * grad_q
    return res

@ti.func
def viscosity_force( p_i, p_j, r):
    # Compute the viscosity force contribution
    v_xy = (v[p_i] -
            v[p_j]).dot(r)
    res = 2 * (dim + 2) * viscosity * (mass / (density[p_j])) * v_xy / (
        r.norm()**2 + 0.01 * support_radius**2) * cubic_kernel_derivative(r)
    return res

@ti.func
def pressure_force( p_i, p_j, r):
    # Compute the pressure force contribution, Symmetric Formula
    res = -density_0 * m_V * (pressure[p_i] / density[p_i] ** 2
            + pressure[p_j] / density[p_j] ** 2) \
            * cubic_kernel_derivative(r)
    return res


@ti.func
def simulate_collisions( p_i, vec, d):
    # Collision factor, assume roughly (1-c_f)*velocity loss after collision
    c_f = 0.3
    x[p_i] += vec * d
    v[p_i] -= (1.0 + c_f) * v[p_i].dot(vec) * vec


@ti.kernel
def solve():
    # 初始化网格，以搜索粒子的邻居
    # initialize_particle_system()
    for p in range(particle_num[None]):
        cell = (x[p] / grid_size).cast(int)
        offset = grid_particles_num[cell].atomic_add(1)
        grid_particles[cell, offset] = p
    
    #搜索邻居，不会打网格，借用助教的代码
    # search_neighbors()
    for i in range(particle_num[None]):

        #感觉这个没什么用？# Skip boundary particles
        # if material[i] == 0:
            # continue
        center_cell = (x[i] / grid_size).cast(int)
        cnt = 0
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * dim)):
            if cnt >= particle_max_num_neighbor:
                break
            cell = center_cell + offset

            flag = True
            for d in ti.static(range(dim)):
                flag = flag and (0 <= cell[d] < grid_num[d])

            if not flag:
                break

            for j in range(grid_particles_num[cell]):
                p_j = grid_particles[cell, j]
                distance = (x[i] - x[p_j]).norm()
                if i != p_j and distance < support_radius:
                    particle_neighbors[i, cnt] = p_j
                    cnt += 1
        particle_neighbors_num[i] = cnt

    #根据周围邻居,计算密度
    # compute_densities()
    for i in range(particle_num[None]):
        x_i = x[i]
        density[i] = 0.0#初始化密度
        for j in range(particle_neighbors_num[i]):
            p_j = particle_neighbors[i, j]
            x_j = x[p_j]
            密度权重=1#暂时解决了圆形周围的粒子会滑向边界
            if(material[i]==2):
                密度权重=6
            density[i] += 密度权重*m_V * cubic_kernel((x_i - x_j).norm())
        density[i] *= density_0


    边界压力权重=1
    边界粘性权重=1
    边界系数_exponent=exponent

    #根据密度,计算压力
    # compute_pressure_forces()            
    for i in range(particle_num[None]):#可以合并到上面的循环里面
        density[i] = ti.max(density[i], density_0)
        if(material[i]==2):
            边界系数_exponent=7
        pressure[i] = stiffness * (ti.pow(density[i] / density_0, 边界系数_exponent) - 1.0)
    

    # 重力、计算压力、计算粘性力
    # compute_non_pressure_forces()
    for i in range(particle_num[None]):
        if material[i] == 2:
            continue
        x_i = x[i]
        dv = ti.Vector([0 ,-280])#重力
        if(material[i]==3):
            dv[1]*=-1

        for j in range(particle_neighbors_num[i]):#根据邻居的数量，分别计算贡献
            p_j = particle_neighbors[i, j]
            if(material[p_j]==2):
                # 边界压力权重
                边界压力权重=3
                边界粘性权重=1
            x_j = x[p_j]
            #计算压力
            dv += 边界压力权重*pressure_force(i, p_j, x_i-x_j)
            # 计算粘性力
            dv += 边界粘性权重*viscosity_force(i, p_j, x_i - x_j)
        d_velocity[i] = dv



    #辛欧拉积分状态更新
    for i in range(particle_num[None]):
        if material[i] == 2:
            continue
        # if d_velocity[i].norm()>100:d_velocity[i]*=0.1#由于耦合的存在，经常产生高速粒子，对加速度做一下限制，不能太大，但可能影响性能，后期再测试

        v[i] += dt * d_velocity[i]
        if v[i].norm()>200:v[i]*=0.2#同上
        x[i] += dt * v[i]

    
    #仍然保留简陋的边界条件，用于限制坐标
    # 虽然粒子也可以当边界，但是高速粒子可以穿透，仍然需要控制一下
    # '''
    for i in range(particle_num[None]):
        if material[i] ==2:
            continue
        pos = x[i]
        #离墙距离2是为了让这个边界稍微比粒子边界宽一点点，以免粒子卡在粒子边界上,v+=1是还想给点往出推的速度，确保不卡住，但现在不需要了
        离墙距离2=离墙距离+0.01
        if pos[0] < 离墙距离2:
            # print("a")
            x[i][0]+=-1.2*(pos[0] - 离墙距离)
            # v[i][0]+=4
        if pos[0] > bound[0] - 离墙距离2:
            # print("y")
            x[i][0]-=-1.2*( bound[0] - 离墙距离-pos[0])
            # v[i][0]-=4
        if pos[1] > bound[1] - 离墙距离2:
            # print("s")
            x[i][1]-=-1.2*(bound[1] - 离墙距离-pos[1])
            # v[i][1]-=4
        if pos[1] < 离墙距离2:
            # print("x")
            x[i][1]+=-1.2*(pos[1] - 离墙距离)
            # v[i][1]+=4

    # '''
    
    
    # 成功将圆形碰撞耦合了进来！圆与粒子交互
    for i in range(circular_num[None]):
        pos1 = c_x[i]
        质量比例=particle_radius/c_r[i]#其实就是面积（体积）比例，半径比例
        for j in range(particle_num[None]):
            direction_vector=pos1-x[j]
            direction_vector_length=ti.sqrt(direction_vector[0]**2+direction_vector[1]**2)
            if (direction_vector_length<=c_r[i]+particle_radius):
                
                # if(material[j]==1):#只与流动的粒子作用，因为这个算法不太完善
                #制作一个切向加速度，近似摩擦力.n为垂直与单位向量的法向量，vrel是速度在法线上的投影
                n=ti.Vector([direction_vector[1],-direction_vector[0]])
                v_rel = (c_v[i] - v[j]).dot(n)
                v[j]-=v_rel*n*dt*7
                c_v[i]+=v_rel*n*dt*7

                if(material[j]==1):x[j]-=direction_vector*direction_vector_length*0.05#把粒子往出推一点，仅推一点，起到缓和冲击力就作用，这一操作会导致水里的物体一动，周围的粒子会跟着震动。。
                elastic_force=2000*(direction_vector/direction_vector_length)*(c_r[i]+particle_radius-direction_vector_length)
                
                v[j]-=elastic_force#由于杨氏模量数值的缘故，dt可以省去了
                c_v[i]+=elastic_force*质量比例*0.3#使粒子对圆的影响小一点

    

    # 圆形的相互碰撞部分、处理链接弹簧
    # for i in range(circular_num[None]):#最外层循环可以合并
        for j in range(i+1,circular_num[None]):
            direction_vector=c_x[j]-pos1   # direction_vector=c_x[j]-c_x[i]，pos1替换一下
            d = (direction_vector).normalized()  # 两个粒子的单位向量
            if rest_length[i, j] == 0:  # 是否存在弹簧
                direction_vector_length=ti.sqrt(direction_vector[0]**2+direction_vector[1]**2)
                if (direction_vector_length<=c_r[i] + c_r[j]):
                    elastic_force=Young_modulus*(direction_vector/direction_vector_length)*(c_r[i]+c_r[j]-direction_vector_length)
                    elastic_damping = (c_v[i] - c_v[j]).dot(direction_vector/direction_vector_length)  
                    c_v[i] += -elastic_damping*10 * (direction_vector/direction_vector_length)*dt
                    c_v[j] -= -elastic_damping*10 * (direction_vector/direction_vector_length)*dt
                    c_v[i]-=elastic_force*dt
                    c_v[j]+=elastic_force*dt
            else: 
                # 计算弹簧要用f，因为阻尼里面有v的影响，不能直接更新v
                # c_v[i] += Young_modulus_spring*(direction_vector.norm()/rest_length[j, i]-1)*d*dt
                c_f[i] += Young_modulus_spring*(direction_vector.norm()/rest_length[j, i]-1)*d
                # c_v[j] += -Young_modulus_spring*(direction_vector.norm()/rest_length[j, i]-1)*d*dt
                c_f[j] += -Young_modulus_spring*(direction_vector.norm()/rest_length[j, i]-1)*d

                v_rel = (c_v[j] - c_v[i]).dot(d)  
                # c_v[i] += v_rel*dashpot_damping* d*dt
                c_f[i] += v_rel*dashpot_damping* d
                # c_v[j] += -v_rel*dashpot_damping* d*dt
                c_f[j] += -v_rel*dashpot_damping* d


    # 半隐式欧拉更新圆形位置
    for i in range(circular_num[None]):
        #在这里做出区分是有必要的，因为粒子会影响圆的速度，而圆的状态更新公式里有受速度影响的项，所以要及时给速度和力置零，否则弹簧会出问题
        if fixed[i]==0:
            c_v[i]+=c_f[i]*dt
            c_v[i]*=0.995
            c_f[i]=[0,-2800]#用完重置力
            c_x[i] +=  c_v[i]*dt
        else:
            c_v[i]=[0,0]
            c_f[i]=[0,0]

    #圆形的碰撞边界可以去掉，但是要测试粒子做边界的效果
    #圆形的边界碰撞,挪用小作业中简化的过的公式,1000本来是杨氏模量，但是这份代码杨氏模量大，沿用1000可以省去*dt
    for i in range(circular_num[None]):
        if(c_x[i][0]<c_r[i]):c_v[i][0]+=(1000*(c_r[i]-c_x[i][0])-0.1*c_v[i][0])
        if(c_x[i][1]<c_r[i]):c_v[i][1]+=(1000*(c_r[i]-c_x[i][1])-0.1*c_v[i][0])
        if(c_x[i][0]+c_r[i]>bound[0]):c_v[i][0]+=(1000*(bound[0]-c_x[i][0]-c_r[i])-0.1*c_v[i][0])
        if(c_x[i][1]+c_r[i]>bound[1]):c_v[i][1]+=(1000*(bound[1]-c_x[i][1]-c_r[i])-0.1*c_v[i][1])
    

          
def substep():
    grid_particles_num.fill(0)
    particle_neighbors.fill(-1)
    solve()

@ti.kernel
def add_particle(posx:ti.f32,posy:ti.f32, vx:ti.f32,vy:ti.f32, material1:ti.i32, color1:ti.i32):
    # print(x1)
    color_=color1
    if(color1==0):
        if(material1==3):color_=0x00cc33
        if(material1==2):color_=0x696969
        if(material1==1):color_=0x87CEFA

    num =particle_num[None]

    x[num]= [posx,posy]
    v[num]= [vx,vy]
    density[num] = 1000

    material[num] = material1
    color[num] = color_
    particle_num[None] += 1


作为边界的的粒子=0
def build_boundary():
   #左
    # 离墙距离=0.15
    边界粒度=36
    边界数量=380
    for i in range(边界数量):
        add_particle(离墙距离,i/边界粒度,0,0,2,0)     
    #下
    for i in range(边界数量):
        add_particle(i/边界粒度,离墙距离,0,0,2,0) 
    #上
    for i in range(边界数量):
        add_particle(i/边界粒度,10.24-离墙距离,0,0,2,0)     
    #右
    for i in range(边界数量):
        add_particle(10.24-离墙距离,i/边界粒度,0,0,2,0) 

    global 作为边界的的粒子
    作为边界的的粒子=particle_num[None]
    print(作为边界的的粒子)

    # 接近边界会失效？网格影响？
    # 修改粒子密度权重，使用一层薄的粒子做边界
    # 全部用粒子做边界，可以不再需要圆形的边界判断

上一个粒子画的线=[0,0]
def p_bondary(pos1_,pos2_,dxdy):
    #两点确定斜率，用描点画线的方式近似画一个边界
    # dxdy粒度
    #换位的目的是，永远只考虑从低往高画
    if (pos1_[1] <=pos2_[1]):
        pos1=pos1_
        pos2=pos2_
    else:
        pos1=pos2_
        pos2=pos1_

    
    print(pos1)
    print(pos2)

    #两点坐标之差算斜率k
    k=(pos2[1]-pos1[1])/(pos2[0]-pos1[0])
    print("k:",k)

    dx=dy=dxdy#默认都为一倍的粒度

    if k<0:
        if(k>-1):
            dx*=-1
            dy=k*dxdy*-1#dy要为正数
        else:
            dx=(1/k)*dxdy
    else:
        if(k>=1):
            dx=(1/k)*dxdy
        else:
            dy=k*dxdy
    
    print("dx,dy:",dx,dy)
    
    posx=posy=0
    global 上一个粒子画的线
    上一个粒子画的线[0]=particle_num[None]
    if(k<0):
        while(1):
            add_particle((pos1[0]+posx),(pos1[1]+posy),0,0,2,0) 

            if(pos1[0]+posx>pos2[0]):posx+=dx#对于斜率的正负要做出区分
            if(pos1[1]+posy<pos2[1]):posy+=dy
            # if(pos1[1]+posy>pos2[1]):break
            if(pos1[0]+posx<pos2[0]):break
            print((posx,posy))
            # print(pos1[0]+posx)

            # add_particle((pos1[0]+posx),(pos1[1]+posy),0,0,2,0x956333) 
    else:
        while(1):
            add_particle((pos1[0]+posx),(pos1[1]+posy),0,0,2,0) 

            if(pos1[0]+posx<pos2[0]):posx+=dx#对于斜率的正负要做出区分
            if(pos1[1]+posy<pos2[1]):posy+=dy
            # if(pos1[1]+posy>pos2[1]):break
            if(pos1[0]+posx>pos2[0]):break
            print((pos1[0]+posx,pos1[1]+posy))

            # add_particle((pos1[0]+posx),(pos1[1]+posy),0,0,2,0x956333) 


    上一个粒子画的线[1]=particle_num[None]
    # print(上一个粒子画的线[1]-上一个粒子画的线[0])


def 边界粒子变流体(上一个粒子画的线):
    if 上一个粒子画的线[0]==0 and 上一个粒子画的线[1]==0 :
        for i in  range(作为边界的的粒子,particle_num[None]):
            if(material[i]==2):
                material[i] = 1
    else:
        for i in range(上一个粒子画的线[0],上一个粒子画的线[1]):
            material[i] = 1




def 范围边界变流体(pos_xx: ti.f32, pos_yy: ti.f32):
    搜索半径=0.2
    # print("aaaa")
    pos=ti.Vector([pos_xx,pos_yy])
    # print(pos)
    for i in range(particle_num[None]):
        # print(i)
        if material[i]== 2: 
            dpos=pos-x[i]
            # print(dpos)
            d = (dpos).norm()  # 粒子与鼠标的距离
            # print(d)
            if(d<=搜索半径):
                material[i]=1
                print("sss")



def add_particle_cube(pos,size,material,color_):
    li=(int)(size[0]*10)
    lj=(int)(size[1]*10)
    for i in range(li):
        for j in range(lj):
            pass
            add_particle(pos[0]+i/18,pos[1]+j/18,0,0,material,color_)


def 一个水枪(水枪位置,水枪速度,material):
    add_particle(水枪位置[0],水枪位置[1]+0.05, 水枪速度[0],水枪速度[1],material,0)
    add_particle(水枪位置[0],水枪位置[1]+0.1, 水枪速度[0],水枪速度[1],material,0)
    add_particle(水枪位置[0],水枪位置[1]+0.15, 水枪速度[0],水枪速度[1],material,0)
    add_particle(水枪位置[0],水枪位置[1]+0.2, 水枪速度[0],水枪速度[1],material,0)


@ti.kernel
def delete_particle(num1:ti.i32):
    
    if(particle_num[None]>作为边界的的粒子):
        num2 =particle_num[None]
        particle_num[None]-=num1
        for i in range(num2-num1,num2):
            x[i]=[0,0]
            v[i]=[0,0]
            d_velocity[i]=[0,0]
            pressure[i]= 0
            density[i] = 0
            particle_neighbors_num[i]=0
            material[i] = 0
            color[i] = 0
    pass


@ti.kernel
def copy_to_numpy_nd( np_arr: ti.ext_arr(), src_arr: ti.template(),num:ti.i32):
    for i in range(num):
        for j in ti.static(range(dim)):
            np_arr[i, j] = src_arr[i][j]


@ti.kernel
def copy_to_numpy_nd1( np_arr: ti.ext_arr(), src_arr: ti.template(),num:ti.i32):
    for i in range(num):
            np_arr[i] = src_arr[i]

@ti.kernel
def copy_to_numpy_for_radius(np_arr: ti.ext_arr(),num:ti.i32):
    for i in range(num):
        if(material[i]==2):
            np_arr[i]=9
        else:
            np_arr[i]=1




@ti.kernel
def copy_to_numpy_for_color(np_arr: ti.ext_arr(),num:ti.i32):
    for i in range(num):
        np_arr[i]=color[i]



@ti.kernel
def add_circular(pos_x: ti.f32, pos_y: ti.f32, r1: ti.f32,vx:ti.f32,vy:ti.f32,spring:ti.i32,fix:ti.i32,判定距离:ti.f32,链接长度:ti.f32):
    #起到默认值的效果
    判定距离_=判定距离
    链接长度_=链接长度
    if(判定距离==0):判定距离_=1.25
    if(链接长度==0):链接长度_=1
    
    num=circular_num[None]
    c_x[num] = ti.Vector([pos_x, pos_y])  # 将新粒子的位置存入x
    c_v[num]=ti.Vector([vx, vy])
    fixed[num]=fix

    c_r[num]=r1
    c_m[num]=r1*r1

    circular_num[None] += 1  # 粒子数量加一

    if(spring==1):
        for i in range(num):  # 遍历粒子库,判断新粒子与其他粒子的距离
            if(c_x[num]-c_x[i]).norm() < 判定距离_:  # 若小于0.15，在弹簧状态矩阵中添加两个粒子之间的弹簧
                rest_length[num, i] = 链接长度_  # 弹簧的静止长度
                rest_length[i, num] = 链接长度_


def add_circular_cube(x,y,size,r):
    li=(int)(size[0]/r)
    lj=(int)(size[1]/r)
    奇怪的参数=(1.6-r)
    #这个数最早是2，意思是两圆相隔两个半径的距离，小一点是为了让每两个圆重合一部分，总体看上去像矩形，但又不会被粒子穿透
    #但是，对于大半径的粒子来说可以让这个数较小，对于小半径的粒子来说，同样的数值会导致爆炸，似乎与杨氏模量有关，1.6暂时在一定范围内稳定
    for i in range (li):
        for j in range (lj):
            #3.8，这个数字要大于2倍根2小于4，可获得稳定矩形，小于则缺少中间的弹簧，大于4则弹簧链接过多
            # add_circular(x+i*2*r,y+j*2*r,r,0,0,1,0,3.8*r,(1.45-r)*r)
            add_circular(x+i*2*r,y+j*2*r,r,0,0,1,0,3.8*r,奇怪的参数*r)


def add_circular_cube_hollow(x,y,size,r):
    pass
    li=(int)(size[0]/r)
    lj=(int)(size[1]/r)
    奇怪的参数=(1.6-r)
    #这个数最早是2，意思是两圆相隔两个半径的距离，小一点是为了让每两个圆重合一部分，总体看上去像矩形，但又不会被粒子穿透
    #但是，对于大半径的粒子来说可以让这个数较小，对于小半径的粒子来说，同样的数值会导致爆炸，似乎与杨氏模量有关，1.6暂时在一定范围内稳定
    for i in range (li):
        for j in range (lj):
            #3.8，这个数字要大于2倍根2小于4，可获得稳定矩形，小于则缺少中间的弹簧，大于4则弹簧链接过多
            # add_circular(x+i*2*r,y+j*2*r,r,0,0,1,0,3.8*r,(1.45-r)*r)
            add_circular(x+i*2*r,y+j*2*r,r,0,0,1,0,3.8*r,奇怪的参数*r)


#做一个弹性绳,注意链接弹簧的技巧,
#由于又是画直线，参考p_bond
#单独操作弹簧矩阵的数据结构,避免不必要的粘连
def build_a_chain(pos1_,pos2_,dxdy,fixed_first,fixed_end,弹簧原长比例):
    #弹簧原长，在固定两端的情况下，也反映了硬度
    # r=dxdy/2
    r=dxdy/1.8#其实应该是2,稍微小一点，否则圆间隔太大

    弹簧长度=1.8

    if (pos1_[1] <=pos2_[1]):
        pos1=pos1_
        pos2=pos2_
    else:
        pos1=pos2_
        pos2=pos1_

     #两点坐标之差算斜率k
    k=(pos2[1]-pos1[1])/(pos2[0]-pos1[0])
    print("k:",k)

    dx=dy=dxdy#默认都为一倍的粒度

    if k<0:
        if(k>-1):
            dx*=-1
            dy=k*dxdy*-1#dy要为正数
        else:
            dx=(1/k)*dxdy
    else:
        if(k>=1):
            dx=(1/k)*dxdy
        else:
            dy=k*dxdy
     
    posx=posy=0
    if(k<0):
        while(1):
            add_circular((pos1[0]+posx),(pos1[1]+posy),r,0,0,0,fixed_first,2*r,弹簧原长比例*r)
            rest_length[circular_num[None],circular_num[None]-1]=弹簧原长比例*r
            rest_length[circular_num[None]-1,circular_num[None]]=弹簧原长比例*r
            fixed_first=0
            if(pos1[0]+posx>pos2[0]):posx+=dx#对于斜率的正负要做出区分
            if(pos1[1]+posy<pos2[1]):posy+=dy
            # if(pos1[1]+posy>pos2[1]):break
            if(pos1[0]+posx<pos2[0]):break
            # print((posx,posy))
            # print(pos1[0]+posx)
            # add_particle((pos1[0]+posx),(pos1[1]+posy),0,0,2,0x956333) 
    else:
        while(1):
            add_circular((pos1[0]+posx),(pos1[1]+posy),r,0,0,0,fixed_first,2*r,弹簧原长比例*r)
            rest_length[circular_num[None],circular_num[None]-1]=弹簧原长比例*r
            rest_length[circular_num[None]-1,circular_num[None]]=弹簧原长比例*r
            fixed_first=0
            if(pos1[0]+posx<pos2[0]):posx+=dx#对于斜率的正负要做出区分
            if(pos1[1]+posy<pos2[1]):posy+=dy
            # if(pos1[1]+posy>pos2[1]):break
            if(pos1[0]+posx>pos2[0]):break
            # print((pos1[0]+posx,pos1[1]+posy))
            # add_particle((pos1[0]+posx),(pos1[1]+posy),0,0,2,0x956333) 

    #要撤掉最后一个圆与下一个圆的弹簧
    rest_length[circular_num[None],circular_num[None]-1]=0
    rest_length[circular_num[None]-1,circular_num[None]]=0
    fixed[circular_num[None]-1]=fixed_end#最后一个点为固定点？



@ti.kernel
def attract(pos_x: ti.f32, pos_y: ti.f32):
    for i in range(circular_num[None]):
        if fixed[i] == 0:  # 如果粒子没有被定住则受外力影响
            v[i] += -dt*10*(x[i]-ti.Vector([pos_x, pos_y]))*100
            # 施加一个速度增量带代替力，因为受力计算在substep（）里并且每次都将其重置为仅有重力
            # 时间步*最小帧步长*位置差*常量系数。因为这函数调用在主函数substep循环之外，所以变化的幅度要乘以最小帧步长，以保证数据同比例变化。
            # 负号是因为，粒子位置减目标位置，向量（x1-x2）的方向是由x2指向x1，所以为了使粒子向着目标方向移动需要得到相反的方向

被选中的圆 = ti.field(int, shape=())
#可以实现删除被选择的圆，有两种手段，给圆的信息加一条是否生效，被删除的圆仍存在，但不生效。或者删除这个圆所有的信息，然后重排数组，稍微有点麻烦

@ti.kernel
def attract_one(pos_x: ti.f32, pos_y: ti.f32):
    if fixed[被选中的圆[None]] == 0:  # 如果粒子没有被定住则受外力影响
        # print("attract_one")
        c_v[被选中的圆[None]] += -dt*(c_x[被选中的圆[None]]-ti.Vector([pos_x, pos_y]))*200000

@ti.kernel
def search_circular(pos_x: ti.f32, pos_y: ti.f32):
    pos=ti.Vector([pos_x,pos_y])
    # print("search_circular")
    # print(pos)
    for i in range(circular_num[None]):
        #顶点也可以被选定
        # if fixed[i]== 1:  # 如果粒子没有被定住则受外力影响
            # continue
        # print(c_x[i])
        dpos=pos-c_x[i]
        # print(dpos)
        d = (dpos).norm()  # 两个粒子的距离
        # print(d)
        # print(c_r[i])
        if( d<=c_r[i]):
            被选中的圆[None]=i
            print("被选中的圆:",被选中的圆[None])


@ti.kernel
def switch_fixed():#切换被选中圆的固定状态
    if fixed[被选中的圆[None]] == 0:
        fixed[被选中的圆[None]]=1
    else:
        fixed[被选中的圆[None]]=0




旋转圆下标buff=[0,0]
def build_a_wheel(pos,size_L,c_r,内半径与sizeL比例):
    pos[0]
    pos[1]

    旋转圆下标buff[0]=circular_num[None]
    #二分之根号三等于0.866

    #手动构造弹簧，为了避免不必要的粘连
    #1
    add_circular(pos[0],pos[1],内半径与sizeL比例*size_L,0,0,0,1,size_L,size_L)
    #2
    add_circular(pos[0]+size_L,pos[1],c_r,0,0,0,0,size_L+0.01,size_L)
    rest_length[circular_num[None]-1,circular_num[None]-2]=size_L
    rest_length[circular_num[None]-2,circular_num[None]-1]=size_L

    #3
    add_circular(pos[0]+0.5*size_L,pos[1]+0.866*size_L,c_r,0,0,0,0,size_L+0.05,size_L)
    
    rest_length[circular_num[None]-1,circular_num[None]-2]=size_L
    rest_length[circular_num[None]-2,circular_num[None]-1]=size_L
    
    rest_length[circular_num[None]-1,旋转圆下标buff[0]]=size_L
    rest_length[旋转圆下标buff[0],circular_num[None]-1]=size_L

    #4
    add_circular(pos[0]-0.5*size_L,pos[1]+0.866*size_L,c_r,0,0,0,0,size_L+0.05,size_L)
    
    rest_length[circular_num[None]-1,circular_num[None]-2]=size_L
    rest_length[circular_num[None]-2,circular_num[None]-1]=size_L
    
    rest_length[circular_num[None]-1,旋转圆下标buff[0]]=size_L
    rest_length[旋转圆下标buff[0],circular_num[None]-1]=size_L

    #5
    add_circular(pos[0]-size_L,pos[1],c_r,0,0,0,0,size_L+0.01,size_L)
    rest_length[circular_num[None]-1,circular_num[None]-2]=size_L
    rest_length[circular_num[None]-2,circular_num[None]-1]=size_L

    rest_length[circular_num[None]-1,旋转圆下标buff[0]]=size_L
    rest_length[旋转圆下标buff[0],circular_num[None]-1]=size_L


    add_circular(pos[0]-0.5*size_L,pos[1]-0.866*size_L,c_r,0,0,0,0,size_L+0.05,size_L)
    rest_length[circular_num[None]-1,circular_num[None]-2]=size_L
    rest_length[circular_num[None]-2,circular_num[None]-1]=size_L
    
    rest_length[circular_num[None]-1,旋转圆下标buff[0]]=size_L
    rest_length[旋转圆下标buff[0],circular_num[None]-1]=size_L

    add_circular(pos[0]+0.5*size_L,pos[1]-0.866*size_L,c_r,0,0,0,0,size_L+0.05,size_L)
    rest_length[circular_num[None]-1,circular_num[None]-2]=size_L
    rest_length[circular_num[None]-2,circular_num[None]-1]=size_L
    
    rest_length[circular_num[None]-1,旋转圆下标buff[0]]=size_L
    rest_length[旋转圆下标buff[0],circular_num[None]-1]=size_L

    rest_length[circular_num[None]-1,旋转圆下标buff[0]+1]=size_L
    rest_length[旋转圆下标buff[0]+1,circular_num[None]-1]=size_L

    旋转圆下标buff[1]=circular_num[None]


@ti.kernel
def applied_rotating(num1:ti.i32,num2:ti.i32):
    #这是什么情况，为什么参数传进来了还必须写这两句话，不写报错，非得写num=num
    num1=num1
    num2=num2
    # print(num1)
    # print(num2)
    
    x1 = c_x[num1]
    num1 += 1
    for i in range(num1, num2):
        #本质上是求一个切线，法向量乘一个系数作为加速度
        c_v[i][0] += (c_x[i][1]-x1[1])*10
        c_v[i][1] += -(c_x[i][0]-x1[0])*10
        # 施加转速注意事项：方向与默认方向相同或相反，根据圆心点减去圆上点或者圆上点减圆心点决定
        # 统一自转方向

@ti.kernel
def reset_allcirculars():
    for i in range(circular_num[None]):
        c_x[i] =[0, 0]
        c_v[i]=[0, 0]
        c_f[i]=[0, 0]
        c_r[i]=0
        c_m[i]=0
        # c_m[i]=0
    # 弹簧清空
    for i in range(circular_num[None]):
        for j in range(circular_num[None]):
            rest_length[i, j] = 0

    circular_num[None]=0


def revocation_a_cirulars():
    circular_num[None] -= 1  
    num=circular_num[None]

    # 重置运动状态
    c_x[num] = ti.Vector([0, 0]) 
    c_v[num]=ti.Vector([0, 0])
    c_r[num]=0
    c_m[num]=0
    fixed[num]=0
    for i in range(num):
        # 卸掉弹簧
        rest_length[i, num] = 0
        rest_length[num, i] = 0
        # 重置运动状态
    

def demo1():

    build_a_wheel((3.6,2.6),0.6,0.1,1)

    add_particle_cube((6,1),(2,2),3,0)

    add_particle_cube((6,7),(3,1),1,0)

    add_particle_cube((1,8),(2,2),1,0)

    p_bondary((0.028*10.24,0.8*10.24),(0.3*10.24,0.4*10.24),0.02)

    build_a_chain((0.7*10.24,0.9*10.24),(0.9*10.24,0.9*10.24),0.1,1,1,1.8)

    add_circular_cube(0.4*10.24,0.08*10.24,(0.4,0.6),0.1)






#所有的可操作参数在这个函数里打印出来
def display_information():
    pass
    # 暂时不太需要了


def main():
    pause=1
    fream=0
    #液体矩形尺寸
    operate_p_cube_w=2
    operate_p_cube_h=2
    #弹性矩形尺寸
    operate_c_cube_w=1
    operate_c_cube_h=1
    #弹性圆半径
    operate_c_r=0.2
    #弹簧数据
    operate_spring_length=1.25#生效的弹簧长度
    operate_spring_detect=1#探测弹簧的距离
    #轮子尺寸，轮子上的圆半径
    wheel_sizeL=0.6
    wheel_c_r=0.1

    #链子的细度和硬度
    chain_粒度=0.4
    chain_弹簧比例=1.8

    #反重力液体
    是否反重力液体=1

    #记录两次坐标的缓存变量
    x_LMB_水枪1=[]
    x_LMB_水枪2=[]

    x_LMB_line=[]
    x_LMB_chain=[]


    水枪开关1=-1
    水枪位置1=(1,8)
    水枪速度1=(1,0)

    水枪开关2=-1
    水枪位置2=(1,8)
    水枪速度2=(1,0)

    旋转开关=-1
    清除粒子开关=-1

    build_boundary()
    demo1()

    # add_particle_cube((2,2),(10,3),1)

    # add_particle_cube((6,1),(2,2),3)

    # build_a_wheel((7,7),wheel_sizeL,wheel_c_r,1)

    gui = ti.GUI("xixi",res=((int)(res[0]*1.8),(int)(res[1]*1.8)),background_color=0xFFDEAD)


    fream=0
    while gui.running:
        for e in gui.get_events(ti.GUI.PRESS):  
            if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
                exit()
            elif e.key == ti.GUI.LMB:  # 鼠标左键，增加粒子
                # add_circular(e.pos[0]*10.24,e.pos[1]*10.24,operate_c_r,0,0,0,0,operate_spring_detect,operate_spring_length)  
                # add_particle(e.pos[0]*10.24,e.pos[1]*10.24,0,0,1,0x956333)
                # attract_lastone(e.pos[0]*10.24,e.pos[1]*10.24)
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    search_circular(e.pos[0]*10.24,e.pos[1]*10.24)
                    # 范围边界变流体(e.pos[0]*10.24,e.pos[1]*10.24)

                elif(gui.is_pressed(ti.GUI.CTRL)):
                    add_circular(e.pos[0]*10.24,e.pos[1]*10.24,operate_c_r,0,0,0,0,operate_spring_detect,operate_spring_length)  

            elif e.key == ti.GUI.RMB:  # 鼠标右键，增加粒子，带弹簧，如果按下ctrl，就是固定点
                if gui.is_pressed('1'):
                    x_LMB_水枪1.append(gui.get_cursor_pos())
                    print(x_LMB_水枪1)
                    if(len(x_LMB_水枪1) == 2):
                        #这里写的有点乱，n是为了算一个单位向量，/20是给n一个合适的值，起到水流最小速度的作用，
                        #+=120是为了让速度的大小，受一定的两点距离影响，公式可以化简，有点乱
                        n=np.linalg.norm([x_LMB_水枪1[1][0]-x_LMB_水枪1[0][0],x_LMB_水枪1[1][1]-x_LMB_水枪1[0][1]])/20
                        print(n)
                        水枪位置1=(x_LMB_水枪1[0][0]* res[0] /screen_to_world_ratio,x_LMB_水枪1[0][1]* res[1] /screen_to_world_ratio )
                        水枪速度1 = [x_LMB_水枪1[1][0]-x_LMB_水枪1[0][0],x_LMB_水枪1[1][1]-x_LMB_水枪1[0][1]]/n

                        水枪速度1[0] +=120*(x_LMB_水枪1[1][0]-x_LMB_水枪1[0][0])
                        水枪速度1[1] +=120*(x_LMB_水枪1[1][1]-x_LMB_水枪1[0][1])
                        
                        x_LMB_水枪1 = []
                elif gui.is_pressed('2'):
                    x_LMB_水枪2.append(gui.get_cursor_pos())
                    print(x_LMB_水枪2)
                    if(len(x_LMB_水枪2) == 2):
                        #这里写的有点乱，n是为了算一个单位向量，/20是给n一个合适的值，起到水流最小速度的作用，
                        #+=120是为了让速度的大小，受一定的两点距离影响，公式可以化简，有点乱
                        n=np.linalg.norm([x_LMB_水枪2[1][0]-x_LMB_水枪2[0][0],x_LMB_水枪2[1][1]-x_LMB_水枪2[0][1]])/20
                        print(n)
                        水枪位置2=(x_LMB_水枪2[0][0]* res[0] /screen_to_world_ratio,x_LMB_水枪2[0][1]* res[1] /screen_to_world_ratio )
                        水枪速度2 = [x_LMB_水枪2[1][0]-x_LMB_水枪2[0][0],x_LMB_水枪2[1][1]-x_LMB_水枪2[0][1]]/n

                        水枪速度2[0] +=120*(x_LMB_水枪2[1][0]-x_LMB_水枪2[0][0])
                        水枪速度2[1] +=120*(x_LMB_水枪2[1][1]-x_LMB_水枪2[0][1])
                        
                        x_LMB_水枪2 = []

                elif gui.is_pressed('l'):#粒子边界画线
                    x_LMB_line.append(gui.get_cursor_pos())
                    if(len(x_LMB_line) == 2):
                        p_bondary((x_LMB_line[0][0]*10.24,x_LMB_line[0][1]*10.24),
                        (x_LMB_line[1][0]*10.24,x_LMB_line[1][1]*10.24),0.02)
                        # print(x_LMB_line)
                        x_LMB_line = []
                elif gui.is_pressed('j'):#弹性圆画线
                    x_LMB_chain.append(gui.get_cursor_pos())
                    if(len(x_LMB_chain) == 2):
                        build_a_chain((x_LMB_chain[0][0]*10.24,x_LMB_chain[0][1]*10.24),
                        (x_LMB_chain[1][0]*10.24,x_LMB_chain[1][1]*10.24),chain_粒度,1,1,chain_弹簧比例)
                        # print(x_LMB_c_line)
                        x_LMB_chain = []
                elif gui.is_pressed('m'):
                    build_a_wheel((e.pos[0]*10.24,e.pos[1]*10.24),wheel_sizeL,wheel_c_r,1)
                else:
                    add_circular(e.pos[0]*10.24,e.pos[1]*10.24,operate_c_r,0,0,1,int(gui.is_pressed(ti.GUI.CTRL)),operate_spring_detect,operate_spring_length)

            #键盘按钮
            elif gui.is_pressed('1'):#水龙头开关
                if(gui.is_pressed(ti.GUI.CTRL)):水枪开关1*=-1

            elif gui.is_pressed('2'):#水龙头开关
                if(gui.is_pressed(ti.GUI.CTRL)):水枪开关2*=-1

            elif gui.is_pressed('q'):
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    if(operate_p_cube_w<6):
                        operate_p_cube_w+=0.1
                        print("液体矩形宽度修改为:",operate_p_cube_w)
                if(gui.is_pressed(ti.GUI.CTRL)):
                    if(operate_p_cube_w>0.5):
                        operate_p_cube_w-=0.1
                        print("液体矩形宽度修改为:",operate_p_cube_w)

            elif gui.is_pressed('a'):
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    if(operate_p_cube_h<6):
                        operate_p_cube_h+=0.1
                        print("液体矩形高度修改为:",operate_p_cube_h)
                if(gui.is_pressed(ti.GUI.CTRL)):
                    if(operate_p_cube_h>0.5):
                        operate_p_cube_h-=0.
                        print("液体矩形高度修改为:",operate_p_cube_h)

            elif gui.is_pressed('w'):
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    if(operate_c_cube_w<4):
                        operate_c_cube_w+=0.05
                        print("弹性矩形宽度修改为:",operate_c_cube_w)
                if(gui.is_pressed(ti.GUI.CTRL)):
                    if(operate_c_cube_w>0.2):
                        operate_c_cube_w-=0.05
                        print("弹性矩形宽度修改为:",operate_c_cube_w)

            elif gui.is_pressed('s'):
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    if(operate_c_cube_h<4):
                        operate_c_cube_h+=0.05
                        print("弹性矩形高度修改为:",operate_c_cube_h)
                if(gui.is_pressed(ti.GUI.CTRL)):
                    if(operate_c_cube_h>0.2):
                        operate_c_cube_h-=0.05
                        print("弹性矩形高度修改为:",operate_c_cube_h)

            elif gui.is_pressed('e'):
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    if(operate_spring_length<2):
                        operate_spring_length+=0.01
                        print("弹簧长度修改为:",operate_spring_length)
                if(gui.is_pressed(ti.GUI.CTRL)):
                    if(operate_spring_length>0.1):
                        operate_spring_length-=0.01
                        print("弹簧长度修改为:",operate_spring_length)

            elif gui.is_pressed('d'):
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    if(operate_spring_detect<2.2):
                        operate_spring_detect+=0.01
                        print("弹簧检测距离修改为:",operate_spring_detect)
                if(gui.is_pressed(ti.GUI.CTRL)):
                    if(operate_spring_detect>0.12):
                        operate_spring_detect-=0.01
                        print("弹簧检测距离修改为:",operate_spring_detect)


            elif gui.is_pressed('x'):
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    if(wheel_sizeL<2):
                        wheel_sizeL+=0.01
                        print("轮子半径修改为:",wheel_sizeL)
                if(gui.is_pressed(ti.GUI.CTRL)):
                    if(wheel_sizeL>0.1):
                        wheel_sizeL-=0.01
                        print("轮子半径修改为:",wheel_sizeL)

            elif gui.is_pressed('3'):
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    if(wheel_c_r<1.8):
                        wheel_c_r+=0.01
                        print("轮子上的圆半径修改为:",wheel_c_r)
                if(gui.is_pressed(ti.GUI.CTRL)):
                    if(wheel_c_r>0.16):
                        wheel_c_r-=0.01
                        print("轮子上的圆半径修改为:",wheel_c_r)

            elif gui.is_pressed('t'):
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    if(chain_粒度<1):
                        chain_粒度+=0.01
                        print("链子粒度修改为:",chain_粒度)
                if(gui.is_pressed(ti.GUI.CTRL)):
                    if(chain_粒度>0.06):
                        chain_粒度-=0.01
                        print("链子粒度修改为:",chain_粒度)

            elif gui.is_pressed('g'):
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    if(chain_弹簧比例<1.5):
                        chain_弹簧比例+=0.01
                        print("链子的弹簧半径比:",chain_弹簧比例)
                if(gui.is_pressed(ti.GUI.CTRL)):
                    if(chain_弹簧比例>0.4):
                        chain_弹簧比例-=0.01
                        print("链子的弹簧半径比:",chain_弹簧比例)
            
            
            elif gui.is_pressed('c'):
                if(gui.is_pressed(ti.GUI.SHIFT)):
                    if operate_c_r<1:
                        operate_c_r+=0.01
                        print("之后生成圆形半径为:",operate_c_r)
                if(gui.is_pressed(ti.GUI.CTRL)):
                    if operate_c_r>0.06:
                        operate_c_r-=0.01
                        print("之后生成圆形半径为:",operate_c_r)


            elif gui.is_pressed('r'):#清空所有圆
                reset_allcirculars()
                
            elif gui.is_pressed('f'):#设置圆的固定开关
                switch_fixed()

            elif gui.is_pressed('h'):#弄一个弹性矩形
                add_circular_cube(e.pos[0]*10,e.pos[1]*10,(operate_c_cube_w,operate_c_cube_h),operate_c_r)
                # add_circular_cube(e.pos[0]*10,e.pos[1]*10,(0.5,1),0.07)
                

            elif gui.is_pressed('g'):
                if(是否反重力液体==1):
                    是否反重力液体=3
                    print("生成的液体设置为:反重力!")
                else:
                    是否反重力液体=1
                    print("生成的液体设置为:正常重力!")

            elif gui.is_pressed('o'):
                add_particle_cube((e.pos[0]*10.24,e.pos[1]*10.24),(operate_p_cube_w,operate_p_cube_h),是否反重力液体,0)

            #上一个边界粒子线变液体，全部边界变液体
            elif gui.is_pressed('b'):
                if(gui.is_pressed(ti.GUI.CTRL)):
                    # print(上一个粒子画的线)
                    边界粒子变流体(上一个粒子画的线)
                else:
                    边界粒子变流体((0,0))                  

            elif gui.is_pressed('n'):
                旋转开关*=-1

            #撤销相关
            elif gui.is_pressed('z'):
                if(gui.is_pressed(ti.GUI.CTRL)):
                    # print(上一个粒子画的线)
                    delete_particle(上一个粒子画的线[1]-上一个粒子画的线[0])
                elif(gui.is_pressed(ti.GUI.SHIFT)):
                    清除粒子开关*=-1
                    # delete_particle(particle_num[None])#清除全部粒子
                else:
                    revocation_a_cirulars()

            elif gui.is_pressed('p'):
                pause*=-1



        # 撤销操作，分别对于圆与流体
        # 点击两点，确定一面墙
        # 旋转的水车，或者带动力的水车
        #（已完成）

        
        
        #增加删除粒子的功能，单个粒子，或所有流体，注意数据结构也要同步处理（完成）

        #撤销上一个粒子画线操作（完成）
        #增加粒子的函数可以改成kernel（完成）
        #制作反重力液体(已完成)


        #粒子与圆耦合的部分，添加摩擦力(粗糙的完成了= =)

        #圆的边界碰撞可以去除？(不可以)
        #制作弹性的链子(完成)
        #弹性圆的尺寸是否合理，弹簧长度是否合理
        #处理弹簧粘连问题（解决了链子的弹簧粘连，其他形状则比较复杂）
        #尝试空心cube（先不做了，主要是构造起来麻烦）
        #圆的边界碰撞可以去除？
        #制作一个作为滑块的圆形，左右移动，类似box2dliu中的滑块正方形（没必要，可以拖拽了）
        #从墙缝伸出一个机械臂，带一个轮子？（不用弄了，已经可以实现）

        
        #更多的交互功能，比如鼠标吸引流体或者圆（吸引圆，已完成）
        #指定圆（已完成）
        #指定圆的固定与解固定（已完成）
        #杨氏模量可交互调参,测试硬东西。。
        #多个水枪(已完成，目前两个,可拓展)
        #对于撤销操作，可否使用堆栈实现多步撤销（太麻烦）

        #作为边界的粒子，显示半径也可以大一点,颜色也可以调整一下


        #合理的调色
        #不同颜色的流体，看看混合效果（反重力液体实现了）
        #颜色切换，提前准备一个颜色数组
        #合理的参数命名
        #制作小场景demo

        #是否启用粒子边界做一个宏开关，应为粒子边界效果一般还有点卡


        #存在一个很不美观的问题，刚生成的矩形粒子一碰就炸，应该是跟边界权重有关（已修复)

        #圆一碰边界，边界粒子变液体
        


        #ggui？？
        #流体表面重建？？？？

        if pause==-1:
            fream+=1
            if gui.is_pressed(ti.GUI.LMB):
                c = gui.get_cursor_pos()
                attract_one(c[0]*10.24,c[1]*10.24)

            if(fream%2==0):
                if 水枪开关1==1:
                    一个水枪(水枪位置1, 水枪速度1,是否反重力液体)
                    # my_add_particle((3,2), (2,0),1,0x956333)#螺旋喷射= =
                if 水枪开关2==1:
                    一个水枪(水枪位置2, 水枪速度2,是否反重力液体)
                    # my_add_particle((3,2), (2,0),1,0x956333)#螺旋喷射= =
            if 旋转开关==1:
                applied_rotating(旋转圆下标buff[0],旋转圆下标buff[1])

            if 清除粒子开关==1:
                delete_particle(5)
                
            for i in range(4):
                substep()
        

        p_num=particle_num[None]
        c_num=circular_num[None]
        np_x = np.ndarray((p_num, dim), dtype=np.float32)
        copy_to_numpy_nd(np_x, x,p_num)
        np_color = np.ndarray(p_num, dtype=np.int32)
        copy_to_numpy_for_color(np_color,p_num)

        r = np.ndarray(p_num, dtype=np.float32)#为了让边界粒子有更大半径，是一组比例系数
        copy_to_numpy_for_radius(r,p_num)


        X = np.ndarray((c_num, dim), dtype=np.float32)
        copy_to_numpy_nd(X, c_x,c_num)
        R = np.ndarray(c_num, dtype=np.float32)
        for i in range(c_num): R[i]=c_r[i]



        # print(particle_info['position'])
        # gui.circles(np_x ,radius=particle_radius / 1.5 * screen_to_world_ratio,color=0x956333)
        # gui.text("test",[0.5,0.5])#gui显示文本有问题、、、
        gui.circles(np_x * screen_to_world_ratio / res,radius=r*particle_radius / 1.5 * screen_to_world_ratio,color=np_color)
        gui.circles(X * screen_to_world_ratio / res,radius=R* screen_to_world_ratio*1.75,color=0x5534993)
        # gui.circles(X * screen_to_world_ratio / 512,radius=256 ,color=0x5534993)

        #画线是严重帧率的事情，能不能把要画的线放到数据结构里，用kernel算好再画
        #圆比较多的时候建议注释掉画线这部分''''''
        '''
        for i in range(c_num):
            for j in range(i+1, c_num):  # 原本是for j in range(n),实际上只需从上一层循环的i开始即可，可以省去一半的遍历量和绘图量
                if rest_length[i, j] != 0:
                    gui.line(begin=X[i] * screen_to_world_ratio / res, end=X[j] * screen_to_world_ratio / res,
                             color=0X888888, radius=2)  # 画线
        '''
        gui.show()



        
if __name__ == "__main__":
    main()
    



'''
#为了提速，尝试将画线改为kernel函数，但是失败了，以后再修吧
@ti.kernel
def p_bondary(pos1_x:ti.f32,pos1_y:ti.f32,pos2_x:ti.f32,pos2_y:ti.f32,dxdy:ti.f32):
    #两点确定斜率，用描点画线的方式近似画一个边界
    # dxdy粒度
    #换位的目的是，永远只考虑从低往高画
    pos1x=0
    pos1y=0
        
    pos2x=0
    pos2y=0
    if (pos1_y <=pos2_y):
        pos1x=pos1_x
        pos1y=pos1_y
        
        pos2x=pos2_x
        pos2y=pos2_y
    else:
        pos1x=pos2_x
        pos1y=pos2_y

        pos2x=pos1_x
        pos2y=pos1_y

    #两点坐标之差算斜率k
    print(pos2y)
    print(pos1y)
    a=pos2y-pos1y
    b=pos2x-pos1x
    k=a/b

    dx=dy=dxdy#默认都为一倍的粒度

    if k<0:
        if(k>-1):
            dx*=-1
            dy=k*dxdy*-1#dy要为正数
        else:
            dx=(1/k)*dxdy
    else:
        if(k>=1):
            dx=(1/k)*dxdy
        else:
            dy=k*dxdy

    # print(dx)
    # print(dy)
    
    posx=posy=0
    #原本调用了addparticles,但是改成kernel之后，因为add是kernel，所以没法用了，只能写开
    # add_particle((pos1x+posx),(pos1y+posy),0,0,2,0x956333) 

    if(k<0):
        while(1):
            if(pos1x+posx>pos2x):posx+=dx#对于斜率的正负要做出区分
            if(pos1y+posy<pos2y):posy+=dy
            if(pos1x+posy>=pos2y):break
            # print((posx,posy))
            # print(x1)
            
            num =particle_num[None]
            print(num)

            # x[num][0] = x1[0]*10
            # x[num][1] = x1[1]*10

            x[num]= [posx,posy]
            print(x[num])

            v[num]= [0,0]
            density[num] = 1000

            material[num] = 2
            color[num] = 0x956333
            particle_num[None] += 1


            # add_particle_forline(((pos1x+posx),(pos1y+posy)),(0,0),2,0x956333) 


    else:
        while(1):
            if(pos1x+posx<pos2x):posx+=dx#对于斜率的正负要做出区分
            if(pos1y+posy<pos2y):posy+=dy
            if(pos1y+posy>=pos2y):break
            # print((posx,posy))
            num =particle_num[None]
            print(num)

            # x[num][0] = x1[0]*1
            # x[num][1] = x1[1]*10

            x[num]= [posx,posy]
            print(x[num])
            v[num]= [0,0]
            density[num] = 1000

            material[num] = 2
            color[num] = 0x956333
            particle_num[None] += 1
            # add_particle_forline(((pos1x+posx),(pos1y+posy)),(0,0),2,0x956333) 
'''
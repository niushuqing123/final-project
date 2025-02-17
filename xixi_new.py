import os
import taichi as ti
import numpy as np
from functools import reduce


#2025/2/17,更新了最新版太极接口，交互窗口替换为ggui


# ti.init(arch=ti.gpu, kernel_profiler=True)
# ti.init(arch=ti.gpu)

ti.init(arch=ti.gpu, kernel_profiler=True, debug=False)

# Global variables (to replace class attributes)
res = (512, 512) # Screen resolution, for example
dim = len(res)
screen_to_world_ratio = 50
bound = np.array(res) / screen_to_world_ratio



# Global variables for the particle system (from the first part)
g = -9.80  # Gravity重力
dt = ti.field(float, shape=())
# dt[None] = 3e-4
dt[None] = 0.0015
dt[None] =3e-4




# Material types
material_fluid = 1
material_boundary = 2
material_antigravity_fluid = 3

# Particle properties
particle_radius = 0.05
particle_diameter = 2 * particle_radius
support_radius = particle_radius * 4.0
m_V = 0.8 * particle_diameter ** dim
particle_max_num = 2 ** 15
particle_max_num_per_cell = 100
particle_max_num_neighbor = 100
particle_num = ti.field(int, shape=())


viscosity = 0.05  # Viscosity黏性
density_0 = 1000.0  # Reference density密度
mass = m_V * density_0 # Mass of the particles, to be initialized
exponent = 7.0
stiffness = 50.0


# Particle-related properties
x = ti.Vector.field(dim, dtype=float)
v = ti.Vector.field(dim, dtype=float)
d_velocity = ti.Vector.field(dim, dtype=float)  # 存储每个粒子的速度变化,dv实际上是加速度
density = ti.field(dtype=float)
pressure = ti.field(dtype=float)
material = ti.field(dtype=int)
color = ti.Vector.field(3, dtype=float)
particle_neighbors = ti.field(dtype=int)
particle_neighbors_num = ti.field(dtype=int)
particle_screen_pos = ti.Vector.field(dim, dtype=float)  # 新增屏幕坐标字段

particle_radius_for_screen= ti.field(dtype=float)

# Particle nodes
particles_node = ti.root.dense(ti.i, particle_max_num)
particles_node.place(x, v, density, pressure,d_velocity, material, color, particle_screen_pos,particle_radius_for_screen)  
particles_node.place(particle_neighbors_num)


# Neighbor nodes，particle_node是particles_node中每一个粒子的元素，树状结构
particle_node = particles_node.dense(ti.j, particle_max_num_neighbor)
particle_node.place(particle_neighbors)


# Grid-related properties
grid_size = support_radius
grid_num = np.ceil(np.array(res) / grid_size).astype(int)
grid_particles_num = ti.field(int)
grid_particles = ti.field(int)
padding = grid_size

# Grid nodes
index = ti.ij if dim == 2 else ti.ijk
grid_node = ti.root.dense(index, grid_num)
grid_node.place(grid_particles_num)

cell_index = ti.k if dim == 2 else ti.l
cell_node = grid_node.dense(cell_index, particle_max_num_per_cell)
cell_node.place(grid_particles)





# 圆形，粒子信息
circular_max_num=1000
circular_num= ti.field(int, shape=())

circular_node = ti.root.dense(ti.i, circular_max_num)
c_x=ti.Vector.field(dim, dtype=float)
c_v=ti.Vector.field(dim, dtype=float)
c_f=ti.Vector.field(dim, dtype=float)
c_r=ti.field(float)
c_m=ti.field(float)
circular_screen_pos = ti.Vector.field(dim, dtype=float)  # 新增屏幕坐标字段
circular_screen_radius = ti.field(float)  



fixed = ti.field(int)

circular_node.place(c_x,c_v,c_f,c_r,c_m,fixed,circular_screen_pos,circular_screen_radius)


Young_modulus=2000000

# 弹簧数据结构
rest_length = ti.field(dtype=float, shape=(circular_max_num, circular_max_num))

Young_modulus_spring=921000
dashpot_damping=300#弹簧减震器


离墙距离=0.2#粒子边界不能距离实际边界太近，否则无效，可能是网格问题？



COLOR_TABLE = {
    "water": (0.34, 0.58, 0.94),  # 水蓝色
    "sand": (0.76, 0.70, 0.50),   # 沙黄色
    "stone": (0.50, 0.50, 0.50),  # 石灰色
    "grass": (0.13, 0.55, 0.13),  # 草绿色
    "Emerald_green": (0.0, 0.8, 0.2),# 宝石绿色(反重力流体颜色)
    "wood": (0.55, 0.27, 0.07),   # 木棕色
    "red": (1.0, 0.0, 0.0),       # 红色
    "green": (0.0, 1.0, 0.0),     # 绿色
    "blue": (0.0, 0.0, 1.0),      # 蓝色
    "white": (1.0, 1.0, 1.0),     # 白色
    "black": (0.0, 0.0, 0.0),     # 黑色
    "circular_color": (1.0, 0.2863,0.5765 ),  # 新增颜色 0x553499
    "background_color" : (1.0, 0.8706, 0.6784),
    "bondary_particle_color" : (0.411765, 0.411765, 0.411765),
    "sky_blue":(0.529, 0.808, 0.980),#原版流体颜色
}






#================流体计算================#
@ti.func
def pos_to_index(pos):
    return (pos / grid_size).cast(int)

@ti.func
def is_valid_cell(cell):
    flag = True
    for d in ti.static(range(dim)):
        flag = flag and (0 <= cell[d] < grid_num[d])
    return flag

@ti.kernel
def allocate_particles_to_grid():
    # 初始化网格，以搜索粒子的邻居
    for p in range(particle_num[None]):
        cell = pos_to_index(x[p])
        offset = ti.atomic_add(grid_particles_num[cell], 1)
        grid_particles[cell, offset] = p

@ti.kernel
def search_neighbors():
    #搜索邻居
    for p_i in range(particle_num[None]):
        #边界粒子也不能跳过邻居搜索
        # if material[p_i] == material_boundary:
        #     continue
        center_cell = pos_to_index(x[p_i])
        cnt = 0
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * dim)):
            if cnt >= particle_max_num_neighbor:
                break
            cell = center_cell + offset
            if not is_valid_cell(cell):
                break
            for j in range(grid_particles_num[cell]):
                p_j = grid_particles[cell, j]
                distance = (x[p_i] - x[p_j]).norm()
                if p_i != p_j and distance < support_radius:
                    particle_neighbors[p_i, cnt] = p_j
                    cnt += 1
        particle_neighbors_num[p_i] = cnt



@ti.func
def cubic_kernel(r_norm):
    res = ti.cast(0.0, ti.f32)
    h = support_radius
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
    return res

@ti.func
def cubic_kernel_derivative(r):
    h = support_radius
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
def viscosity_force(p_i, p_j, r):
    v_xy = (v[p_i] - v[p_j]).dot(r)
    res = 2 * (dim + 2) * viscosity * (mass / (density[p_j])) * v_xy / (
        r.norm()**2 + 0.01 * support_radius**2) * cubic_kernel_derivative(r)
    return res

@ti.func
def pressure_force(p_i, p_j, r):
    res = -density_0 * m_V * (pressure[p_i] / density[p_i] ** 2
          + pressure[p_j] / density[p_j] ** 2) * cubic_kernel_derivative(r)
    return res


@ti.kernel
def compute_densities():
    #根据周围邻居,计算密度
    for p_i in range(particle_num[None]):
        x_i = x[p_i]
        density[p_i] = 0.0#初始化密度
        for j in range(particle_neighbors_num[p_i]):
            p_j = particle_neighbors[p_i, j]
            x_j = x[p_j]
            密度权重=1#暂时解决了圆形周围的粒子会滑向边界
            if(material[p_i]==material_boundary):
                密度权重=6
            density[p_i] += 密度权重*m_V * cubic_kernel((x_i - x_j).norm())
        density[p_i] *= density_0



#合并compute_non_pressure_forces与compute_pressure_forces
@ti.kernel
def compute_pressure_and_nonpressure_forces():
    #计算压力与非压力的力。
    压力权重=1
    边界粘性权重=1
    # 边界系数_exponent=exponent
    for p_i in range(particle_num[None]):#可以合并到上面的循环里面(密度循环)
        # 确保粒子的密度不低于参考密度
        density[p_i] = ti.max(density[p_i], density_0)
        # 根据状态方程计算粒子的压力
        # if(material[p_i]==2):
            # 边界系数_exponent=7
        # pressure[p_i] = stiffness * (ti.pow(density[p_i] / density_0, 边界系数_exponent) - 1.0)
        pressure[p_i] = stiffness * (ti.pow(density[p_i] / density_0, exponent) - 1.0)
    
    #根据密度、邻居,计算受力
    for p_i in range(particle_num[None]):
        if material[p_i] != material_boundary:
            x_i = x[p_i]
            d_v = ti.Vector([0 ,-580])#重力，直接在初始化中添加重力加速度
            if(material[p_i]==material_antigravity_fluid):
                d_v[1]*=-1
            #根据邻居的数量，分别计算贡献
            for j in range(particle_neighbors_num[p_i]):
                p_j = particle_neighbors[p_i, j]
                x_j = x[p_j]
                if(material[p_j]==material_boundary):
                    # 边界压力权重
                    压力权重=4
                    # 边界粘性权重=2
                #计算压力
                d_v += 压力权重 * pressure_force(p_i, p_j, x_i - x_j)
                # 计算粘性力
                # d_v += 边界粘性权重*viscosity_force(p_i, p_j, x_i - x_j)
                d_v += viscosity_force(p_i, p_j, x_i - x_j)

            d_velocity[p_i] = d_v



@ti.kernel
def updata_particle():
    # Symplectic Euler
    for p_i in range(particle_num[None]):
        if material[p_i] != material_boundary:
            v[p_i] += dt[None] * d_velocity[p_i]
            if v[p_i].norm()>200:v[p_i]*=0.2#速度限制
            x[p_i] += dt[None] * v[p_i]
        # d_velocity[p_i]=0#每一轮的加速度没有被重置，但不影响？其实compute_non_pressure_forces里面，d_velocity[p_i] = d_v时已经重置了


@ti.func
def simulate_collisions(p_i, vec, d):
    c_f = 0.3  # Collision factor
    x[p_i] += vec * d
    v[p_i] -= (1.0 + c_f) * v[p_i].dot(vec) * vec

@ti.kernel
def enforce_boundary():
    for p_i in range(particle_num[None]):
        if dim == 2:
            if material[p_i] != material_boundary:
                pos = x[p_i]
                if pos[0] < padding:
                    simulate_collisions(p_i, ti.Vector([1.0, 0.0]),
                                        padding - pos[0])
                if pos[0] > bound[0] - padding:
                    simulate_collisions(p_i, ti.Vector([-1.0, 0.0]),
                                        pos[0] - (bound[0] - padding))
                if pos[1] > bound[1] - padding:
                    simulate_collisions(p_i, ti.Vector([0.0, -1.0]),
                                        pos[1] - (bound[1] - padding))
                if pos[1] < padding:
                    simulate_collisions(p_i, ti.Vector([0.0, 1.0]),
                                        padding - pos[1])

@ti.kernel
def enforce_boundary0():
  
    #仍然保留简陋的边界条件，用于限制坐标
    # 虽然粒子也可以当边界，但是高速粒子可以穿透，仍然需要控制一下
    
    for i in range(particle_num[None]):
        if material[i] ==material_boundary:
            continue
        pos = x[i]
        #离墙距离2是为了让这个边界稍微比粒子边界宽一点点，以免粒子卡在粒子边界上,v+=1是还想给点往出推的速度，确保不卡住，但现在不需要了
        离墙距离2=离墙距离+0.01
        if pos[0] < 离墙距离2:
            # print("a")
            x[i][0]+=-1.2*(pos[0] - 离墙距离2)
            # v[i][0]+=4
        if pos[0] > bound[0] - 离墙距离2:
            # print("y")
            x[i][0]-=-1.2*( bound[0] - 离墙距离2-pos[0])
            # v[i][0]-=4
        if pos[1] > bound[1] - 离墙距离2:
            # print("s")
            x[i][1]-=-1.2*(bound[1] - 离墙距离2-pos[1])
            # v[i][1]-=4
        if pos[1] < 离墙距离2:
            # print("x")
            x[i][1]+=-1.2*(pos[1] - 离墙距离2)
            # v[i][1]+=4

@ti.kernel
def Coupling_circular_spring():
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
                v[j]-=v_rel*n*dt[None]*1
                c_v[i]+=v_rel*n*dt[None]*1
                

                # if(material[j]==material_fluid):x[j]-=direction_vector*direction_vector_length*0.05#把粒子往出推一点，仅推一点，起到缓和冲击力就作用，这一操作会导致水里的物体一动，周围的粒子会跟着震动。。
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
                    c_v[i] += -elastic_damping*10 * (direction_vector/direction_vector_length)*dt[None]
                    c_v[j] -= -elastic_damping*10 * (direction_vector/direction_vector_length)*dt[None]
                    c_v[i] -= elastic_force*dt[None]
                    c_v[j] += elastic_force*dt[None]
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

@ti.kernel
def updata_circular():
    # 半隐式欧拉更新圆形位置
    for i in range(circular_num[None]):
        #在这里做出区分是有必要的，因为粒子会影响圆的速度，而圆的状态更新公式里有受速度影响的项，所以要及时给速度和力置零，否则弹簧会出问题
        if fixed[i]==0:
            c_v[i]+=c_f[i]*dt[None]
            c_v[i]*=0.995
            c_f[i]=[0,-2800]#用完重置力
            c_x[i] +=  c_v[i]*dt[None]
        else:
            c_v[i]=[0,0]
            c_f[i]=[0,0]

@ti.kernel
def circular_boundary():
    #圆形的碰撞边界可以去掉，但是要测试粒子做边界的效果
    #圆形的边界碰撞,挪用小作业中简化的过的公式,1000本来是杨氏模量，但是这份代码杨氏模量大，沿用1000可以省去*dt
    for i in range(circular_num[None]):
        if(c_x[i][0]<c_r[i]):c_v[i][0]+=(1000*(c_r[i]-c_x[i][0])-0.1*c_v[i][0])
        if(c_x[i][1]<c_r[i]):c_v[i][1]+=(1000*(c_r[i]-c_x[i][1])-0.1*c_v[i][0])
        if(c_x[i][0]+c_r[i]>bound[0]):c_v[i][0]+=(1000*(bound[0]-c_x[i][0]-c_r[i])-0.1*c_v[i][0])
        if(c_x[i][1]+c_r[i]>bound[1]):c_v[i][1]+=(1000*(bound[1]-c_x[i][1]-c_r[i])-0.1*c_v[i][1])
    



def solve_step():
    grid_particles_num.fill(0)
    particle_neighbors.fill(-1)
    allocate_particles_to_grid()
    search_neighbors()
    compute_densities()
    compute_pressure_and_nonpressure_forces()
    updata_particle()
    enforce_boundary()

    Coupling_circular_spring()
    updata_circular()
    circular_boundary()



#-----------------添加粒子-------------#
@ti.kernel
def add_particle(posx: ti.f32, posy: ti.f32, vx: ti.f32, vy: ti.f32, material1: ti.i32, color_rgb: ti.types.vector(3, float)): # type: ignore
    num = particle_num[None]
    x[num] = [posx, posy]
    v[num] = [vx, vy]
    density[num] = 1000
    material[num] = material1
    color[num] = color_rgb  # 直接存储RGB向量
    particle_num[None] += 1




#添加一个粒子构成的矩形
def add_particle_cube(pos, size, material, color_key):
    # 从颜色表中获取RGB值
    if color_key in COLOR_TABLE:
        color_rgb = COLOR_TABLE[color_key]
    else:
        raise ValueError(f"颜色键 '{color_key}' 不在颜色表中，请检查COLOR_TABLE。")
    
    # 计算粒子数量
    li = int(size[0] * 10)
    lj = int(size[1] * 10)
    
    # 添加粒子
    for i in range(li):
        for j in range(lj):
            add_particle(pos[0] + i / 18, pos[1] + j / 18, 0, 0, material, color_rgb)





作为边界的的粒子=0
def build_boundary():
    #左
    # 离墙距离=0.15
    边界粒度=36
    边界数量=380
    #粒度与数量可以优化减少，具体可以在计算受力和密度时添加较大权重
    for i in range(边界数量):
        add_particle(离墙距离,i/边界粒度,0,0,2,COLOR_TABLE["bondary_particle_color"])     
    #下
    for i in range(边界数量):
        add_particle(i/边界粒度,离墙距离,0,0,2,COLOR_TABLE["bondary_particle_color"]) 
    #上
    for i in range(边界数量):
        add_particle(i/边界粒度,res[0] /screen_to_world_ratio-离墙距离,0,0,2,COLOR_TABLE["bondary_particle_color"])     
    #右
    for i in range(边界数量):
        add_particle(res[0] /screen_to_world_ratio-离墙距离,i/边界粒度,0,0,2,COLOR_TABLE["bondary_particle_color"]) 

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
            add_particle((pos1[0]+posx),(pos1[1]+posy),0,0,material_boundary,COLOR_TABLE["bondary_particle_color"]) 

            if(pos1[0]+posx>pos2[0]):posx+=dx#对于斜率的正负要做出区分
            if(pos1[1]+posy<pos2[1]):posy+=dy
            # if(pos1[1]+posy>pos2[1]):break
            if(pos1[0]+posx<pos2[0]):break
            print((posx,posy))
            # print(pos1[0]+posx)

            # add_particle((pos1[0]+posx),(pos1[1]+posy),0,0,2,0x956333) 
    else:
        while(1):
            add_particle((pos1[0]+posx),(pos1[1]+posy),0,0,material_boundary,COLOR_TABLE["bondary_particle_color"]) 

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
            if(material[i]==material_boundary):
                material[i] = material_fluid
                color[i]=COLOR_TABLE["sky_blue"]
    else:
        for i in range(上一个粒子画的线[0],上一个粒子画的线[1]):
            material[i] = material_fluid
            color[i]=COLOR_TABLE["sky_blue"]



# @ti.kernel
def 范围边界变流体(pos_xx: ti.f32, pos_yy: ti.f32,搜索半径: ti.f32): # type: ignore
    # print("aaaa")
    pos=ti.Vector([pos_xx,pos_yy])
    # print(pos)
    for i in range(particle_num[None]):
        # print(i)
        if material[i]== material_boundary: 
            dpos=pos-x[i]
            # print(dpos)
            d = (dpos).norm()  # 粒子与鼠标的距离
            # print(d)
            if(d<=搜索半径):
                material[i]=material_fluid
                color[i]=COLOR_TABLE["sky_blue"]

                # color[i]=0x87CEFA
                print("sss")





def 一个水枪(水枪位置,水枪速度,material,color_key):
    add_particle(水枪位置[0],水枪位置[1], 水枪速度[0],水枪速度[1],material,COLOR_TABLE[color_key])
    add_particle(水枪位置[0]+0.05,水枪位置[1]+0.05, 水枪速度[0],水枪速度[1],material,COLOR_TABLE[color_key])
    # add_particle(水枪位置[0]+0.1,水枪位置[1]+0.1, 水枪速度[0],水枪速度[1],material,COLOR_TABLE[color_key])
    # add_particle(水枪位置[0]+0.15,水枪位置[1]+0.15, 水枪速度[0],水枪速度[1],material,COLOR_TABLE[color_key])


@ti.kernel
def delete_particle(num1:ti.i32): # type: ignore
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
            color[i] = [0,0,0]
            particle_screen_pos[i]=[0,0]
            particle_radius_for_screen[i]=0


    pass





#--------------------添加圆------------------#

@ti.kernel
def add_circular(pos_x: ti.f32, pos_y: ti.f32, r1: ti.f32,vx:ti.f32,vy:ti.f32,spring:ti.i32,fix:ti.i32,判定距离:ti.f32,链接长度:ti.f32): # type: ignore
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
    print("c_r[num]=r1:",c_r[num])

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






#用鼠标对圆形产生吸引力
@ti.kernel
def attract(pos_x: ti.f32, pos_y: ti.f32): # type: ignore
    for i in range(circular_num[None]):
        if fixed[i] == 0:  # 如果粒子没有被定住则受外力影响
            v[i] += -dt[None]*10*(x[i]-ti.Vector([pos_x, pos_y]))*100
            # 施加一个速度增量带代替力，因为受力计算在substep（）里并且每次都将其重置为仅有重力
            # 时间步*最小帧步长*位置差*常量系数。因为这函数调用在主函数substep循环之外，所以变化的幅度要乘以最小帧步长，以保证数据同比例变化。
            # 负号是因为，粒子位置减目标位置，向量（x1-x2）的方向是由x2指向x1，所以为了使粒子向着目标方向移动需要得到相反的方向

被选中的圆 = ti.field(int, shape=())
#可以实现删除被选择的圆，有两种手段，给圆的信息加一条是否生效，被删除的圆仍存在，但不生效。或者删除这个圆所有的信息，然后重排数组，稍微有点麻烦

@ti.kernel
def attract_one(pos_x: ti.f32, pos_y: ti.f32): # type: ignore
    if fixed[被选中的圆[None]] == 0:  # 如果粒子没有被定住则受外力影响
        # print("attract_one")
        c_v[被选中的圆[None]] += -dt[None]*(c_x[被选中的圆[None]]-ti.Vector([pos_x, pos_y]))*200000


#在范围内搜索圆形
@ti.kernel
def search_circular(pos_x: ti.f32, pos_y: ti.f32): # type: ignore
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



#旧版变量名：旋转圆下标buff
上一组构成轮子的圆下标buff=[0,0]
def build_a_wheel(pos,size_L,c_r,内半径与sizeL比例):
    pos[0]
    pos[1]

    上一组构成轮子的圆下标buff[0]=circular_num[None]
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
    
    rest_length[circular_num[None]-1,上一组构成轮子的圆下标buff[0]]=size_L
    rest_length[上一组构成轮子的圆下标buff[0],circular_num[None]-1]=size_L

    #4
    add_circular(pos[0]-0.5*size_L,pos[1]+0.866*size_L,c_r,0,0,0,0,size_L+0.05,size_L)
    
    rest_length[circular_num[None]-1,circular_num[None]-2]=size_L
    rest_length[circular_num[None]-2,circular_num[None]-1]=size_L
    
    rest_length[circular_num[None]-1,上一组构成轮子的圆下标buff[0]]=size_L
    rest_length[上一组构成轮子的圆下标buff[0],circular_num[None]-1]=size_L

    #5
    add_circular(pos[0]-size_L,pos[1],c_r,0,0,0,0,size_L+0.01,size_L)
    rest_length[circular_num[None]-1,circular_num[None]-2]=size_L
    rest_length[circular_num[None]-2,circular_num[None]-1]=size_L

    rest_length[circular_num[None]-1,上一组构成轮子的圆下标buff[0]]=size_L
    rest_length[上一组构成轮子的圆下标buff[0],circular_num[None]-1]=size_L


    add_circular(pos[0]-0.5*size_L,pos[1]-0.866*size_L,c_r,0,0,0,0,size_L+0.05,size_L)
    rest_length[circular_num[None]-1,circular_num[None]-2]=size_L
    rest_length[circular_num[None]-2,circular_num[None]-1]=size_L
    
    rest_length[circular_num[None]-1,上一组构成轮子的圆下标buff[0]]=size_L
    rest_length[上一组构成轮子的圆下标buff[0],circular_num[None]-1]=size_L

    add_circular(pos[0]+0.5*size_L,pos[1]-0.866*size_L,c_r,0,0,0,0,size_L+0.05,size_L)
    rest_length[circular_num[None]-1,circular_num[None]-2]=size_L
    rest_length[circular_num[None]-2,circular_num[None]-1]=size_L
    
    rest_length[circular_num[None]-1,上一组构成轮子的圆下标buff[0]]=size_L
    rest_length[上一组构成轮子的圆下标buff[0],circular_num[None]-1]=size_L

    rest_length[circular_num[None]-1,上一组构成轮子的圆下标buff[0]+1]=size_L
    rest_length[上一组构成轮子的圆下标buff[0]+1,circular_num[None]-1]=size_L

    上一组构成轮子的圆下标buff[1]=circular_num[None]


@ti.kernel
def applied_rotating(num_c1:ti.i32,num_c2:ti.i32): # type: ignore
    #这是什么情况，为什么参数传进来了还必须写这两句话，不写报错，非得写num=num
    num1=num_c1
    num2=num_c2
    # print(num1)
    # print(num2)
    
    x1 = c_x[num1]
    num1 += 1
    for i in range(num1, num2):
        #本质上是求一个切线，法向量乘一个系数作为加速度

        c_v[i][0] = (c_x[i][1]-x1[1])*80#速度恒定
        c_v[i][1] = -(c_x[i][0]-x1[0])*80    
        
        # c_v[i][0] += (c_x[i][1]-x1[1])*80#加速度恒定
        # c_v[i][1] += -(c_x[i][0]-x1[0])*80
        # 施加转速注意事项：方向与默认方向相同或相反，根据圆心点减去圆上点或者圆上点减圆心点决定
        # 统一自转方向


@ti.kernel
def increase_rotation(num_c1:ti.i32,num_c2:ti.i32): # type: ignore
    #这是什么情况，为什么参数传进来了还必须写这两句话，不写报错，非得写num=num
    num1=num_c1
    num2=num_c2
    # print(num1)
    # print(num2)
    
    x1 = c_x[num1]
    num1 += 1
    for i in range(num1, num2):
        #本质上是求一个切线，法向量乘一个系数作为加速度

        c_v[i][0] += (c_x[i][1]-x1[1])*20#速度恒定
        c_v[i][1] += -(c_x[i][0]-x1[0])*20    
        
        # c_v[i][0] += (c_x[i][1]-x1[1])*80#加速度恒定
        # c_v[i][1] += -(c_x[i][0]-x1[0])*80
        # 施加转速注意事项：方向与默认方向相同或相反，根据圆心点减去圆上点或者圆上点减圆心点决定
        # 统一自转方向

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
def reset_allcirculars():
    for i in range(circular_num[None]):
        c_x[i] =[0, 0]
        c_v[i]=[0, 0]
        c_f[i]=[0, 0]
        c_r[i]=0
        c_m[i]=0
        # c_m[i]=0

        circular_screen_pos[i]=ti.Vector([0, 0]) 
        circular_screen_radius[i]=0

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

    circular_screen_pos[num]=ti.Vector([0, 0]) 
    circular_screen_radius[num]=0



indices = ti.field(dtype=ti.i32, shape=circular_max_num * circular_max_num * 2 )


#渲染前预处理
#计算坐标系到屏幕坐标系转换
@ti.kernel
def compute_screen_pos():
    for i in range(particle_num[None]):
        if(material[i]==material_boundary):
            particle_radius_for_screen[i]=0.01
        if(material[i]==material_fluid):
            particle_radius_for_screen[i]=0.00325
        if(material[i]==material_antigravity_fluid):
            particle_radius_for_screen[i]=0.00325
        for d in ti.static(range(dim)):
            particle_screen_pos[i][d] = x[i][d] * screen_to_world_ratio / res[d]

    for i in range(circular_num[None]):
        for d in ti.static(range(dim)):
            circular_screen_pos[i][d] = c_x[i][d] * screen_to_world_ratio / res[d]
        
        circular_screen_radius[i]=c_r[i]*screen_to_world_ratio/res[0]*1
        #原版*1.75系数，因为原版1.75倍显示半径才能在显示时接近实际物理作用半径
        # 可能因为新ui接口偏差（其实大概率是原版gui接口有误差，新ui在经过计算到屏幕坐标缩放之后仍然能保持较好的一致性）


    for i in indices:
        indices[i] = circular_max_num - 1
    n = circular_num[None]
    for i in range(n):
        for j in range(i + 1, n):
            line_id = i * circular_max_num + j
            if rest_length[i, j] != 0:
                indices[line_id * 2] = i
                indices[line_id * 2 + 1] = j



def demo1():
    # add_particle_cube((6, 1), (5, 2), 1, "grass")  # 使用草绿色
    add_particle_cube((6, 2), (4, 1), 1, "water")  # 使用水蓝色
    # add_particle_cube((1, 8), (2, 2), 1, "sand")   # 使用沙黄色

    p_bondary((0.028*res[0] /screen_to_world_ratio,0.8*res[1] /screen_to_world_ratio),(0.3*res[0] /screen_to_world_ratio,0.4*res[1] /screen_to_world_ratio),0.02)
    p_bondary((0.31*res[0] /screen_to_world_ratio,0.41*res[1] /screen_to_world_ratio),(0.3*res[0] /screen_to_world_ratio,0.028*res[1] /screen_to_world_ratio),0.02)


#从维护状态数组过度到使用简易状态机模式
class StateMachine:
    # 状态机类，统一管理按钮状态和子窗口状态
    def __init__(self):
        # 子窗口状态标记
        self.subwindow_circular_single = -1  #单个圆形窗口
        self.subwindow_circular_cube = -1  # 圆形构成的结构
        self.subwindow_circular_wheel = -1  # 圆形构成的结构
        self.subwindow_circular_chain = -1  # 圆形构成的结构

        self.subwindow_fluid_operate = -1  #流体参数操作
        self.subwindow_fluid_cube = -1  #流体矩形
        self.subwindow_particle_boundary = -1  #粒子边界
        self.subwindow_fluid_watergun = -1  #水枪
        self.subwindow_control = -1  #总控制



        # 鼠标状态标记
        self.mouse_circular_single = 1  # 添加圆形物体
        self.mouse_circular_select = 2  # 选择圆形物体
        self.mouse_circular_attract = 2345 # 吸引圆形物体
        self.mouse_circular_cube = 3345  # 添加圆形构成的矩形
        self.mouse_circular_wheel= 445 # 添加圆形构成的轮子
        self.mouse_circular_chain= 4548 # 添加圆形构成的链条

        self.mouse_fluid_cube = 156  # 添加流体矩形
        self.mouse_particle_boundary = 956  # 添加粒子边界
        self.mouse_fluid_watergun1= 1826  # 添加水枪
        self.mouse_fluid_watergun2= 1926  # 添加水枪
        self.mouse_boundary_to_fuild = 53456  # 粒子边界变流体

        self.mouse_state = -1  # 鼠标状态
        
        
        self.particle_current_color_state = "water" #粒子颜色状态



    def set_mouse_state(self, state):
        # 设置鼠标状态
        self.mouse_state = state

    def get_current_subwindow_state(self):
        # 返回当前子窗口状态
        return self.subwindow_state

    def get_current_mouse_state(self):
        # 返回当前鼠标状态
        return self.mouse_state


def main():
    # pause=-1
    pause=False
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
    operate_spring_length=1#生效的弹簧长度
    operate_spring_detect=1.25#探测弹簧的距离
    #轮子尺寸，轮子上的圆半径
    wheel_sizeL=0.6
    wheel_c_r=0.15

    is_spring=False

    #链子的细度和硬度
    chain_粒度=0.2
    chain_弹簧比例=1.8

    显示弹簧=False



    物质种类=1

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
    清除粒子开关=False

    擦除的搜索半径=0.2

    
    # 初始化窗口和画布
    window = ti.ui.Window("SPH", res=((int)(res[0]*1.8),(int)(res[1]*1.8)))
    canvas = window.get_canvas()
    gui = window.get_gui()

    canvas.set_background_color(COLOR_TABLE["background_color"])

    # 创建状态机对象
    state_machine = StateMachine()

    build_boundary()
    demo1()


    # radius = (particle_radius / 1.5) * screen_to_world_ratio / res[0]
    print(particle_radius)
    # print(radius)
    print("当前粒子数：",particle_num[None])


    # 在循环中调用 canvas.circles
    while window.running:
        # ti.profiler.print_kernel_profiler_info()

        #在这里可以实现按住持续生效
        # if window.is_pressed(ti.ui.RMB):
        #     if state_machine.get_current_mouse_state() == state_machine.mouse_circular_attract: # 吸引圆形
        #         attract_one(pos[0]*res[0] /screen_to_world_ratio,pos[1]*res[1] /screen_to_world_ratio)
   



        for e in window.get_events(ti.ui.PRESS):
            print(state_machine.get_current_mouse_state())

            pos = window.get_cursor_pos()
            if e.key in [ti.ui.ESCAPE]:
                exit()
            # elif e.key == ti.ui.SPACE:
            #     pause*=-1
            # elif e.key == ti.ui.LMB:
            #     pos = window.get_cursor_pos()
            #     print(pos)
            #     add_circular(pos[0]*res[0] /screen_to_world_ratio,pos[1]*res[1] /screen_to_world_ratio,operate_c_r,0,0,0,0,operate_spring_detect,operate_spring_length)  
            elif e.key == ti.ui.RMB:
            # if window.is_pressed(ti.ui.RMB):
                print(state_machine.get_current_mouse_state())
                if state_machine.get_current_mouse_state() == state_machine.mouse_circular_single: # 添加单个圆形
                    pos = window.get_cursor_pos()
                    print(pos)
                    add_circular(pos[0]*res[0] /screen_to_world_ratio,pos[1]*res[1] /screen_to_world_ratio,operate_c_r,0,0,np.square(is_spring),0,operate_spring_detect,operate_spring_length)  
                elif state_machine.get_current_mouse_state() == state_machine.mouse_circular_select: # 选择圆形
                    search_circular(pos[0]*res[0] /screen_to_world_ratio,pos[1]*res[1] /screen_to_world_ratio)
                elif state_machine.get_current_mouse_state() == state_machine.mouse_circular_cube: # 添加圆形构成的矩形结构
                    add_circular_cube(pos[0]*res[0] /screen_to_world_ratio,pos[1]*res[1] /screen_to_world_ratio,(operate_c_cube_w,operate_c_cube_h),operate_c_r)
                elif state_machine.get_current_mouse_state() == state_machine.mouse_circular_attract: # 吸引圆形
                    attract_one(pos[0]*res[0] /screen_to_world_ratio,pos[1]*res[1] /screen_to_world_ratio)
                elif state_machine.get_current_mouse_state() == state_machine.mouse_circular_wheel: # 吸引圆形
                    build_a_wheel((pos[0]*res[0] /screen_to_world_ratio,pos[1]*res[1] /screen_to_world_ratio),wheel_sizeL,wheel_c_r,1)
                elif state_machine.get_current_mouse_state() == state_machine.mouse_circular_chain: #弹性圆画线
                    x_LMB_chain.append(pos)
                    if(len(x_LMB_chain) == 2):
                        build_a_chain((x_LMB_chain[0][0]*res[0] /screen_to_world_ratio,x_LMB_chain[0][1]*res[1] /screen_to_world_ratio),
                        (x_LMB_chain[1][0]*res[0] /screen_to_world_ratio,x_LMB_chain[1][1]*res[1] /screen_to_world_ratio),chain_粒度,1,1,chain_弹簧比例)
                        # print(x_LMB_c_line)
                        x_LMB_chain = []
                elif state_machine.get_current_mouse_state() == state_machine.mouse_fluid_cube: # 添加流体矩形
                    add_particle_cube((pos[0]*res[0] /screen_to_world_ratio,pos[1]*res[1] /screen_to_world_ratio),(operate_p_cube_w,operate_p_cube_h),物质种类,state_machine.particle_current_color_state)
                elif state_machine.get_current_mouse_state() == state_machine.mouse_particle_boundary: # 添加粒子边界
                    x_LMB_line.append(pos)
                    if(len(x_LMB_line) == 2):
                        p_bondary((x_LMB_line[0][0]*res[0] /screen_to_world_ratio,x_LMB_line[0][1]*res[1] /screen_to_world_ratio),
                        (x_LMB_line[1][0]*res[0] /screen_to_world_ratio,x_LMB_line[1][1]*res[1] /screen_to_world_ratio),0.02)
                        # print(x_LMB_line)
                        x_LMB_line = []
                elif state_machine.get_current_mouse_state() == state_machine.mouse_boundary_to_fuild: # 粒子边界变流体
                    范围边界变流体(pos[0]*res[0] /screen_to_world_ratio,pos[1]*res[1] /screen_to_world_ratio,擦除的搜索半径)


                elif state_machine.get_current_mouse_state() ==  state_machine.mouse_fluid_watergun1: # 添加水枪
                    x_LMB_水枪1.append(pos)
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
                elif state_machine.get_current_mouse_state() ==  state_machine.mouse_fluid_watergun2: # 添加水枪
                    x_LMB_水枪2.append(pos)
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
                    
            elif e.key == "c":
                pass


        with gui.sub_window("Control Panel", 0.7, 0.1, 0.25, 0.4) as w:
            w.text("control")
            # 显示粒子数和圆形数
            w.text(f"Current particle count: {particle_num[None]}")
            w.text(f"Current circle count: {circular_num[None]}")  
            pause = w.checkbox("paused", old_value=pause)
            if w.button("circular_single"):
                state_machine.subwindow_circular_single*=-1
            if w.button("circular_cube"):
                state_machine.subwindow_circular_cube*=-1
            if w.button("circular_wheel"):
                state_machine.subwindow_circular_wheel*=-1
            if w.button("circular_Chain"):
                state_machine.subwindow_circular_chain*=-1
            if w.button("fluid_operate"):
                state_machine.subwindow_fluid_operate*=-1        
            if w.button("fluid_cube"):
                state_machine.subwindow_fluid_cube*=-1
            if w.button("fluid_boundary"):
                state_machine.subwindow_particle_boundary*=-1
            if w.button("fluid_watergun"):
                state_machine.subwindow_fluid_watergun*=-1
            if w.button("control"):
                state_machine.subwindow_control*=-1
        
        # 根据按钮状态显示子窗口
        if 1 == state_machine.subwindow_circular_single:  # "circular" 子窗口
            with gui.sub_window("circular_operate", 0.05, 0.05, 0.25, 0.4) as w:
                w.text("Circular Operations")
                w.text(f"Current circle count: {circular_num[None]}")  

                if w.button("single circular"):
                    state_machine.set_mouse_state(state_machine.mouse_circular_single)  # 设置鼠标状态为添加单个圆形
                operate_c_r = w.slider_float("next_c_r", operate_c_r, 0.05, 2)
                
                is_spring = w.checkbox("is_spring", old_value=is_spring)
                显示弹簧 = w.checkbox("display_sping", old_value=显示弹簧)


                operate_spring_length = w.slider_float("next_spring_length", operate_spring_length, 0.2, 2)
                operate_spring_detect = w.slider_float("next_spring_detect", operate_spring_detect, 0.22, 2.4)
                
                
                if w.button("select circular"):
                    state_machine.set_mouse_state(state_machine.mouse_circular_select)  # 设置鼠标状态为选择圆形
                w.text(f"The currently selected circle:{被选中的圆[None]}")
                
                if w.button("attract circular"):
                    state_machine.set_mouse_state(state_machine.mouse_circular_attract)  # 设置鼠标状态为吸引圆形
                if w.button("switch_fixed select circular"):
                    switch_fixed()
                if w.button("revocation_a_cirulars"):
                    revocation_a_cirulars()


        if state_machine.subwindow_circular_cube == 1:  #圆形构成的矩形
            with gui.sub_window("circular_cube", 0.1, 0.1, 0.25, 0.4) as w:
                w.text("subwindow_circular_cube")
                if w.button("circular_cube"):
                    state_machine.set_mouse_state(state_machine.mouse_circular_cube)  # 设置鼠标状态为添加圆形构成的矩形结构
                    print(state_machine.get_current_mouse_state())

                operate_c_cube_w = w.slider_float("next_c_cube_w", minimum=0.05, maximum=2.0, old_value=operate_c_cube_w)
                operate_c_cube_h = w.slider_float("next_c_cube_h", minimum=0.05, maximum=2.0, old_value=operate_c_cube_h)

        if state_machine.subwindow_circular_wheel == 1:  # 圆形构成的轮子
            with gui.sub_window("circular_wheel", 0.15, 0.15, 0.25, 0.4) as w:
                if w.button("circular_wheel"):
                    state_machine.set_mouse_state(state_machine.mouse_circular_wheel)  # 设置鼠标状态为添加圆形构成的矩形结构
                wheel_sizeL = w.slider_float("wheel_sizeL", wheel_sizeL, 0.2, 2)
                wheel_c_r = w.slider_float("wheel_c_r", wheel_c_r, 0.05, 0.6)
                if w.button("applied_rotating"):
                        旋转开关*=-1
                if w.button("increase_rotation"):
                        increase_rotation(上一组构成轮子的圆下标buff[0],上一组构成轮子的圆下标buff[1])

        if state_machine.subwindow_circular_chain == 1:  # 圆形构成的链子
            with gui.sub_window("circular_chain", 0.2, 0.2, 0.25, 0.4) as w:
                if w.button("circular_chain"):
                    state_machine.set_mouse_state(state_machine.mouse_circular_chain)  #

                chain_粒度 = w.slider_float("chain_Granularity", chain_粒度, 0.05, 0.6)
                chain_弹簧比例 = w.slider_float("chain_Spring_ratio", chain_弹簧比例, 0.05, 2)

        if state_machine.subwindow_fluid_operate == 1: 
            with gui.sub_window("fluid_operate", 0.25, 0.25, 0.25, 0.4) as w:
                w.text(f"Current particle count: {particle_num[None]}")
                w.text(f"Current fluid color: {state_machine.particle_current_color_state}")
                w.text(f"Current material type: {物质种类}")

                if w.button("particle_type_water"):
                    物质种类=material_fluid
                    state_machine.particle_current_color_state="water"

                if w.button("particle_type_boundary"):
                    物质种类=material_boundary
                    state_machine.particle_current_color_state="bondary_particle_color"

                if w.button("particle_type_antigravity_water"):
                    物质种类=material_antigravity_fluid
                    state_machine.particle_current_color_state="Emerald_green"


                if w.button("fuluid_color_water"):
                    state_machine.particle_current_color_state="water"
                if w.button("fuluid_color_Emerald_green"):
                    state_machine.particle_current_color_state="Emerald_green"
                if w.button("fuluid_color_sand"):
                    state_machine.particle_current_color_state="sand"
                if w.button("fuluid_color_white"):
                    state_machine.particle_current_color_state="white"
                if w.button("fuluid_color_sky_blue"):
                    state_machine.particle_current_color_state="sky_blue"




        if state_machine.subwindow_fluid_cube == 1: # 流体矩形
            with gui.sub_window("fluid_cube", 0.3, 0.3, 0.25, 0.4) as w:
                if w.button("fluid_cube"):
                    state_machine.set_mouse_state(state_machine.mouse_fluid_cube)  # 设置鼠标状态为添加圆形构成的矩形结构
                
                operate_p_cube_w = w.slider_float("next_p_cube_w", operate_p_cube_w, 0.05, 4)
                operate_p_cube_h = w.slider_float("next_p_cube_h", operate_p_cube_h, 0.05, 4)

        if state_machine.subwindow_particle_boundary == 1:# 粒子边界
            with gui.sub_window("particle_boundary", 0.35, 0.35, 0.25, 0.4) as w:
                if w.button("particle_boundary"):
                    state_machine.set_mouse_state(state_machine.mouse_particle_boundary)  # 设置鼠标状态为添加圆形构成的矩形结构
                if w.button("particle_boundary_to_fluid"):
                    边界粒子变流体(上一个粒子画的线)
                if w.button("particle_boundary_to_fluid_scope"):
                    state_machine.set_mouse_state(state_machine.mouse_boundary_to_fuild)  # 设置鼠标状态为添加圆形构成的矩形结构
                擦除的搜索半径 = w.slider_float("b2f_Detection_scope", 擦除的搜索半径, 0.05, 0.6)
                if w.button("delete_last_boundary"):
                    delete_particle(上一个粒子画的线[1]-上一个粒子画的线[0])


        if state_machine.subwindow_fluid_watergun == 1:# 水枪
            with gui.sub_window("fluid_watergun", 0.4, 0.4, 0.25, 0.4) as w:
                if w.button("switch_watergun1"):
                    水枪开关1*=-1
                if w.button("fluid_watergun1"):
                    state_machine.set_mouse_state(state_machine.mouse_fluid_watergun1)  # 设置鼠标状态为添加圆形构成的矩形结构
                if w.button("switch_watergun2"):
                    水枪开关2*=-1
                if w.button("fluid_watergun2"):
                    state_machine.set_mouse_state(state_machine.mouse_fluid_watergun2)  # 设置鼠标状态为添加圆形构成的矩形结构


        if state_machine.subwindow_control == 1:# 总控制
            with gui.sub_window("control", 0.45, 0.45, 0.25, 0.4) as w:
                if w.button("reset_allcirculars"):
                    reset_allcirculars()

                清除粒子开关 = w.checkbox("delete_particle_switch", old_value=清除粒子开关)



        if 旋转开关==1:
            applied_rotating(上一组构成轮子的圆下标buff[0],上一组构成轮子的圆下标buff[1])

        if(fream%4==0):
            if 水枪开关1==1:
                一个水枪(水枪位置1, 水枪速度1,material_fluid,state_machine.particle_current_color_state)      
            if 水枪开关2==1:
                一个水枪(水枪位置2, 水枪速度2,material_fluid,state_machine.particle_current_color_state)

        if 清除粒子开关==1:
            delete_particle(5)
        #更新物理、绘制窗口
        if pause==False:
            for _ in range(3):
                solve_step()
        
        compute_screen_pos()  # 调用坐标转换核函数



        canvas.circles(centers=particle_screen_pos,radius=1,per_vertex_radius=particle_radius_for_screen,per_vertex_color=color)
        canvas.circles(centers=circular_screen_pos,radius=1,per_vertex_radius=circular_screen_radius,color=COLOR_TABLE["circular_color"])

        if 显示弹簧:
            canvas.lines(circular_screen_pos, indices=indices, color=COLOR_TABLE["stone"], width=0.004)

        
        window.show()
main()
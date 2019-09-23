from pysit import *
from pysit.gallery import marmousi
import scipy.io as io
import os
import matplotlib.pyplot as plt
import time
import numpy as np
if __name__ == '__main__':
    # 生成默认的二维marmousi模型
    # 物理尺寸：横坐标-0~9200；纵坐标-0~3000；单位-m
    # 网格参数：横向网格数-460；纵向网格数-150；
    # 网格划分间隔-20；单位-m；横向网格点数-461；纵向网格点数-151
    # C-二维Marmousi模型各网格波速；C0-反演初始模型各网格波速；
    # m-网格对象，保存二维Marmousi模型的网格剖分信息；
    # d-物理模型对象，保存二维Marmousi模型的物理信息
    # initial_model_style-初始模型生成规则，可填参数constant、smooth_width、smooth_low_pass、gradient
    C, C0, m, d = marmousi(patch='mini_square',initial_model_style='smooth_low_pass')
    # 绘制模型
    plt.figure()
    plt.subplot(2, 1, 1)
    vis.plot(C0, m)
    b1=plt.colorbar()
    b1.set_label('Velocity (m/s)')
    plt.title('Initial Model')
    plt.xlabel('Length (m)')
    plt.ylabel('Depth (m)')
    plt.subplot(2, 1, 2)
    vis.plot(C, m)
    b2=plt.colorbar()
    b2.set_label('Velocity (m/s)')
    plt.title('True Model')
    plt.xlabel('Length (m)')
    plt.ylabel('Depth (m)')
    #模型及网格参数
    xmin = d.x.lbound #横坐标下限-0
    xmax = d.x.rbound #横坐标上限-9200
    zmin = d.z.lbound #纵坐标下限-0
    zmax = d.z.rbound #纵坐标上限-3000
    nx = m.x.n #横轴网格的点数-461
    nz=m.z.n #纵轴网格的点数-151
    #配置震源位置
    source_list = [] #震源列表
    source_x=4600#震源横坐标
    source_z=1500#震源纵坐标
    if source_x<xmin or source_x>xmax:
        raise Exception('source x not right..')
    if source_z<zmin or source_z>zmax:
        raise Exception('source z not right..')
    # RickerWavelet(10.0)-震源子波为峰值频率为10.0Hz的Ricker子波；
    # intensity-子波的强度倍数
    source_list.append(PointSource(m,(source_x,source_z), RickerWavelet(10.0),intensity=(10)))
    source_set = SourceSet(m, source_list)
    #配置检波器位置
    receiver_list=[]
    receiver_x=0#检波器横坐标-0
    receiver_z=0#检波器纵坐标-0
    if receiver_x<xmin or receiver_x>xmax:
        raise Exception('receiver x not right..')
    if receiver_z<zmin or receiver_z>zmax:
        raise Exception('receiver z not right..')
    receiver_list.append(PointReceiver(m,(receiver_x,receiver_z)))
    receiver_x = 4600#检波器横坐标-4600
    receiver_z = 0#检波器纵坐标-0
    if receiver_x < xmin or receiver_x > xmax:
        raise Exception('receiver x not right..')
    if receiver_z < zmin or receiver_z > zmax:
        raise Exception('receiver z not right..')
    receiver_list.append(PointReceiver(m, (receiver_x, receiver_z)))
    receiver_x = 9200#检波器横坐标-4600
    receiver_z = 0#检波器纵坐标-0
    if receiver_x < xmin or receiver_x > xmax:
        raise Exception('receiver x not right..')
    if receiver_z < zmin or receiver_z > zmax:
        raise Exception('receiver z not right..')
    receiver_list.append(PointReceiver(m, (receiver_x, receiver_z)))
    receivers = ReceiverSet(m,receiver_list)
    shots=[]
    shot = Shot(source_set, receivers)
    shots.append(shot)
    trange = (0,3)#正演时长，单位-秒
    # spatial_accuracy_order-空间精度，值越大计算时间越长
    # trange-仿真时长
    # kernel_implementation-使用c语言编写的计算程序
    solver = ConstantDensityAcousticWave(m,spatial_accuracy_order=2,trange=trange,kernel_implementation='cpp')
    true_model = solver.ModelParameters(m, {'C': C})
    wavefields = []
    #注意！执行此语句后会修改dt，即波场计算的时间间隔，原因未知
    # solver.model_parameters=true_model
    print('generate_seismic_data')
    #save_method-制定保存的地震记录格式，支持的参数有pickle savemat h5py
    # verbose-生成地震记录时是否输出更多信息，为True，会在屏幕上输出更多的log
    # wavefields 为波场数据，可用于生成波场动画
    t1=time.time()
    generate_seismic_data(shots, solver, true_model, save_method='savemat', verbose=True, wavefields=wavefields)
    print('generate time:{0}s'.format(time.time()-t1))
    #保存模型、波场
    newpath = r'./shots_time'#保存路径为当前文件所在目录下的shots_time文件夹
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    mod_path = newpath + '/mod_and_wavefield.mat'#保存的文件名为mod_and_wavefield.mat
    true_mod= C.reshape([nx,nz])#将Marmousi模型保存为二维矩阵
    newWaveFields=[]
    for i in range(len(wavefields)):
        newWaveFields.append(wavefields[i].reshape([nx,nz]))
    #注意！此处保存wavefields会占用大量内存，应分步保存
    # io.savemat(mod_path, mdict={'trueMod': true_mod,'waveFields':newWaveFields})
    # display_rate-波场动画显示间隔的帧数
    # vis.animate(wavefields, m, display_rate=15)

    print('inversion start...')
    objective = TemporalLeastSquares(solver)#构造最小二乘法目标函数
    invalg = LBFGS(objective)#使用L-BFGS算法进行寻优
    initial_value = solver.ModelParameters(m, {'C': C0})#寻优初始值
    print('Running Descent...')
    tt = time.time()#寻优起始时刻
    nsteps = 10#迭代次数
    status_configuration = {'value_frequency': 1,'residual_length_frequency': 1,'objective_frequency': 1,'step_frequency': 1,'step_length_frequency': 1,'gradient_frequency': 1,'gradient_length_frequency': 1,'run_time_frequency': 1,'alpha_frequency': 1,}
    line_search = 'backtrack'#使用梯度法求解凸优化问题时选择梯度下降步长的方法
    result = invalg(shots, initial_value, nsteps,line_search=line_search,status_configuration=status_configuration, verbose=True)#获得寻优结果
    print('...run time:  {0}s'.format(time.time() - tt))
    obj_vals = np.array([v for k, v in list(invalg.objective_history.items())])#各次迭代中获得的最优目标函数值
    plt.figure()
    plt.semilogy(obj_vals)
    plt.title('Objective Function Value of Each Iteration')
    plt.xlabel('Number of iterations')
    plt.ylabel('Value of objective function')
    plt.figure()
    clim = C.min(), C.max()
    vis.plot(result.C, m, clim=clim)
    plt.title('Reconstruction')
    b3 = plt.colorbar()
    b3.set_label('Velocity (m/s)')
    plt.xlabel('Length (m)')
    plt.ylabel('Depth (m)')
    plt.show()

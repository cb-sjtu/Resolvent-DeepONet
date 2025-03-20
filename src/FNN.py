import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, io
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import deepxde as dde
from deepxde.backend import tf
import tensorflow_addons as tfa
import math
from deepxde.nn.tensorflow_compat_v1.mionet import Direct
from deepxde.data.triple import Triple
# scale_0=io.loadmat('data_0605/database/stat_Re1000.mat')
#最优的结构
# num = [670,1000,1410,2000,2540,3030,3270,3630,3970,4060]
def network(problem, m,N_points):
    if problem == "ODE":
        branch_1 = [m, 200, 200]
        branch_2 = [N_points, 200, 200]
        trunk = [1, 200, 200]
    elif problem == "DR":
        branch_1 = [m, 200, 200]
        branch_2 = [m, 200, 200]
        trunk = [2, 200, 200]
    elif problem == "ADVD":
        branch_1 = [m, 200, 200]
        branch_2 = [m, 200, 200]
        trunk = [2, 300, 300, 300]
    elif problem == "flow":
       
        branch=[200,256,512,512,1024,200*173]
       
   
        trunk=[1,128,256,512,512,1024,2048,4096,2048,50*47]


    return branch,trunk

def get_data():
    
    kzs=np.load('data_gen/data_kzs.npy').astype(np.float32).reshape((5,1,173))
    uum=np.load('data_gen/data_um.npy').astype(np.float32)

    uum=kzs*uum

    uum_new=uum[[0,1,3,4],0:200].reshape((4,-1))
    uum_new_test=uum[[2],0:200].reshape((1,-1))

    u=np.load('data_gen/data_us.npy').astype(np.float32)[[0,1,3,5],0:200]
    u_test=np.load('data_gen/data_us.npy').astype(np.float32)[[2],0:200]

    trunk_out=np.load('data_gen/data_motai.npy').astype(np.float32)[[0,1,3,4]].reshape((4,200,-1))
    trunk_out_test=np.load('data_gen/data_motai.npy').astype(np.float32)[[2]].reshape((1,200,-1))
    motai=trunk_out.transpose(0,2,1)
    motai_test=trunk_out_test.transpose(0,2,1)


    for i in range(4):
        scaler_Euuc = StandardScaler().fit(trunk_out[i])
        std_Euuc = np.sqrt(scaler_Euuc.var_.astype(np.float32))
        trunk_out[i]=(trunk_out[i]-scaler_Euuc.mean_.astype(np.float32))/std_Euuc
    #np.save("data_gen/data_motai_zz.npy",trunk_out)
    np.save("data_gen/data_motai_zz.npy",trunk_out)
    for i in range(1):  
        scaler_Euuc_test = StandardScaler().fit(trunk_out_test[i])
        std_Euuc_test = np.sqrt(scaler_Euuc_test.var_.astype(np.float32))
        trunk_out_test[i]=(trunk_out_test[i]-scaler_Euuc_test.mean_.astype(np.float32))/std_Euuc_test


    kzs_s=kzs[[0,1,3,4]]
    kzs_s[0]=2*math.pi/kzs_s[0]*182.088
    kzs_s[1]=2*math.pi/kzs_s[1]*543.496
    kzs_s[2]=2*math.pi/kzs_s[2]*1994.756
    kzs_s[3]=2*math.pi/kzs_s[3]*5185.897
    kzs_s_test=kzs[[2]]
    kzs_s_test=2*math.pi/kzs_s_test*1000.512

    dcPs_s=np.load('data_gen/data_dcps.npy').astype(np.float32)[[0,1,3,4]].reshape((-1,1,46))
    dcPs_s_test=np.load('data_gen/data_dcps.npy').astype(np.float32)[[2]].reshape((-1,1,46))
 
    y=np.load('data_gen/data_yy.npy')
    real_2d_y=y
    real_2d_x=np.zeros((5,173))
    real_2d_x[[0,1,3,4]]=kzs_s.reshape((4,173))
    real_2d_x[2]=kzs_s_test.reshape((1,173))
    
    y[0]=y[0]*182.088
    y[1]=y[1]*543.496
    y[2]=y[2]*1000.512
    y[3]=y[3]*1994.756
    y[4]=y[4]*5185.897
    y=y[[0,1,3,4]]
    y_test=y[[2]]

    real_2d=np.zeros((5,200,173,2))

    for i in range(5):
    # 如果y的长度大于100则裁剪，如果小于则重复直到100
        real_2d_y_i = real_2d_y[i]
        
        repeats = 200 // real_2d_y_i.shape[0] + 1
        real_2d_y_i = np.tile(real_2d_y_i, repeats)[:200]  # 重复直到100个
        
        # 填充坐标
        for j in range(200):
            real_2d[i, j, :, 0] = real_2d_x[i]  # 复制x坐标对该列
            real_2d[i, j, :, 1] = real_2d_y_i[j]
   

    return u,u_test,uum_new,uum_new_test,trunk_out, trunk_out_test,dcPs_s,dcPs_s_test,kzs_s,kzs_s_test,y,y_test,real_2d,kzs




def main():
    dataframe="1-2-2-3"
    u,u_test,uum,uum_test,trunk_out, trunk_out_test, dcPs_s ,dcPs_s_test,kzs,kzs_test,yy,yy_test,real_2d,kzss, = get_data()
    kzs=np.reshape(kzs,(-1,1)).astype(np.float32)
    kzs_test=np.reshape(kzs_test,(-1,1)).astype(np.float32)
    uum=np.reshape(uum,(-1,200,173)).transpose(0,2,1).reshape((-1,200*173))
    # uum=(uum,y)
    uum_test=np.reshape(uum_test,(-1,200,173)).transpose(0,2,1).reshape((-1,200*173))
    # uum_test=(uum_test,y_test)
    trunk_out=trunk_out.transpose(0,2,1)
    trunk_out_test=trunk_out_test.transpose(0,2,1)
    trunk_out_input=(u,kzs)
    trunk_out_input_test=(u_test,kzs_test)
    
   
    problem = "flow"
    N_points = 87*24
    data = dde.data.Fifthple(trunk_out_input,uum,trunk_out_input_test,uum_test)
    m = 200*47*173
    activation = (
        ["relu", "relu", "relu"] if problem in ["ADVD"] else ["relu", "relu", "relu"]
    )
    initializer = "Glorot normal"
    

    branch_net,trunk_net,dot = network(problem,m,N_points)
   
   
    net = Direct(
        branch_net,
        trunk_net,
        {"branch1": activation[0], "branch2": activation[1], "trunk": activation[2]},
        kernel_initializer=initializer,
        regularization=None,
    )
    
    if isinstance(uum, tuple):
        uum_zhengze = uum[0]
    else:
        uum_zhengze = uum
    scaler = StandardScaler().fit(uum_zhengze)
    std = np.sqrt(scaler.var_.astype(np.float32))

    def output_transform( outputs):
        return outputs * std + scaler.mean_.astype(np.float32)           

    net.apply_output_transform(output_transform)
# andom_uniform
    model = dde.Model(data, net)
    model.compile("adam", lr=0.0002,
        loss="def_l2_relative_error",
        metrics=["l2 relative error"], 
        decay=("inverse time", 1, 1e-4))
    checker = dde.callbacks.ModelCheckpoint(
        dataframe+"/model.ckpt", save_better_only=False, period=1000
    )
    losshistory, train_state = model.train(epochs=30000, batch_size=None,display_every=50,callbacks=[checker],model_save_path=dataframe
    )  
    dde.saveplot(losshistory, train_state,issave=True,isplot=True,loss_fname=dataframe+"/loss.dat")







if __name__ == "__main__":
   
       
    main()

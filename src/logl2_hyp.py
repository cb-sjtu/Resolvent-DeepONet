import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import fft, io
from sklearn.preprocessing import StandardScaler,MinMaxScaler

import deepxde as dde
from deepxde.backend import tf
import tensorflow_addons as tfa
import math
from deepxde.nn.tensorflow_compat_v1.mionet import DeepONet_resolvent_jiami
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
        branch = tf.keras.Sequential(
            [
                
                tf.keras.layers.InputLayer(input_shape=(173*47*200)),
                tf.keras.layers.Reshape((173,47,200)),
                # tf.keras.layers.transpose,
                tf.keras.layers.Conv2D(200, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(200, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(500, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(1000, (3, 3), strides=1, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                # tf.keras.layers.Conv2D(1000, (1, 1), strides=1, activation="ReLU"),
                # tf.keras.layers.Conv2D(500, (1, 1), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(100, (1, 1), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(10, (1, 1), strides=1, activation="ReLU"),
                tf.keras.layers.Conv2D(1, (1, 1), strides=1, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(1000, activation="ReLU"),
                tf.keras.layers.Dense(1000, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                # tf.keras.layers.Dense(1000, activation="ReLU"),
                # tf.keras.layers.Dense(1000, activation="ReLU"),
                tf.keras.layers.Dense(500, activation="ReLU"),
                tf.keras.layers.Dense(200, activation="ReLU"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(47*50),
              
                tf.keras.layers.Reshape((47,50)),
                # tf.keras.layers.Lambda(lambda x: transpose_last_two_dims(x))

            ]
        )
        branch.summary()

        branch=[m,branch]
       
        # trunk =  tf.keras.Sequential(
        #     [
        #         tf.keras.layers.InputLayer(input_shape=(100,)),
        #         tf.keras.layers.Flatten(),
        #         # tf.keras.layers.Dense(128, activation="relu"),
        #         tf.keras.layers.Dense(200),
        #         # tf.keras.layers.Dense(500),
        #     ]
        # )
        # trunk.summary()
        trunk=[1,128,256,512,512,1024,2048,4096,2048,50*47]





        dot= tf.keras.Sequential(
        [   tf.keras.layers.InputLayer(input_shape=(87*24,2,)),
            # tf.keras.layers.Reshape((2,200, 1)),
            # # tf.keras.layers.Dense(1000, activation="relu"),
            # # tf.keras.layers.Dense(581248, activation="relu"),
            # # tf.keras.layers.Reshape((38,239,64)),
            # # tf.keras.layers.Conv2D(64, (5, 5), strides=2, activation="relu"),
            # tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=[2,2], strides=[5,5], activation="relu",padding="valid",data_format='channels_last'),
            # tf.keras.layers.Conv2D(filters=1, kernel_size=[1,1], strides=[1,1], activation="relu",padding="valid",data_format='channels_last'),
            tf.keras.layers.Flatten(),   
            # tf.keras.layers.Dense(1000, activation="relu"),
            # # tf.keras.layers.Dense(1000, activation="relu"),
            # tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Dense(87*24*1, activation="ReLU"),
            # tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Dense(87*24*100),
            # tf.keras.layers.Reshape((87*24,100)),

        ]
      )
    dot.summary()
    dot=[0,dot]
    return branch,trunk,dot

def get_data():
    
    kzs=np.load('data_gen/data_kzs.npy').astype(np.float32).reshape((5,1,173))
    uum=np.load('data_gen/data_um.npy').astype(np.float32)

    uum=kzs*uum

    uum_new=uum[[0,1,3,4],0:200].reshape((4,-1))
    uum_new_test=uum[[2],0:200].reshape((1,-1))
    
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
    y=y[[0,1,2,3,4]]
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

    
    

    return uum_new,uum_new_test,trunk_out, trunk_out_test,dcPs_s,dcPs_s_test,kzs_s,kzs_s_test,y,y_test,real_2d,kzs,motai,motai_test



def main():
    dataframe="model_Euukc_old_logl2"
    uum,uum_test,trunk_out, trunk_out_test, dcPs_s ,dcPs_s_test,lambda_z,lambda_z_test,yy,yy_test,real_2d,kzs,motai,motai_test = get_data()
    lambda_z=np.reshape(lambda_z,(-1,1)).astype(np.float32)
    lambda_z_test=np.reshape(lambda_z_test,(-1,1)).astype(np.float32)
    uum=np.reshape(uum,(-1,200,173)).transpose(0,2,1).reshape((-1,200*173))
    # uum=(uum,y)
    uum_test=np.reshape(uum_test,(-1,200,173)).transpose(0,2,1).reshape((-1,200*173))
    # uum_test=(uum_test,y_test)
    trunk_out=trunk_out.transpose(0,2,1)
    trunk_out_test=trunk_out_test.transpose(0,2,1)
    trunk_out_input=(trunk_out.reshape((-1,173*47*200)),lambda_z,motai ,dcPs_s)
    trunk_out_input_test=(trunk_out_test.reshape((-1,173*47*200)),lambda_z_test,motai_test,dcPs_s_test)
    
    # #归一化
    # scaler_Euuc = MinMaxScaler().fit(trunk_out_input[0])
    # scaler_uum = MinMaxScaler().fit(uum)
    # trunk_out_input = (scaler_Euuc.transform(trunk_out_input[0]),scaler_Euuc.transform(trunk_out_input[1]))
    # trunk_out_input_test = (scaler_Euuc.transform(trunk_out_input_test[0]),scaler_Euuc.transform(trunk_out_input_test[1]))
    # # uum = scaler_uum.transform(uum)
    problem = "flow"
    N_points = 87*24
    data = dde.data.Fifthple(trunk_out_input,uum,trunk_out_input_test,uum_test)
    m = 200*47*173
    activation = (
        ["relu", "relu", "relu"] if problem in ["ADVD"] else ["relu", "relu", "relu"]
    )
    initializer = "Glorot normal"
    

    branch_net,trunk_net,dot = network(problem,m,N_points)
   
   
    net = DeepONet_resolvent_jiami(
        branch_net,
        trunk_net,
        dot,
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
    model.compile("adam", lr=0.0001,
        loss="def_l2_relative_error",
        metrics=["l2 relative error"], 
        decay=("inverse time", 1, 1e-4))
    checker = dde.callbacks.ModelCheckpoint(
        dataframe+"/model.ckpt", save_better_only=False, period=200
    )
    losshistory, train_state = model.train(epochs=30000, batch_size=None,display_every=50,callbacks=[checker],model_save_path=dataframe
    )  
    dde.saveplot(losshistory, train_state,issave=True,isplot=True,loss_fname=dataframe+"/loss.dat")





    # model.restore(dataframe+"/model.ckpt-30000.ckpt")
    real_2d=np.transpose(real_2d,(0,2,1,3))
    real_2d=np.reshape(real_2d,(-1,200*173,2))
    # real_2d=np.transpose(real_2d,(0,2,1))
    cols = 47
    rows = 173

    # 定义网格的范围
    x_min = 0
    x_max = 47
    y_min = 0
    y_max = 173

    # 在x和y范围内生成均匀分布的网格点
    x = np.linspace(x_min, x_max, cols)
    y = np.linspace(y_min, y_max, rows)

    # 创建一个二维数组，其中每个元素都是一个包含x和y坐标的元组
    y = np.array([(x_val, y_val) for y_val in y for x_val in x])

    # y=np.zeros((100*87,2))
    
    # xy=np.load('data_xy_5.npy').astype(np.float32)
    # xy=np.log10(xy)
    dkxs_s=np.load('data_gen/data_dkzs.npy').astype(np.float32)[[0,1,2,3,4]].reshape((-1))[:172]
    dkxs_s_test=np.load('data_gen/data_dkzs.npy').astype(np.float32)[[2]].reshape((-1))[:172]
    now="train"
    if now=="train":
        real_2d=real_2d[[0,1,2,3,4]]
        label_train = model.predict(trunk_out_input)
        # label_train=scaler_uum.inverse_transform(label_train)
        # np.save('data_show/q.npy',q)
        # np.save('data_show/n.npy',n)
        # np.save('data_show/label_train.npy',label_train)
        # np.save('data_show/x1.npy',x1)
        # np.save('data_show/x_sum1.npy',x_sum1)
        # np.save('data_show/x_sum2.npy',x_sum2)
        # np.save('data_show/x_sum3.npy',x_sum3)
        # np.save('data_show/x_1.Anpy',x_1)
        
        # for i in range(100):
        #     np.savetxt('q_1_'+str(i)+'.dat',q[1,i])
        # branch_out=np.reshape(n,(-1,2088))
        # label_train=scaler.inverse_transform(label_train)
        label_train=np.reshape(label_train,(-1,173*47))
        label_train=np.transpose(label_train)
        uum=np.transpose(uum)
        # error=np.abs(label_train-uum)
        #y=np.transpose(y)
        for i in range(5): 
            # result=np.hstack((real_2d[i].reshape((200*173,2)),label_train[:200*173,i].reshape((200*173,1)),uum[:200*173,i].reshape((200*173,1)),error[:200*173,i].reshape((200*173,1))))
            result=np.hstack((y.reshape((47*173,2)),label_train[:47*173,i].reshape((47*173,1))))
            np.savetxt(dataframe+'/1real-Euukc_new_6_train_'+str(i)+'.dat',result)
            #np.savetxt(dataframe+'/loss2'+str(i)+'_branch_w100.dat',branch_out[i])
            with open(dataframe+'/1real-Euukc_new_6_train_'+str(i)+'.dat', 'r') as file :
                filedata = file.read()

        # 创建新的行
            # newline = 'VARIABLES="x","y","u",up","error"'+'\n'+'ZONE I=200,J=173'+'\n'
            newline = 'VARIABLES="x","y","u"'+'\n'+'ZONE I=47,J=173'+'\n'

            # 把新的行加入到原文件内容之前
            filedata = newline + filedata

            # 写回文件，注意使用'w'模式，它将覆盖原文件内容
            with open(dataframe+'/1real-Euukc_new_6_train_'+str(i)+'.dat', 'w') as file:
                file.write(filedata)

        #积分
        label_train=np.transpose(label_train).reshape((-1,173,200))
        kzs_train=kzs[[0,1,2,3,4]].reshape((-1,173,1))
        label_train=label_train/kzs_train
        uum=np.transpose(uum)
        uum=uum.reshape((-1,173,200))
        uum=uum/kzs_train

        label_train=np.reshape(label_train,(-1,173,200))
        label_train=np.transpose(label_train,(0,2,1)).reshape((-1,173))
        label_train_1=label_train[:,:-1]
        label_train_2=label_train[:,1:]
        label_train_sum=(label_train_1+label_train_2).reshape((-1,200,172))
        dkxs_s=dkxs_s.reshape((1,1,172))
        label_profile=(label_train_sum*dkxs_s*0.5).sum(axis=2).transpose()

        # label_train=np.reshape(label_train,(-1,173,200))
        # # label_train=np.transpose(label_train,(0,2,1)).reshape((-1,173))
        # label_train_1=label_train[:,:-1]
        # label_train_2=label_train[:,1:]
        # label_train_sum=(label_train_1+label_train_2)
        # label_profile=(label_train_sum*dkxs_s*0.5).sum(axis=2).transpose()

        # uum=np.reshape(uum,(-1,173))
        # uum=np.reshape(uum,(-1,173,200))
        uum=np.transpose(uum,(0,2,1)).reshape((-1,173))
        uum_1=uum[:,:-1]
        uum_2=uum[:,1:]
        uum_sum=(uum_1+uum_2).reshape((-1,200,172))
        dkxs_s=dkxs_s.reshape((1,1,172))
        uum_profile=(uum_sum*dkxs_s*0.5).sum(axis=2).transpose()
         
        yy=np.transpose(yy)
        
        for i in range(5): 
            result=np.hstack((yy[:200,i].reshape((200,1)),label_profile[:200,i].reshape((200,1)),uum_profile[:200,i].reshape((200,1))))
            np.savetxt(dataframe+'/2loss1_7_train'+str(i)+'.dat',result)


        

    else:

        # y[0]=(io.loadmat('data_0605/database/stat_Re1000.mat')['y'].reshape((200,)))[:100]*1000.512  

        # y=np.arange(0,100*87).reshape((1,100*87))
    
    
        real_2d=real_2d[[2]]
        label_pre = model.predict(trunk_out_input_test)
        # label_pre=scaler_uum.inverse_transform(label_pre)
        label_pre=np.reshape(label_pre,(-1,173*200))
        label_pre=np.transpose(label_pre)
        #label_pre=np.reshape(label_pre,(-1,173,200))
        
        uum_test=np.transpose(uum_test)
     
        error=np.abs(label_pre-uum_test)


        for i in range(1): 
            
            #result=np.hstack((y.reshape((47*173,2)),label_pre[:47*173,i].reshape((47*173,1))))
            result=np.hstack((real_2d[i],label_pre[:200*173,i].reshape((200*173,1)),uum_test[:200*173,i].reshape((200*173,1)),error[:200*173,i].reshape((200*173,1))))
            np.savetxt(dataframe+'/Euukc_new_6_test'+str(i)+'.dat',result)
            with open(dataframe+'/Euukc_new_6_test'+str(i)+'.dat', 'r') as file :
                filedata = file.read()

        # 创建新的行
            newline = 'VARIABLES="x","y","u","u_p","u_r"'+'\n'+'ZONE I=200,J=173'+'\n'
            #newline = 'VARIABLES="x","y","u"'+'\n'+'ZONE I=47,J=173'+'\n'
            # 把新的行加入到原文件内容之前
            filedata = newline + filedata

            # 写回文件，注意使用'w'模式，它将覆盖原文件内容
            with open(dataframe+'/Euukc_new_6_test'+str(i)+'.dat', 'w') as file:
                file.write(filedata)
    
        #积分
        label_pre=np.transpose(label_pre)
        uum_test=np.transpose(uum_test)
        kzs_test=kzs[[2]].reshape((-1,173,1))
        label_pre=label_pre/kzs_test
        uum_test=uum_test/kzs_test

        label_pre=np.reshape(label_pre,(-1,173))
        label_pre=np.reshape(label_pre,(-1,173,200))
        label_pre=np.transpose(label_pre,(0,2,1)).reshape((-1,173))
        label_pre_1=label_pre[:,:-1]
        label_pre_2=label_pre[:,1:]
        label_pre_sum=(label_pre_1+label_pre_2).reshape((-1,200,172))
        label_profile=(label_pre_sum*dkxs_s_test*0.5).sum(axis=2).transpose()

        uum_test=np.reshape(uum_test,(-1,173))
        uum_test=np.reshape(uum_test,(-1,173,200))
        uum_test=np.transpose(uum_test,(0,2,1)).reshape((-1,173))
        uum_test_1=uum_test[:,:-1]
        uum_test_2=uum_test[:,1:]
        uum_test_sum=(uum_test_1+uum_test_2).reshape((-1,200,172))
        uum_profile=(uum_test_sum*dkxs_s_test*0.5).sum(axis=2).transpose()
        
        yy_test=np.transpose(yy_test)
        for i in range(1):
            result=np.hstack((yy_test[:200,i].reshape((200,1)),label_profile[:200,i].reshape((200,1)),uum_profile[:200,i].reshape((200,1))))
            np.savetxt(dataframe+'/2loss1_7_test'+str(i)+'.dat',result)


if __name__ == "__main__":
   
       
    main()

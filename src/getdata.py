import numpy as np
import math
from sklearn.preprocessing import StandardScaler

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


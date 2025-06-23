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
from getdata import get_data
from network import network
from postprocess import save_results




def main():
    #初始化wandb
   
    dataframe="model_Euukc_old_logl2"
    uum,uum_test,trunk_out, trunk_out_test, dcPs_s ,dcPs_s_test,lambda_z,lambda_z_test,yy,yy_test,real_2d,kzs,motai,motai_test = get_data()
    kzs_test=kzs[[2]].reshape((-1,176,1))
    kzs=kzs[[0,1,3,4]].reshape((-1,176,1))
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
        loss="mse",
        metrics=["l2 relative error"], 
        decay=("inverse time", 1, 1e-4))
    checker = dde.callbacks.ModelCheckpoint(
        dataframe+"/model.ckpt", save_better_only=False, period=200
    )
    losshistory, train_state = model.train(epochs=30000, batch_size=None,display_every=50,callbacks=[checker],model_save_path=dataframe
    )  
    dde.saveplot(losshistory, train_state,issave=True,isplot=True,loss_fname=dataframe+"/loss.dat")

    # 后处理
    model.restore(dataframe + "/model.ckpt-30000.ckpt")
    label_train = model.predict(trunk_out_input)
    save_results(dataframe, real_2d, label_train, uum, None, yy,kzs, mode="train")

    label_test = model.predict(trunk_out_input_test)
    save_results(dataframe, real_2d, label_test, uum_test, None, yy_test, kzs_test, mode="test")




if __name__ == "__main__":
   
       
    main()

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 20:41:48 2019

@author: akarsh
"""

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter
import numpy as np
from scipy import io
import matplotlib.pyplot as plt

import random



def v2q(w):
    x = w[0]
    y = w[1]
    z = w[2]
    theta = np.sqrt(x*x+y*y+z*z)
    if theta ==0:
        return np.array([1,0,0,0])
    else:
        temp1 = np.cos(theta/2)
        temp2 = np.sin(theta/2)
        w1 = temp2*x/theta
        w2 = temp2*y/theta
        w3 = temp2*z/theta
        return np.array([temp1, w1,w2,w3])
def v2q_vec(w):
    if w.shape[0] <4:
        w = np.transpose(w)    
    x = w[:,0]
    y = w[:,1]
    z = w[:,2]
    theta = np.sqrt(x*x+y*y+z*z)
    if theta ==0:
        return np.array([1,0,0,0])
    else:
        temp1 = np.cos(theta/2)
        temp2 = np.sin(theta/2)
        w1 = temp2*x/theta
        w2 = temp2*y/theta
        w3 = temp2*z/theta
        return np.array([temp1, w1,w2,w3])
  

def q2v(q):
    q1 = q[0]
    q2 = q[1]
    q3 = q[2]
    q4 = q[3]
    theta = 2*np.arccos(q1)
    if theta ==0:
        return np.zeros((3))
    else:
        temp = np.sqrt(1-q1*q1)
        return np.array([theta*q2/temp,theta*q3/temp,theta*q4/temp])
    
def q_inv(q):
    #start = time.time()
    q_norm = np.linalg.norm(q)
    temp = np.zeros(q.shape)
    temp[0] = q[0]
    temp[1] = -q[1]
    temp[2] = -q[2]
    temp[3] = -q[3]
    #print("--- %s seconds --- qinv" % (time.time() - start))
    return temp/(q_norm**2)
def q_inv_vec(q):
    #start = time.time()
    #q is nx4
    q_norm = np.linalg.norm(q)
    temp = np.zeros(q.shape)
    temp[:,0] = q[:,0]
    temp[:,1] = -q[:,1]
    temp[:,2] = -q[:,2]
    temp[:,3] = -q[:,3]
    #print("--- %s seconds --- qinv" % (time.time() - start))
    return temp/(q_norm**2)

def q_mult(q1, r1):
    q = np.zeros((1,4))
    r = np.zeros((1,4))
    q[:,0] = q1[0]
    q[:,1] = q1[1]
    q[:,2] = q1[2]
    q[:,3] = q1[3]
    
    r[:,0] = r1[0]
    r[:,1] = r1[1]
    r[:,2] = r1[2]
    r[:,3] = r1[3]
    t = np.empty(([1,4]))
    t[:,0] = r[:,0]*q[:,0] - r[:,1]*q[:,1] - r[:,2]*q[:,2] - r[:,3]*q[:,3]
    t[:,1] = (r[:,0]*q[:,1] + r[:,1]*q[:,0] - r[:,2]*q[:,3] + r[:,3]*q[:,2])
    t[:,2] = (r[:,0]*q[:,2] + r[:,1]*q[:,3] + r[:,2]*q[:,0] - r[:,3]*q[:,1])
    t[:,3] = (r[:,0]*q[:,3] - r[:,1]*q[:,2] + r[:,2]*q[:,1] + r[:,3]*q[:,0])
    z = np.zeros((4))
    z[0] = t[:,0]
    z[1] = t[:,1]
    z[2] = t[:,2]
    z[3] = t[:,3]

    return z

def q_mult_vec(q1, r1):
    #r1 is 4xn, q1 is normal
    r = np.zeros((r1.shape[0],4))
    q = np.zeros((1,4))
    q[:,0] = q1[0]
    q[:,1] = q1[1]
    q[:,2] = q1[2]
    q[:,3] = q1[3]
    
    r[:,0] = r1[:,0]
    r[:,1] = r1[:,1]
    r[:,2] = r1[:,2]
    r[:,3] = r1[:,3]
    t = np.empty(([r1.shape[0],4]))
    t[:,0] = r[:,0]*q[:,0] - r[:,1]*q[:,1] - r[:,2]*q[:,2] - r[:,3]*q[:,3]
    t[:,1] = (r[:,0]*q[:,1] + r[:,1]*q[:,0] - r[:,2]*q[:,3] + r[:,3]*q[:,2])
    t[:,2] = (r[:,0]*q[:,2] + r[:,1]*q[:,3] + r[:,2]*q[:,0] - r[:,3]*q[:,1])
    t[:,3] = (r[:,0]*q[:,3] - r[:,1]*q[:,2] + r[:,2]*q[:,1] + r[:,3]*q[:,0])
    z = np.zeros((r1.shape[0],4))
    z[:,0] = t[:,0]
    z[:,1] = t[:,1]
    z[:,2] = t[:,2]
    z[:,3] = t[:,3]

    return z
def q_mult_vec_all(q1, r1):
    q = np.zeros((q1.shape))
    r = np.zeros((r1.shape))
    q[:,0] = q1[:,0]
    q[:,1] = q1[:,1]
    q[:,2] = q1[:,2]
    q[:,3] = q1[:,3]
    
    r[:,0] = r1[:,0]
    r[:,1] = r1[:,1]
    r[:,2] = r1[:,2]
    r[:,3] = r1[:,3]
    t = np.empty(([r1.shape[0],4]))
    t[:,0] = r[:,0]*q[:,0] - r[:,1]*q[:,1] - r[:,2]*q[:,2] - r[:,3]*q[:,3]
    t[:,1] = (r[:,0]*q[:,1] + r[:,1]*q[:,0] - r[:,2]*q[:,3] + r[:,3]*q[:,2])
    t[:,2] = (r[:,0]*q[:,2] + r[:,1]*q[:,3] + r[:,2]*q[:,0] - r[:,3]*q[:,1])
    t[:,3] = (r[:,0]*q[:,3] - r[:,1]*q[:,2] + r[:,2]*q[:,1] + r[:,3]*q[:,0])
    z = np.zeros((r1.shape[0],4))
    z[:,0] = t[:,0]
    z[:,1] = t[:,1]
    z[:,2] = t[:,2]
    z[:,3] = t[:,3]

    return z
def q_mult_vec_other(q1, r1):
    #q1 is 4xn, r1 is normal
    q = np.zeros((q1.shape[0],4))
    r = np.zeros((1,4))
    q[:,0] = q1[:,0]
    q[:,1] = q1[:,1]
    q[:,2] = q1[:,2]
    q[:,3] = q1[:,3]
    
    r[:,0] = r1[0]
    r[:,1] = r1[1]
    r[:,2] = r1[2]
    r[:,3] = r1[3]
    t = np.empty(([q1.shape[0],4]))
    t[:,0] = r[:,0]*q[:,0] - r[:,1]*q[:,1] - r[:,2]*q[:,2] - r[:,3]*q[:,3]
    t[:,1] = (r[:,0]*q[:,1] + r[:,1]*q[:,0] - r[:,2]*q[:,3] + r[:,3]*q[:,2])
    t[:,2] = (r[:,0]*q[:,2] + r[:,1]*q[:,3] + r[:,2]*q[:,0] - r[:,3]*q[:,1])
    t[:,3] = (r[:,0]*q[:,3] - r[:,1]*q[:,2] + r[:,2]*q[:,1] + r[:,3]*q[:,0])
    z = np.zeros((q1.shape[0],4))
    z[:,0] = t[:,0]
    z[:,1] = t[:,1]
    z[:,2] = t[:,2]
    z[:,3] = t[:,3]

    return z
def rot2euler(R):
    roll = -np.arcsin(R[1,2])
    pitch = -np.arctan2(-R[0,2]/np.cos(roll),R[2,2]/np.cos(roll))
    yaw = -np.arctan2(-R[1,0]/np.cos(roll),R[1,1]/np.cos(roll))
    
    return roll, pitch, yaw

def rot2euler_vec(R):
    roll = -np.arcsin(R[1,2,:])
    pitch = -np.arctan2(-R[0,2,:]/np.cos(roll),R[2,2,:]/np.cos(roll))
    yaw = -np.arctan2(-R[1,0,:]/np.cos(roll),R[1,1,:]/np.cos(roll))
    
    return roll, pitch, yaw


def norm_quat(q):
    return np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
def norm_quat_vec(q):
    return np.sqrt(q[:,0]*q[:,0] + q[:,1]*q[:,1] + q[:,2]*q[:,2] + q[:,3]*q[:,3])


def quat2rot_vec(q):
    q = q/np.transpose(np.matrix(norm_quat_vec(q)))
    qhat = np.zeros([3,3,q.shape[0]])
    cd = np.identity(3)
    
    q = np.matrix(q)
    #qhat = np.zeros([3,3])
    qhat[0,1,:] = -q[:,3].reshape(np.shape(q)[0],)
    qhat[0,2,:] = q[:,2].reshape(np.shape(q)[0],)
    qhat[1,2,:] = -q[:,1].reshape(np.shape(q)[0],)
    qhat[1,0,:] = q[:,3].reshape(np.shape(q)[0],)
    qhat[2,0,:] = -q[:,2].reshape(np.shape(q)[0],)
    qhat[2,1,:] = q[:,1].reshape(np.shape(q)[0],)
    
    bd = np.repeat(cd[:, :, np.newaxis], q.shape[0], axis=2)

    R = bd + 2*qhat*qhat + 2*np.array(q[:,0]).T*qhat
    return R

def norm_vec(q):
    return (np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]))

def q_exp(rs):
    exp_q = np.zeros(np.shape(rs))
    
    exp_q[0] = np.cos((norm_vec(rs[1:])))
    if norm_vec(rs[1:]) == 0:
        exp_q = np.array([exp_q[0],0,0,0])
    else:
        exp_q[1:] = np.dot(rs[1:]/norm_vec(rs[1:]), np.sin(norm_vec(rs[1:])))
    return (np.exp(rs[0])*exp_q)

def quat2rot(q):
    q = q/norm_quat(q)
    q = np.matrix(q)
    qhat = np.zeros([3,3])
    qhat[0,1] = -q[:,3]
    qhat[0,2] = q[:,2]
    qhat[1,2] = -q[:,1]
    qhat[1,0] = q[:,3]
    qhat[2,0] = -q[:,2]
    qhat[2,1] = q[:,1]

    R = np.identity(3) + 2*np.dot(qhat,qhat) + 2*np.array(q[:,0])*qhat
    return R
    
def quat_avg(q_k,Y):
    qt = q_k
    iterations = 100
    temp1 = np.zeros([1,4])
    e = np.zeros((Y.shape[0],3))
    for i1 in range(iterations):
        for i2 in range(Y.shape[0]):
            Y[i2,:] = Y[i2,:]/norm_quat(Y[i2,:])
            qe = q_mult(Y[i2,:], q_inv(qt))
            if np.round(norm_vec(qe[1:]),8) == 0:
                if np.round(norm_quat(qe),8) == 0:
                    e[i2,:] = np.zeros(3)
                else:
                    e[i2,:] = np.zeros(3)
            if np.round(norm_vec(qe[1:]),8) != 0:
                if np.round(norm_quat(qe),8) == 0:
                    e[i2,:] = np.zeros(3)
                else:
                    temp1[0,0] = np.log(norm_quat(qe))
                    temp1[0,1:4] = np.dot((qe[1:]/norm_vec(qe[1:])),np.arccos(qe[0]/norm_quat(qe)))
                    e[i2,:] = 2*temp1[0,1:4]
                    e[i2,:] = ((-np.pi + (np.mod((norm_vec(e[i2,:]) + np.pi),(2*np.pi))))/norm_vec(e[i2,:]))*e[i2,:]
        e_mean = np.mean(e, axis = 0)
        temp2 = np.array(np.zeros(4))
        temp2[0] = 0
        temp2[1:] = e_mean/2.0
        qt = q_mult(q_exp(temp2),qt)

        if norm_vec(e_mean) < 0.0001:
            return qt, e
def scalingas(imu):
    
    imu_data = np.transpose(imu['vals'])  
    accel = imu_data[:,:3]
    gyro = imu_data[:,3:]
    
    accel[:,:2] = -accel[:,:2]
    
    vref = 3300
    sens = 330.0
    acc_factor = vref/1023.0/sens
    acc_bias = accel[0] - np.array([0,0,1])/acc_factor
    #acc_bias = np.array([[510.8436],[500.9864],[501]]).T
    #accel_input = (accel - acc_bias)*acc_factor
    accel_input =( (np.array(accel*acc_factor))- acc_bias*acc_factor)
   
    gyro_x = np.array(imu_data[:,4])
    gyro_y = np.array(imu_data[:,5])
    gyro_z = np.array(imu_data[:,3])
    gyro = [gyro_x, gyro_y, gyro_z]
    gyro = np.array(gyro)
    gyro = np.transpose(gyro)
    gyros_sens = 3.33
    gyro_factor = vref/1023/gyros_sens
    #gyro_bias = np.mean(gyro, axis=0)
    gyro_bias = gyro[0]
    #gyro_bias = np.array([[379],[376],[376]]).T
    #gyro_bias = np.array([530,530,530])
    
    gyro_input =((np.array(gyro*gyro_factor))- gyro_bias*gyro_factor)*np.pi/180
    
    '''
    sens_acc = np.array([[33.5295],[330.5295],[-338.4347]])
    sens_gyro = np.array([[-5.3],[3.5],[3.5]])
    acc_bias = np.array([[510.8436],[500.9864],[501]])
    gyro_bias = np.array([[379],[376],[376]])
    accel_input =np.divide (((accel - acc_bias.T)*(3300/1023)*9.81),sens_acc.T)
    gyro_input = np.divide (((gyro - gyro_bias.T)*(3300/1023)*(np.pi/180)),sens_gyro.T)
    '''
    return accel_input, gyro_input

def sigma_points(q_k,p_k,Q):
    n = p_k.shape[0]
    X = np.zeros((2*n, 4))
    S = np.linalg.cholesky(p_k + Q)
    W = np.zeros((3,2*n))
    W = np.hstack((S * np.sqrt(2*n), -S * np.sqrt(2*n)))
    q_k_W = np.zeros((2*n, 4))
    for k in range(2*n):
        q_k_W[k,:] = v2q(W[:, k])
    X = q_mult_vec(q_k, q_k_W)
    #X = np.vstack((q_k, X))
    return X
    
def transformed_sigma_points(X, q_delta):
    Y = np.zeros((X.shape[0],4))
    Y = q_mult_vec_other(X,q_delta)
    return Y

def mean_covariance(Y,e):
    p_pred = np.zeros((3,3)) 
    for k2 in range(Y.shape[0]):
        W_i = np.zeros((1,3))
        W_i[0][0] = e[k2,:][0]
        W_i[0][1] = e[k2,:][1]
        W_i[0][2] = e[k2,:][2]
        p_pred = p_pred + np.transpose(W_i)*W_i
    p_pred = p_pred/12
    return p_pred  

def measurement_step(Y,g,R,e):
    Z_i = np.zeros((Y.shape[0],3))
    temp = q_mult_vec_all(q_mult_vec_other(q_inv_vec(Y),g), Y)
    Z_i = temp[:,1:]
    #COmpute the mean
    z_k = np.mean(Z_i,axis = 0)
    
    #Compute the cvoarinaces
    p_zz = np.zeros((3,3))
    p_xz = np.zeros((3,3))
    temp = np.matrix(Z_i-z_k)
    for f in range(Z_i.shape[0]):
        W_i = np.zeros((1,3))
        W_i[0][0] = e[f,:][0]
        W_i[0][1] = e[f,:][1]
        W_i[0][2] = e[f,:][2]
        p_zz = p_zz + np.transpose(temp[f])*temp[f]
        p_xz = p_xz + np.transpose(W_i)*temp[f]
    p_zz = p_zz/12.0
    p_xz = p_xz/12.0
    #innovation
    p_vv = p_zz + R
    
    return p_zz,p_xz,p_vv,z_k

def update_step(acc,z_k,q_pred,p_pred,p_vv,k_gain):
    v_k = acc - z_k
    temp_quat = (np.array(k_gain).dot(v_k))[:3]
    q_k = q_mult(v2q(temp_quat),q_pred)
    p_k = p_pred - np.array(k_gain).dot(p_vv).dot(np.array(k_gain).T)
    return q_k,np.array(p_k)

def process_step(q0,q_k,p_k,Q,i,imu_time_step,omega):
    X = sigma_points(q_k,p_k,Q)
    if i==0:
        q_delta = v2q(omega*imu_time_step[0])
    else:
        q_delta = v2q(omega*(imu_time_step[i] - imu_time_step[i-1]))
    #Transforming X to Y 
    Y=transformed_sigma_points(X, q_delta)
    #precition step
    q_pred,e = quat_avg(q_k, Y)
    p_pred = mean_covariance(Y,e) 
    
    return q_pred,p_pred,e,Y

def estimate_rot(data_num=1):
	#your code goes here  
    bluff = 0
    
    imu = io.loadmat("imu/imuRaw1.mat")
    vicon = io.loadmat("vicon/viconRot1.mat")
    imu_time_step = imu['ts'].T 
    vicon_data = vicon['rots']

   
    #Get scaled Data
    acc_input, omega_input = scalingas(imu)
        
    while(bluff == 0):    
        #Tuning
        p_k_tune = 0.0001
        Q_tune =  0.0001
        R_tune =  0.0001
        q_k_tune = 1
        q_k = np.array([1,0,0,0])*q_k_tune
        q0 = np.array([1,0,0,0])
        p_k = np.identity(3)*p_k_tune
        Q = np.identity(3)
        R = np.identity(3)
        #Initial Assumption
        #q_k = np.array([1,0,0,0])*q_k_tune
        '''
        q_k = np.array([[0.9999813],[-0.0008],[-0.0056252],[0.0022751]])
        q0 = np.array([1,0,0,0])
        p_k = np.identity(3)
        Q = np.identity(3)*0.0000015 + np.ones((3,3))*0.0000000002
        R = np.identity(3)*0.0345 + np.full(3,0.00295)
        '''
        
        Q[0,0] = random.randint(1,10)*Q_tune
        Q[1,1] = random.randint(1,10)*Q_tune
        Q[2,2] = random.randint(1,10)*Q_tune
        
        R[0,0] = random.randint(10,30)*R_tune
        R[1,1] = random.randint(1,10)*R_tune
        R[2,2] = random.randint(1,10)*R_tune
        '''
        Q[0,0] = 1.5*Q_tune
        Q[1,1] = 1.3*Q_tune
        Q[2,2] = 1*Q_tune
        
        R[0,0] = random.uniform(0,50)*R_tune
        R[1,1] = random.uniform(0,10)*R_tune
        R[2,2] = random.uniform(0,10)*R_tune
        '''
        #R = np.array([[13.5047, 0.0791, -1.3018],[0.0791,15.8731,-0.5306],[-1.3018,-0.5306,12.6943]])
        #R = np.array([[35*R_tune, 0.0, 0.0],[0.0,2*R_tune,0.0],[0.0,0.0,2*R_tune]])
        
        g = np.array([0,0,0,1])
        
        #for keeping a record of average dt
        count1 = 1
        
        for i in range(imu_time_step.shape[0]):
            acc = acc_input[i,:]
            omega = omega_input[i,:]
            
            if i==0:
                initial = q0
            #Process Step
            q_pred,p_pred,e,Y = process_step(q0,q_k,p_k,Q,i,imu_time_step,omega)
            
            #Measurement step
            p_zz,p_xz,p_vv,z_k = measurement_step(Y,g,R,e)
            
            #Kalman Gain
            k_gain = np.dot(p_xz,np.linalg.inv(p_vv))
            
            #update
            q_k,p_k = update_step(acc,z_k,q_pred,p_pred,p_vv,k_gain)
            #p_k = p_k + 1E-4*np.eye(3)
            initial = np.vstack((initial, q_k))
            if i == count1*1000:
                print(i)
                count1 = count1+1
            
        
        roll = np.zeros([np.shape(initial)[0], 1])
        pitch = np.zeros([np.shape(initial)[0], 1])
        yaw = np.zeros([np.shape(initial)[0], 1])
        
        R = np.zeros((3,3,np.shape(initial)[0]))
        for n in range(initial.shape[0]):
            R[:,:,n] = quat2rot(initial[n,:])
        #    roll[n], pitch[n], yaw[n]= rot2euler(R)
        #R = quat2rot_vec(initial)   
        roll, pitch, yaw = rot2euler_vec(R)
        #vicon_angles = np.zeros((vicon_data.shape[2], 3))
        vicon_roll = np.zeros([np.shape(initial)[0], 1])
        vicon_pitch = np.zeros([np.shape(initial)[0], 1])
        vicon_yaw = np.zeros([np.shape(initial)[0], 1])
        for b in range(vicon_data.shape[2]):
            vicon_roll[b], vicon_pitch[b], vicon_yaw[b]= rot2euler(vicon_data[:,:,b])
    
        roll_error = np.sqrt(np.mean((roll-vicon_roll)**2))
        pitch_error =np.sqrt(np.mean((pitch-vicon_pitch)**2))
        yaw_error = np.sqrt(np.mean((yaw-vicon_yaw)**2))
        
        if (max(roll_error,pitch_error,yaw_error)<0.5):
            bluff = 1
            print(R[0,0],R[1,1],R[2,2])
            print(roll_error,pitch_error,yaw_error)
    
        print(roll_error,pitch_error,yaw_error)
            
    plt.figure(1)
    plt.subplot(311)
    plt.plot(vicon_roll, 'b', roll, 'r')
    #plt.plot(vicon_roll, 'b')
    plt.ylabel('Roll')
    plt.subplot(312)
    plt.plot(vicon_pitch, 'b', pitch, 'r')
    #plt.plot(vicon_pitch, 'b')
    plt.ylabel('Pitch')
    plt.subplot(313)
    plt.plot(vicon_yaw, 'b', yaw, 'r')
    #plt.plot(vicon_yaw, 'b')#
    plt.ylabel('Yaw')
    plt.show()
    return roll,pitch,yaw

if __name__ == '__main__':
    estimate_rot(data_num=1)
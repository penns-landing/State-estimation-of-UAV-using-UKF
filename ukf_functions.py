#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:45:40 2019

@author: akarsh
"""

import numpy as np
from quat_functions import *

def sigma_points(q_k,p_k,Q):
    n = p_k.shape[0]
    X = np.zeros((2*n, 4))
    S = np.linalg.cholesky(p_k + Q)
    W = np.hstack((S * np.sqrt(2*n), -S * np.sqrt(2*n)))
    for k in range(2*n):
        q_k_W = v2q(W[:, k])
        X[k, :4] = q_mult(q_k, q_k_W)
    #X = np.vstack((q_k, X))
    return X
    
def transformed_sigma_points(X, q_delta):
    Y = np.zeros((X.shape[0],4))
    #q_delta = v2q(omega*dt)
    for jd in range(X.shape[0]):
        temp_q = X[jd, :]
        Y[jd,:] = q_mult(temp_q,q_delta)
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
    for d in range(Y.shape[0]):
        temp = q_mult(q_mult(q_inv( Y[d,:]),g), Y[d,:])
        Z_i[d,:] = temp[1:]

    #COmpute the mean
    z_k = np.mean(Z_i,axis = 0)
    #z_k = z_k/np.linalg.norm(z_k)
    
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
    return q_k,p_k

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
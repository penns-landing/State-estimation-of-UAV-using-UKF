#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:42:45 2019

@author: akarsh
"""

import numpy as np

def v2q(w):
    #start = time.time()
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
'''
def q_mult(p,q):
    temp1 = p[1:]
    temp2 = q[1:]
    a = p[0]*q[0] - (temp1[0]*temp2[0] + temp1[1]*temp2[1] + temp1[2]*temp2[2])
    a1 = p[1:]
    a2 = q[1:]
    cross_pdt = np.array([a1[1]*a2[2] - a1[2]*a2[1],a1[2]*a2[0] - a1[0]*a2[2], a1[0]*a2[1] - a1[1]*a2[0] ])
    b = p[0]*q[1:] + q[0]*p[1:] + cross_pdt
    return np.array([a,b[0],b[1],b[2]])
'''
'''
def q_mult(q1,q2):
    q1w = q1[0]
    q1x = q1[1]
    q1y = q1[2]
    q1z = q1[3]
    
    q2w = q2[0]
    q2x = q2[1]
    q2y = q2[2]
    q2z = q2[3]
    
    x =  q1x * q2w + q1y * q2z - q1z * q2y + q1w * q2x
    y = -q1x * q2z + q1y * q2w + q1z * q2x + q1w * q2y
    z =  q1x * q2y - q1y * q2x + q1z * q2w + q1w * q2z
    w = -q1x * q2x - q1y * q2y - q1z * q2z + q1w * q2w
    
    return np.array([w,x,y,z])
'''

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

def rot2euler(R):
    roll = -np.arcsin(R[1,2])
    pitch = -np.arctan2(-R[0,2]/np.cos(roll),R[2,2]/np.cos(roll))
    yaw = -np.arctan2(-R[1,0]/np.cos(roll),R[1,1]/np.cos(roll))
    
    return roll, pitch, yaw

def norm_quat(q):
    return np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])

def quat2rpy(qk):
    q0 = qk[0]
    q1 = qk[1]
    q2 = qk[2]
    q3 = qk[3]
    r = np.arctan(2*(q0*q1+q2*q3)/(1-2*(q1*q1 + q2*q2)))
    p = np.arcsin(2*(q0*q2-q3*q1))
    y = np.arctan(2*(q0*q3+q1*q2)/(1-2*(q2*q2 + q3*q3)))
    return np.array([r,p,y])

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

def norm_vec(q):
    return (np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]))

def q_exp(rs):
    exp_q = np.zeros(np.shape(rs))
    
    exp_q[0] = np.cos((norm_vec(rs[1:])))
    if norm_vec(rs[1:]) == 0:
        exp_q = np.zeros(3)
    else:
        exp_q[1:] = np.dot(rs[1:]/norm_vec(rs[1:]), np.sin(norm_vec(rs[1:])))
    return (np.exp(rs[0])*exp_q)

    
def quat_avg(q_k,Y):
    qt = q_k
    iterations = 1000
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
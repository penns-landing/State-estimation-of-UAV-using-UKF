#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:42:59 2019

@author: akarsh
"""

import numpy as np

def scalingas(imu):
    
    imu_data = np.transpose(imu['vals'])  
    accel = imu_data[:,:3]
    gyro = imu_data[:,3:]
    
    accel[:,:2] = -accel[:,:2]
    
    #g = [0,0,-9.81]
    
    
    
    #accel_new = accel - bias
    
    vref = 3300
    sens = 330.0
    acc_factor = vref/1023.0/sens
    acc_bias = accel[0] - np.array([0,0,1])/acc_factor
    accel_input = (accel - acc_bias)*acc_factor
   
    gyro_x = np.array(imu_data[:,4])
    gyro_y = np.array(imu_data[:,5])
    gyro_z = np.array(imu_data[:,3])
    gyro = [gyro_x, gyro_y, gyro_z]
    gyro = np.array(gyro)
    gyro = np.transpose(gyro)
    gyros_sens = 3.33
    gyro_factor = vref/1023/gyros_sens
    gyro_bias = gyro[0]
    
    gyro_input =( (np.array(gyro*gyro_factor))- gyro_bias*gyro_factor)*np.pi/180
    
    return accel_input, gyro_input
        
    
    
    
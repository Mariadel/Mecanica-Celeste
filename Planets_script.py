##############################################################################################
#                                                                                            # 
# Author: María del Mar Ruiz Martín                                                          # 
#                                                                                            #                                   
# Mecánica Celeste                                                                           #                                   
#                                                                                            # 
##############################################################################################
import numpy as np
import math
import random
import scipy.special as ss
from aux import *


#Globar variables with constants information
global planets
global epsilon
global a
global period
global Omega
global omega
global w
global colors

planets = np.array(["Mercurio","Venus","Tierra", "Marte","Júpiter","Saturno","Urano","Neptuno"])
a = np.array([0.387,0.723,1,1.524,5.203,9.546,19.2,30.09])
epsilon = np.array([0.206,0.007,0.017,0.093,0.048,0.056,0.047,0.009])   
period = np.array([87.97,224.7,365.26,686.98,4332.6,10759,30687,60784])
Omega = np.array([47.14,75.78,0,48.78,99.44,112.79,73.48,130.68])
omega = np.array([75.9,130.15,101.22,334.22,12.72,91.09,169.06,43.83])
w = omega - Omega
w = np.array([math.radians(i) for i in w])
colors = np.array(['rgb(100, 100, 100)','rgb(190, 100, 210)','rgb(0, 0, 250)','rgb(250, 0, 0)',
                    'rgb(77, 0, 77)','rgb(100, 30, 30)','rgb(260, 150, 0)','rgb(0, 150, 150)'])

class Planet:
    def __init__(self,name, a,epsilon, period,w, color):
        self.name = name
        self.a = a
        self.epsilon = epsilon
        self.period = period
        self.getMu()
        self.getAngularMomentum_constant()
        self.w = w
        self.rot = np.array([[math.cos(self.w),-math.sin(self.w)],[math.sin(self.w),math.cos(self.w)]])
        self.orbit = None
        self.color = color

    #Function to get Eccentric Anomaly with Newton Raphson method
    def getEccentricAnomaly(self,t):
        return getEccentricAnomaly_Newton(t,self.period, self.epsilon)

    #Function to get Eccentric Anomaly with Bessel's functions
    def getEccentricAnomaly_Bessel(self,t):
        return getEccentricAnomaly_Bessel(t,self.period,self.epsilon)

    #Function to get the position of the given planet at time t
    def getPosition(self,t): 
        u = self.getEccentricAnomaly(t)
        eps = self.epsilon
        return np.dot(self.rot,self.a*np.array([math.cos(u)-eps,math.sqrt(1-eps**2)*math.sin(u)]))
    
    #Function to get the distance of the given planet at time t
    def getDistance(self,t):
        return np.linalg.norm(self.getPosition(t))

    #Auxiliar function to get mu given the planet index
    def getMu(self):
        self.mu = 4*math.pi**2*self.a**3/self.period**2

    #Function to get the speed of a given planet at time t
    def getSpeed(self,t):
        u = self.getEccentricAnomaly(t)
        eps = self.epsilon
        return np.dot(self.rot,np.array([-math.sin(u),math.sqrt(1-eps**2)*math.cos(u)])*math.sqrt(self.mu/self.a)/(1-eps*math.cos(u)))

    #Function to get the speed modul of a given planet at time t
    def getSpeedModul(self,t):
        return np.linalg.norm(self.getSpeed(t))

    #Function to get angular Momentum
    def getAngularMomentum(self,t):
        return np.array([0,0,np.cross(self.getPosition(t),self.getSpeed(t))])
    
    #Function to get angular Momentum
    def getAngularMomentum_constant(self):
        self.c = np.array([0,0,math.sqrt(self.mu*self.a*(1-self.epsilon**2))])

    #Funtion to get theta derivative given theta
    def theta_deriv(self,t,theta):
        return np.linalg.norm(self.c)/(self.a**2*(1-self.epsilon**2)**2)*(1+self.epsilon*math.cos(theta))**2

    #Function to get Real Anomaly with Runge Kutta method
    def getRealAnomaly(self,t):
        return runge_Kutta(t,self.theta_deriv)

    #Function to get position using firts equation sistem
    def get_position_S1(self,t):
        theta = self.getRealAnomaly(t)
        return np.array([math.cos(theta+self.w),math.sin(theta+self.w)])*self.a*(1-self.epsilon**2)/(1+self.epsilon*math.cos(theta))

    #Function to get Energy of a planet
    def getEnergy(self,t):
        kineticEnergy = 0.5*self.getSpeedModul(t)**2
        potentialEnergy = -self.mu/self.getDistance(t)
        return kineticEnergy + potentialEnergy
    
    #Function to get Energy of a planet with the constant expression
    def getEnergy_constant(self):
        return -np.linalg.norm(self.c)**2/(2*self.a**2*(1-self.epsilon**2))

    #Function to get the Real anomaly of a given planet given eccentric anomaly
    def getRealAnomaly_2(self,u):
        eps = self.epsilon
        theta = math.acos((math.cos(u)-eps)/(1-eps*math.cos(u)))

        if u%(2*math.pi) > math.pi:
            theta =  2*math.pi-theta
        return theta    

    #Function to calculate the orbit of the planet
    def getOrbit(self):
        if self.orbit is None:
            N = 500
            self.orbit = np.asarray([self.getPosition(self.period*i/N) for i in range(N)]) 
        return self.orbit

    #Function to calculate N points of the planet's orbit
    def getPoints(self,N):
        return  np.asarray([self.getPosition(self.period*i/N) for i in range(N)]) 

    def __str__(self):
        return self.name


class SolarSistem:
    def __init__(self):
        self.planets = []
        for i in range(len(planets)):
            self.planets.append(Planet(planets[i],a[i],epsilon[i],period[i],w[i],colors[i]))  

# mybinder.org

#TODO: poner online
#TODO: hacer memoria
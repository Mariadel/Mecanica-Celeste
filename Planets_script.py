##############################################################################################
#                                                                                            # 
# Author: María del Mar Ruiz Martín                                                          # 
#                                                                                            #                                   
# Mecánica Celeste                                                                           #                                   
#                                                                                            # 
##############################################################################################

#Forma de uso: python3 Newton.py

import numpy as np
import math
import random
import scipy.special as ss
from matplotlib import pyplot as plt


#Globar variables with constants information
global planets
global epsilon
global a
global period
global w


#Todo: los planetas no deben dibujar el Sol

planets = np.array(["Mercurio","Venus","Tierra", "Marte","Júpiter","Saturno","Urano","Neptuno"])
a = np.array([0.387,0.723,1,1.524,5.203,9.546,19.2,30.09])
epsilon = np.array([0.206,0.007,0.017,0.093,0.048,0.056,0.047,0.009])




period = np.array([87.97,224.7,365.26,686.98,4332.6,10759,30687,60784])
w = [math.pi/2,1,0,1,1,1,1,1,]


#Auxiliar function to get u_{n+1}
def phi(u_ant,gi,eps):
    return ((-u_ant*math.cos(u_ant)+math.sin(u_ant))*eps+gi)/(1-eps*math.cos(u_ant))

#Function to get initial point to Newton-Raphson method
def getInitialPoint(u_0,t,period,gi,eps):
    u_1 = phi(u_0,gi,eps)

    if t%period < period/2:
        while(not(0 <= u_1 <= math.pi)):
            u_0 = random.uniform(0,math.pi)
            u_1 = phi(u_0,gi,eps)
    else:
        while(not(math.pi <= u_1 <= 2*math.pi)):
            u_0 = random.uniform(math.pi,2*math.pi)
            u_1 = phi(u_0,gi,eps)  

    return [u_0,u_1]

#Auxiliar function to get next Bessel term of the serie
def Bessel(n,eps,gi):
    return 2/n*ss.jv(n,n*eps)*math.sin(n*gi)  


class Planet:
    def __init__(self,name, a,epsilon, period,w):
        self.name = name
        self.a = a
        self.epsilon = epsilon
        self.period = period
        self.getMu()
        self.getAngularMomentum_constant()
        self.w = w
        self.rot = np.array([[math.cos(self.w),-math.sin(self.w)],[math.sin(self.w),math.cos(self.w)]])

    def getName(self) :
        return self.name

    #Implementation of Newton-Raphson method to get eccentric Anomaly
    def getEccentricAnomaly_Newton(self,t):
        u_0 = math.pi
        tol = 0.00001

        gi = 2*math.pi*t/self.period
        eps = self.epsilon

        u_ant, u_n = getInitialPoint(u_0,t,self.period,gi,eps)

        while(abs(u_n-u_ant) > tol):
            u_ant = u_n
            u_n = phi(u_ant,gi,eps)

        return u_n 

    #Function to get eccentric anomaly with Bessel's functions
    def getEccentricAnomaly_Bessel(self,t):
        tol = 0.00001
        n = 1
        ant = 0
        eps = self.epsilon
        gi = 2*math.pi*t/self.period
        val = Bessel(n,eps,gi) 
    
        while(abs(val-ant)>tol):
            ant = val
            n += 1
            val += Bessel(n,eps,gi)
    
        return gi + val

    #Function to get the position of the given planet at time t
    def getPosition(self,t): 
        t = t%self.period
        u = self.getEccentricAnomaly_Newton(t)
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
        u = self.getEccentricAnomaly_Newton(t)
        eps = self.epsilon
        return np.dot(self.rot,np.array([-math.sin(u),math.sqrt(1-eps**2)*math.cos(u)])/(self.a*math.sqrt(1-eps**2)*(1-eps*math.cos(u)))*math.sqrt(self.mu*(self.a*(1-eps**2))))

    #Function to get the speed modul of a given planet at time t
    def getSpeedModul(self,t):
        return np.linalg.norm(self.getSpeed(t))

    #Function to get angular Momentum
    def getAngularMomentum(self,t):
        return np.array([0,0,np.cross(self.getPosition(t),self.getSpeed(t))])
    
    #Function to get angular Momentum
    def getAngularMomentum_constant(self):
        self.c = np.array([0,0,math.sqrt(self.mu*self.a*(1-self.epsilon**2))])

    def f(self,t,theta):
        return np.linalg.norm(self.c)/(self.a**2*(1-self.epsilon**2)**2)*(1+self.epsilon*math.cos(theta))**2

    #Function to get real anomaly with Runge Kutta method
    def getRealAnomaly_rungeKutta(self,t):
        theta_n = 0
        t_i = 0
        h = 0.01
        N = int(t/h)
        for i in range(N):
            k1 = self.f(t_i,theta_n)
            k2 = self.f(t_i+0.5*h,theta_n+0.5*k1*h)
            k3 = self.f(t_i+0.5*h,theta_n+0.5*k2*h)
            k4 = self.f(t_i+h,theta_n+k3*h)
            theta_n += + h*(k1+2*k2+2*k3+k4)/6 
            t_i += h  
        return theta_n

    def getX(self,t):
        theta = self.getRealAnomaly_rungeKutta(t)
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
    def getRealAnomaly(self,u):
        eps = self.epsilon
        return math.acos((math.cos(u)-eps)/(1-eps*math.cos(u)))%(2*math.pi)

    #Function to plot the ellipse of a planet
    def plotEllipse(self):
        N = 500
        coordinates = np.asarray([self.getPosition(self.period*i/N) for i in range(N)])
        x = coordinates[:,0]
        y = coordinates[:,1]
        plt.scatter(x,y,c=(0,0,0),linewidths = 0.5)
        plt.show()
     
    #Function to plot the ellipse and the planet at time t
    def plotEllipse_withPlanet(self,t):
        N = 500
        #coordinates = np.asarray([self.getPosition(self.period*i/N) for i in range(N)])
        #x = coordinates[:,0]
        #y = coordinates[:,1]
        #plt.scatter(x,y,c=(0,0,0),linewidths = 0.1)
        N = 500
        coordinates = np.asarray([self.getX(self.period*i/N) for i in range(N)])
        x = coordinates[:,0]
        y = coordinates[:,1]
        plt.scatter(x,y,c=(0,1,0),linewidths = 0.1)
        position = self.getPosition(t)
        plt.scatter(position[0],position[1],c = (1,0.4,0),linewidths = 11)
        position = self.getX(t)
        plt.scatter(position[0],position[1],c = (0.4,0.4,0),linewidths = 11)
        position = self.getX_2(t)
        plt.scatter(position[0],position[1],c = (0,0.4,1),linewidths = 11)
        plt.scatter(0,0,c = (1,1,0),linewidths = 12)
        plt.show()
        return [x,y,position]

    def __str__(self):
        return self.name

class SolarSistem:
    def __init__(self):
        self.planets = []
        for i in range(len(planets)):
            self.planets.append(Planet(planets[i],a[i],epsilon[i],period[i],w[i]))  

    def plot(self,t):
        #self.planets[0].plotEllipse_withPlanet(t)
        print("**************++")
        #self.planets[1].plotEllipse_withPlanet(t)
        #print("**************++")
        #self.planets[2].plotEllipse_withPlanet(t)
        ##print("**************++")
        #self.planets[3].plotEllipse_withPlanet(t)
        ##print("**************++")
        #self.planets[4].plotEllipse_withPlanet(t)
        ##print("**************++")
        #self.planets[5].plotEllipse_withPlanet(t)
        ##print("**************++")
        #self.planets[6].plotEllipse_withPlanet(t)
        ##print("**************++")
        #self.planets[7].plotEllipse_withPlanet(t)
        #print("**************++")

        #for planet in self.planets:
        #    planet.plotEllipse_withPlanet(t)
        for i in range(4):
            self.planets[i].plotEllipse_withPlanet(t)
        plt.show()

def main():

    t = int(input("Introduzca tiempo en días: "))
    print("Los índices de los planetas son: ")
    for i in range(len(planets)):
        print(planets[i],":  ", i)
    planet_index = int(input("Introduzca el índice del planeta: "))

    t = t%period[planet_index]

    #print("El planeta se encuentra en la posición: ")
    #print(getPosition(t,planet_index))

    #print("El planeta se encuentra a una distancia del Sol de: ")
    #print(getDistance(t,planet_index))

    #print("La velocidad es:")
    #print(getSpeed(t, planet_index))

    #print("Su módulo es:")
    #print(getSpeedModul(t,planet_index))

    #ŦODO: resolver la anomalía real a partir del primer sistema

    #print("La anomalía real es:")
    #print(getRealAnomaly(t,planet_index))

    print("La energía del planeta es:")
    #print(getEnergy(t,planet_index))
    #print(getEnergy_2(planet_index))

    print("El momento angular es:")
    #print(getAngularMomentum(t, planet_index))
    #print(getAngularMomentum_2(planet_index))

    #eccentricAnomaly = float(input("Introduzca una anomalía excéntrica:"))
    #print("La anomalía real obtenida es:")
    #print(getRealAnomaly_2(eccentricAnomaly,planet_index))

    #print("El valor de la anomalía excéntrica es: ")
    #print(getEccentricAnomaly_Newton(t,planet_index))

    #print("El valor de la excentricidad obtenida con funciones de Bessel es:")
    #print(getEccentricAnomaly_Bessel(t,planet_index))

    #print("La diferencia obtenida es:")
    #print(abs(getEccentricAnomaly_Newton(t,planet_index)-getEccentricAnomaly_Bessel(t,planet_index)))

    print("La trayectoria del planeta es la siguiente: (ver gráfica)")
    #plotEllipse(planet_index)

    print("La posición del planeta en su trayectoria es: (ver gráfica)")
    #plotEllipse_withPlanet(t,planet_index)

    #print("RUNGE KUTTA")
    #print(rungeKutta(t,planet_index))

    #sistema_solar = SolarSistem()
    Tierra = Planet(planets[planet_index],a[planet_index],epsilon[planet_index],period[planet_index],w[planet_index])
    print(Tierra.getEccentricAnomaly_Newton(t))
    print(Tierra.getEccentricAnomaly_Bessel(t))
    print(abs(Tierra.getEccentricAnomaly_Newton(t) - Tierra.getEccentricAnomaly_Bessel(t)))
    print(Tierra.getPosition(t))
    print(Tierra.getDistance(t))
    print(Tierra.getSpeed(t))
    print(Tierra.getSpeedModul(t))
    print(Tierra.getAngularMomentum(t))
    print(Tierra.getRealAnomaly_rungeKutta(t))
    print(Tierra.getEnergy(t))
    print(Tierra.getEnergy_constant())
    print(Tierra.getAngularMomentum(t))
    print(Tierra.c)
    u = float(input("Introduzca una anomalía excéntrica:"))
    print(Tierra.getRealAnomaly(u))
    print(Tierra)
    #Tierra.plotEllipse_withPlanet(t)
    #sistema_solar.plot(t)




if __name__ == "__main__":
    main()


#main()


#Plotly

#plotly.offline, plotly.graphobjs-->py, go
#scattergl, plotly.plot([objetos])

#Funcion para imprimir información, 


import ipywidgets

#Dropdown, int text

#interact manual


#cells

#im

# Crear repo
# hacer los requerimientos (pip freeze > requirements.txt)
# mybinder.org
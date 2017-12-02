import numpy as np
import math
import random
import scipy.special as ss
import plotly.offline as py
import plotly.graph_objs as go

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


#Implementation of Newton-Raphson method to get eccentric Anomaly
def getEccentricAnomaly_Newton(t, period, eps):
    u_0 = math.pi
    tol = 0.00001

    gi = 2*math.pi*t/period
    u_ant, u_n = getInitialPoint(u_0,t,period,gi,eps)

    while(abs(u_n-u_ant) > tol):
        u_ant = u_n
        u_n = phi(u_ant,gi,eps)

    return u_n 


#Auxiliar function to get next Bessel term of the serie
def Bessel(n,eps,gi):
    return 2/n*ss.jv(n,n*eps)*math.sin(n*gi)  

#Function to get eccentric anomaly with Bessel's functions
def getEccentricAnomaly_Bessel(t, period, eps):
    tol = 0.00001
    n = 1
    ant = 0
    gi = 2*math.pi*t/period
    val = Bessel(n,eps,gi) 
    
    while(abs(val-ant)>tol):
        ant = val
        n += 1
        val += Bessel(n,eps,gi)
    
    return gi + val

#Function to get real anomaly with Runge Kutta method
def runge_Kutta(t,f):
    theta_n = 0
    t_i = 0
    h = 0.01
    N = int(t/h)
    for i in range(N):
        k1 = f(t_i,theta_n)
        k2 = f(t_i+0.5*h,theta_n+0.5*k1*h)
        k3 = f(t_i+0.5*h,theta_n+0.5*k2*h)
        k4 = f(t_i+h,theta_n+k3*h)
        theta_n += + h*(k1+2*k2+2*k3+k4)/6 
        t_i += h  
    return theta_n        


def display_orbit(planet, t):
    coordinates = planet.getOrbit()
    position = planet.getPosition(t)

    planet_pos = go.Scattergl(x=[position[0]], y=[position[1]], name=planet.name, mode='markers', marker=dict(size=25,color='rgb(250, 100, 0)',line=dict(width=1, color='rgb(250,100,0)')))
    sun = go.Scattergl(x=[0], y=[0],name='Sol',mode='markers',marker=dict(size=37,color='rgb(250, 250, 0)',line=dict(width=1, color='rgb(100,100,0)')))
    orbit = go.Scattergl(x=coordinates[:, 0], y=coordinates[:, 1], name="órbita de "+ planet.name,mode='markers', marker = dict(color = 'rgb(0,0,0)',size=3,line = dict(width = 0)))

    r = int(planet.a)*2
    layout = go.Layout(width=1000, height=700,xaxis=dict(anchor='y',range=[-r, r]),
            yaxis=dict(anchor='x',autorange=False,range=[-r, r],))

    data = [planet_pos,orbit,sun]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)   
    #py.plot(fig) 

def showInfo(planet,t,u):
    print("Información para ", planet.name," en el día ", t)    
    t = t%planet.period
    print("Posición: ", planet.getPosition(t))
    print("Distancia al Sol: ", planet.getDistance(t))
    print("Vector velocidad: ", planet.getSpeed(t))
    print("Módulo del vector velocidad: ", planet.getSpeedModul(t))
    print("Anomalía real: ", planet.getRealAnomaly(t))
    print("Energía total (expresión en función del tiempo): ", planet.getEnergy(t))
    print("Energía total (expresión constante): ", planet.getEnergy_constant())
    print("Momento angular (expresión en función del tiempo): ", planet.getAngularMomentum(t))
    print("Momento angular (expresión constante): ", planet.c)

    #u = float(input("Introduzca una anomalía excéntrica: "))
    print("Para una anomalía excéntrica de ", u, "la anomalía real es: ",planet.getRealAnomaly(u))

    print("Anomalía excéntrica (mediante el método de Newton-Raphson): ",planet.getEccentricAnomaly(t))
    print("Anomalía excéntrica (mediante funciones de Bessel): ",planet.getEccentricAnomaly_Bessel(t))
    print("La diferencia obtenida entre ambos métodos es: ",abs(planet.getEccentricAnomaly(t) - planet.getEccentricAnomaly_Bessel(t)))

    display_orbit(planet,t)
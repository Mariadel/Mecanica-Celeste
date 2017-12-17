import numpy as np
import math
import random
import scipy.special as ss
import plotly.offline as py
import plotly.graph_objs as go
from datetime import datetime as dt


global M
M=1.9891*10**30   #Kg



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

    ji = 2*math.pi*t/period
    nsv = int(ji/math.pi)  

    if nsv % 2 == 0:
        ji -= math.pi*nsv 
        ind = 1
        a = 0
    else:
        ji -= math.pi*nsv + math.pi
        ji = -ji
        ind = -1
        a = 1
        
    #u_ant, u_n = getInitialPoint(u_0,t,period,gi,eps)
    u_ant = 0
    u_n = math.pi

    while(abs(u_n-u_ant) > tol):
        u_ant = u_n
        u_n = phi(u_ant,ji,eps)

    u_n  = u_n * ind + (nsv+a)*math.pi
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


def getSunPosition(planet_pos, planet_mass, sun_mass):
    return -planet_mass/sun_mass*planet_pos


def translate(v,a,b,t):
    return v + a + b*t



#Function to plot the orbit of the planet
def display_orbit(planet, t, opt, N, a,b):
    position = translate(planet.getPosition(t),a,b,t)
    planet_pos = go.Scatter3d(x=[position[0]], y=[position[1]], z = [position[2]], name=planet.name, mode='markers', marker=dict(size=5,color=planet.color,line=dict(width=1, color=planet.color)))  

    if opt:
        coordinates = planet.getOrbit()
        coordinates = np.array([translate(i,a,b,t) for i in coordinates])
        orbit = go.Scatter3d(x=coordinates[:, 0], y=coordinates[:, 1],z = coordinates[:,2], name="órbita de "+ planet.name,mode='lines', marker = dict(color = 'rgb(0,0,0)',size=3,line = dict(width = 0)))
    else:
        coordinates = planet.getPoints(N)
        coordinates = np.array([translate(i,a,b,t) for i in coordinates])
        
        orbit = go.Scatter3d(x=coordinates[:, 0], y=coordinates[:, 1], z = coordinates[:,2],name="órbita de "+ planet.name,mode='markers', marker = dict(color = 'rgb(0,0,0)',size=1,line = dict(width = 0)))

    sun_poss = translate(getSunPosition(position, planet.mass, M),a,b,t)
    print("Posición del Sol: ("+str(sun_poss[0])+", "+str(sun_poss[1])+", "+str(sun_poss[2])+") ")
    sun_speed = -planet.getSpeed(t)*planet.mass/M
    print("Velocidad del Sol: ("+str(sun_speed[0])+", "+str(sun_speed[1])+", "+str(sun_speed[2])+") ")

    sun = go.Scatter3d(x = [sun_poss[0]], y = [sun_poss[1]], z = [sun_poss[2]],name='Sol',mode='markers',marker=dict(size=15,color='rgb(250, 250, 0)',line=dict(width=1, color='rgb(100,100,0)')))

    #r = int(planet.a)*2
    r = np.max(coordinates,axis = 0)
    r = np.max(r)
    r_2 = np.min(coordinates, axis = 0)
    r_2 = np.min(r_2)
    layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4, range = [r_2,r],),
                    yaxis = dict(
                        nticks=4, range = [r_2,r],),
                    zaxis = dict(
                        nticks=4, range = [r_2,r],),),
                    width=700,
                    margin=dict(
                    r=20, l=10,
                    b=10, t=10)
                  )


    r_3 = np.max(coordinates, axis = 0)
    r_4 = np.min(coordinates, axis = 0)
    r_5 = np.max(r_3 - r_4)
    dif = np.array([r_5 for i in range(3)]) - (r_3 - r_4)

    
    #layout = go.Layout(
    #                scene = dict(
    #                xaxis = dict(
    #                    nticks=4, range = [r_4[0] - dif[0]/2,r_4[0] + dif[0]/2],),
    #                yaxis = dict(
    #                    nticks=4, range = [r_4[1] - dif[1]/2,r_4[1] + dif[1]/2],),
    #                zaxis = dict(
    #                    nticks=4, range = [r_4[2] - dif[2]/2,r_4[2] + dif[2]/2],),),
    #                width=700,
    #                margin=dict(
    #                r=20, l=10,
    #                b=10, t=10)
    #              )
    #layout = go.Layout(width=1000, height=700,xaxis=dict(anchor='y',range=[-r, r]), yaxis=dict(anchor='x',autorange=True,range=[-r, r],))

    data = [sun, orbit,planet_pos]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig) 

def getC_m(m,M,x_0,x_1,y_1):
    alpha = m/(m+M)*x_0
    beta = m/(m+M)*x_1 + m/(m+M)*y_1 - alpha
    return alpha, beta

def getC_m_2(m,M,x_0,speed_0):
    alpha = m/(m+M)*x_0
    beta = m/(m+M)*speed_0
    return alpha, beta


#Function to display Info
def showInfo(planet,t,u,opt,N):
    sol_1 = np.array([0.001, 0.001, 0.001])
    alpha , beta = getC_m_2(planet.mass, M, planet.getPosition(0), planet.getPosition(1))
    print("Información para ", planet.name," en el día ", t)    
    #t = t%planet.period
    print("Posición: ", translate(planet.getPosition(t),alpha,beta,t))
    print("Distancia al Sol: ", planet.getDistance(t))
    print("Vector velocidad: ", planet.getSpeed(t))
    print("Módulo del vector velocidad: ", planet.getSpeedModul(t))
    print("Anomalía real: ", planet.getRealAnomaly(t))
    print("Energía total (expresión en función del tiempo): ", planet.getEnergy(t))
    print("Energía total (expresión constante): ", planet.getEnergy_constant())
    print("Momento angular (expresión en función del tiempo): ", planet.getAngularMomentum(t))
    print("Momento angular (expresión constante): ", planet.c)

    print("Para una anomalía excéntrica de ", u, "la anomalía real es: ",planet.getRealAnomaly_2(u))

    print("Anomalía excéntrica (mediante el método de Newton-Raphson): ",planet.getEccentricAnomaly(t))
    print("Anomalía excéntrica (mediante funciones de Bessel): ",planet.getEccentricAnomaly_Bessel(t))
    print("La diferencia obtenida entre ambos métodos es: ", abs(planet.getEccentricAnomaly(t) - planet.getEccentricAnomaly_Bessel(t)))
    print("El centro de masas es: ", alpha + beta*t)

    display_orbit(planet,t,opt,N,alpha,beta)








#Function to plot the orbit of the planet
def display_orbit_2(planet, t, opt, N):
    position = planet.getPosition(t)
    planet_pos = go.Scatter3d(x=[position[0]], y=[position[1]], z = [position[2]], name=planet.name, mode='markers', marker=dict(size=5,color=planet.color,line=dict(width=1, color=planet.color)))  

    if opt:
        coordinates = planet.getOrbit()
        orbit = go.Scatter3d(x=coordinates[:, 0], y=coordinates[:, 1],z = coordinates[:,2], name="órbita de "+ planet.name,mode='lines', marker = dict(color = 'rgb(0,0,0)',size=3,line = dict(width = 0)))
    else:
        coordinates = planet.getPoints(N)
        
        orbit = go.Scatter3d(x=coordinates[:, 0], y=coordinates[:, 1], z = coordinates[:,2],name="órbita de "+ planet.name,mode='markers', marker = dict(color = 'rgb(0,0,0)',size=1,line = dict(width = 0)))

    sun_poss = np.array([0,0,0])

    sun = go.Scatter3d(x = [sun_poss[0]], y = [sun_poss[1]], z = [sun_poss[2]],name='Sol',mode='markers',marker=dict(size=15,color='rgb(250, 250, 0)',line=dict(width=1, color='rgb(100,100,0)')))

    #r = int(planet.a)*2
    r = np.max(coordinates,axis = 0)
    r = np.max(r)
    r_2 = np.min(coordinates, axis = 0)
    r_2 = np.min(r_2)
    layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        nticks=4, range = [r_2,r],),
                    yaxis = dict(
                        nticks=4, range = [r_2,r],),
                    zaxis = dict(
                        nticks=4, range = [r_2,r],),),
                    width=700,
                    margin=dict(
                    r=20, l=10,
                    b=10, t=10)
                  )


    r_3 = np.max(coordinates, axis = 0)
    r_4 = np.min(coordinates, axis = 0)
    r_5 = np.max(r_3 - r_4)
    dif = np.array([r_5 for i in range(3)]) - (r_3 - r_4)

    
    #layout = go.Layout(
    #                scene = dict(
    #                xaxis = dict(
    #                    nticks=4, range = [r_4[0] - dif[0]/2,r_4[0] + dif[0]/2],),
    #                yaxis = dict(
    #                    nticks=4, range = [r_4[1] - dif[1]/2,r_4[1] + dif[1]/2],),
    #                zaxis = dict(
    #                    nticks=4, range = [r_4[2] - dif[2]/2,r_4[2] + dif[2]/2],),),
    #                width=700,
    #                margin=dict(
    #                r=20, l=10,
    #                b=10, t=10)
    #              )
    #layout = go.Layout(width=1000, height=700,xaxis=dict(anchor='y',range=[-r, r]), yaxis=dict(anchor='x',autorange=True,range=[-r, r],))

    data = [sun, orbit,planet_pos]
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig) 




#Function to display Info
def showInfo_2(planet,t,u,opt,N):
    alpha = np.array([0,0,0.1])
    beta = np.array([0,0.2,0.5])
    print("Información para ", planet.name," en el día ", t)    
    #t = t%planet.period
    print("Posición: ", planet.getPosition(t))
    print("Distancia al Sol: ", planet.getDistance(t))
    print("Vector velocidad: ", planet.getSpeed(t))
    print("Módulo del vector velocidad: ", planet.getSpeedModul(t))
    print("Anomalía real: ", planet.getRealAnomaly(t))
    print("Energía total (expresión en función del tiempo): ", planet.getEnergy(t))
    print("Energía total (expresión constante): ", planet.getEnergy_constant())
    print("Momento angular (expresión en función del tiempo): ", planet.getAngularMomentum(t))
    print("Momento angular (expresión constante): ", planet.c)

    print("Para una anomalía excéntrica de ", u, "la anomalía real es: ",planet.getRealAnomaly_2(u))

    print("Anomalía excéntrica (mediante el método de Newton-Raphson): ",planet.getEccentricAnomaly(t))
    print("Anomalía excéntrica (mediante funciones de Bessel): ",planet.getEccentricAnomaly_Bessel(t))
    print("La diferencia obtenida entre ambos métodos es: ", abs(planet.getEccentricAnomaly(t) - planet.getEccentricAnomaly_Bessel(t)))

    display_orbit_2(planet,t,opt,N)



t_ini = dt(1900,1,1)


#Function to change dato to time
def date2Time(date):
    formato = "%d/%m/%Y"
    return (dt.strptime(t,formato)-t_ini).days



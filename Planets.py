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
global mass
global G_0
global G
global M
global I

planets = np.array(["Mercurio","Venus","Tierra", "Marte","Júpiter","Saturno","Urano","Neptuno"])
a = np.array([0.387,0.723,1,1.524,5.203,9.546,19.2,30.09])
epsilon = np.array([0.206,0.007,0.017,0.093,0.048,0.056,0.047,0.009])   
period = np.array([87.97,224.7,365.26,686.98,4332.6,10759,30687,60194.84,])#60784])
mass = np.array([3.301, 48.67, 59.72,6.417,18990,5685,868.2,1024])*10**23
Omega = np.array([47.14,75.78,0,48.78,99.44,112.79,73.48,130.68])
omega = np.array([75.9,130.15,101.22,334.22,12.72,91.09,169.06,43.83])
M=1.9891*10**30   #Kg
G_0=6.6738431*(10**-11)  #m3/kg/s**2
G=G_0*(60*60*24)**2/(1.495978707*10**11)**3 #UA3/kg/día**2
mu=M*G
w = omega - Omega
w = np.array([math.radians(i) for i in w])
I = np.array([7, 3.59, 0, 1.85, 1.31, 2.5, 0.77, 1.78])
I = np.array([math.radians(i) for i in I])
Omega = np.array([math.radians(i) for i in Omega])
omega = np.array([math.radians(i) for i in omega])

colors = np.array(['rgb(100, 100, 100)','rgb(190, 100, 210)','rgb(0, 0, 250)','rgb(250, 0, 0)',
                    'rgb(77, 0, 77)','rgb(100, 30, 30)','rgb(260, 150, 0)','rgb(0, 150, 150)'])


class Planet:
    def __init__(self,name, a,epsilon,Omega, w, i,mass, sun_mass, G, color):
        self.name = name
        self.a = a
        self.epsilon = epsilon
        self.mass = mass 
        self.getMu(sun_mass, G)
        self.calculatePeriod()
        self.w = w
        self.i = i
        self.rot = np.array([[math.cos(self.w),-math.sin(self.w),0],[math.sin(self.w),math.cos(self.w),0],[0,0,1]])#El giro lo queremos hacer de \overline{\omega}
        self.rot = np.dot(np.array([[math.cos(i)+math.cos(Omega)**2*(1-math.cos(i)), math.cos(Omega)*math.sin(Omega)*(1-math.cos(i)),-math.sin(Omega)*math.sin(i)],
                    [math.cos(Omega)*math.sin(Omega)*(1-math.cos(i)),math.cos(i)+math.sin(Omega)**2*(1-math.cos(i)),-math.cos(Omega)*math.sin(i)],
                    [math.sin(Omega)*math.sin(i), math.cos(Omega)*math.sin(i),math.cos(i)]]),self.rot)
        self.getC_m()
        self.getAngularMomentum_constant()

        self.orbit = None
        self.color = color

    def getC_m(self):
        u_0 = self.getEccentricAnomaly(0)
        u_1 = self.getEccentricAnomaly(1)
        eps = self.epsilon
        x_0 = np.dot(self.rot,self.a*np.array([math.cos(u_0)-eps,math.sqrt(1-eps**2)*math.sin(u_0),0]))
        x_1 = np.dot(self.rot,self.a*np.array([math.cos(u_1)-eps,math.sqrt(1-eps**2)*math.sin(u_1),0]))
        a = getC_m_2(self.mass, M, x_0,x_1)
        self.alpha = a[0]# + 0.0001 
        self.beta = a[1]# + 0.0005 
        #print(a)


    #Function to calculate period
    def calculatePeriod(self):
        self.period = self.a**1.5*2*math.pi/math.sqrt(self.mu)

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
        x = np.dot(self.rot,self.a*np.array([math.cos(u)-eps,math.sqrt(1-eps**2)*math.sin(u),0]))
        return x
    
    #Function to get the distance of the given planet at time t
    def getDistance(self,t):
        x = self.getPosition(t)
        return np.linalg.norm(abs(x-getSunPosition(x,self.mass, M)))

    #Auxiliar function to get mu given the planet index
    def getMu(self, sun_mass, G):
        self.mu = G*sun_mass**3 /(sun_mass + self.mass)**2

    #Function to get the speed of a given planet at time t
    def getSpeed(self,t):
        u = self.getEccentricAnomaly(t)
        eps = self.epsilon
        return np.dot(self.rot,np.array([-math.sin(u),math.sqrt(1-eps**2)*math.cos(u),0])*math.sqrt(self.mu/self.a)/(1-eps*math.cos(u)))

    #Function to get the speed modul of a given planet at time t
    def getSpeedModul(self,t):
        return np.linalg.norm(self.getSpeed(t))

    #Function to get angular Momentum
    def getAngularMomentum(self,t):
        return np.cross(self.getPosition(t),self.getSpeed(t))
        #return np.dot(self.rot,np.cross(self.getPosition(t),self.getSpeed(t)))
    
    #Function to get angular Momentum
    def getAngularMomentum_constant(self):
        #self.c = np.dot(self.rot,np.array([0,0,math.sqrt(self.mu*self.a*(1-self.epsilon**2))]))
        self.c = np.dot(self.rot,np.array([0,0,math.sqrt(self.mu*self.a*(1-self.epsilon**2))]))

    #Funtion to get theta derivative given theta
    def theta_deriv(self,t,theta):
        return np.linalg.norm(self.c)/(self.a**2*(1-self.epsilon**2)**2)*(1+self.epsilon*math.cos(theta))**2

    #Function to get Real Anomaly with Runge Kutta method
    def getRealAnomaly(self,t):
        return runge_Kutta(t%self.period,self.theta_deriv)

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
        n = u//(math.pi*2)
        eps = self.epsilon
        theta = math.acos((math.cos(u)-eps)/(1-eps*math.cos(u)))

        if u%(2*math.pi) > math.pi:
            theta =  2*math.pi-theta
        return theta +  2*n*math.pi

    #Function to calculate the orbit of the planet
    def getOrbit(self):
        if self.orbit is None:
            N = 500
            self.orbit = np.asarray([self.getPosition(self.period*i/N) for i in range(N*5)]) 
        return self.orbit

    #Function to calculate N points of the planet's orbit
    def getPoints(self,N):
        return  np.asarray([self.getPosition(self.period*i/N) for i in range(N)]) 

    def __str__(self):
        return self.name


class SolarSistem:
    def __init__(self):
        self.planets = []
        self.mass = M
        for i in range(len(planets)):
            self.planets.append(Planet(planets[i],a[i],epsilon[i],Omega[i],omega[i],I[i],mass[i],self.mass,G,colors[i]))  


class Planet_2:
    def __init__(self,name, a,epsilon, period,Omega, w,i, color):
        self.name = name
        self.a = a
        self.epsilon = epsilon
        self.period = period
        self.getMu()
        self.w = w
        self.i = i
        self.rot = np.array([[math.cos(self.w),-math.sin(self.w),0],[math.sin(self.w),math.cos(self.w),0],[0,0,1]])
        self.rot = np.dot(np.array([[math.cos(i)+math.cos(Omega)**2*(1-math.cos(i)), math.cos(Omega)*math.sin(Omega)*(1-math.cos(i)),-math.sin(Omega)*math.sin(i)],
                    [math.cos(Omega)*math.sin(Omega)*(1-math.cos(i)),math.cos(i)+math.sin(Omega)**2*(1-math.cos(i)),-math.cos(Omega)*math.sin(i)],
                    [math.sin(Omega)*math.sin(i), math.cos(Omega)*math.sin(i),math.cos(i)]]),self.rot)
        self.getAngularMomentum_constant()
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
        return np.dot(self.rot,self.a*np.array([math.cos(u)-eps,math.sqrt(1-eps**2)*math.sin(u),0]))
    
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
        return np.dot(self.rot,np.array([-math.sin(u),math.sqrt(1-eps**2)*math.cos(u),0])*math.sqrt(self.mu/self.a)/(1-eps*math.cos(u)))

    #Function to get the speed modul of a given planet at time t
    def getSpeedModul(self,t):
        return np.linalg.norm(self.getSpeed(t))

    #Function to get angular Momentum
    def getAngularMomentum(self,t):
        return np.cross(self.getPosition(t),self.getSpeed(t))

        #return np.array([0,0,np.cross(self.getPosition(t),self.getSpeed(t))])
    
    #Function to get angular Momentum
    def getAngularMomentum_constant(self):
        self.c = np.dot(self.rot,np.array([0,0,math.sqrt(self.mu*self.a*(1-self.epsilon**2))]))

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


class SolarSistem_2:
    def __init__(self):
        self.planets = []
        for i in range(len(planets)):
            self.planets.append(Planet_2(planets[i],a[i],epsilon[i],period[i],Omega[i],omega[i],I[i], colors[i]))  

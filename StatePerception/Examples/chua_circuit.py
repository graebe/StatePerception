import numpy as np
import matplotlib.pyplot as plt

class chua_circuit():
    
    def __init__(self,x1_init=0.01,x2_init=0.01,x3_init=0.01,tau=0.02,alpha=10,beta=14):
        
        self.x1_init = x1_init
        self.x2_init = x2_init
        self.x3_init = x3_init
        
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
    
    def chua_circuit_step(self,x1_t,x2_t,x3_t,tau,alpha,beta):
        phi    = (x1_t**3)/16 - x1_t/6
        x1_tp1 = x1_t + tau*alpha*(x2_t-phi)
        x2_tp1 = (1-tau)*x2_t + tau*(x1_t+x3_t)
        x3_tp1 = x3_t - tau*beta*x2_t
        y_t    = x3_t/3
        v_t    = x1_t/2 + 4*x1_t**2/15
        return x1_tp1, x2_tp1, x3_tp1, y_t, v_t
        
    def simulate_chua_circuit(self,steps=1000):
        # First Step
        x1_t, x2_t, x3_t = self.x1_init, self.x2_init, self.x3_init
        _, _, _, y_t, v_t = self.chua_circuit_step(x1_t,x2_t,x3_t,self.tau,self.alpha,self.beta)
        
        # Initialize Reults
        sim_results = [[x1_t,x2_t,x3_t,y_t,v_t]]
        
        # simulate
        for t in range(steps-1):
            
            # Simulate Step
            x1_tp1, x2_tp1, x3_tp1, y_t, v_t = self.chua_circuit_step(x1_t,x2_t,x3_t,self.tau,self.alpha,self.beta)
            
            # Append Results
            sim_results.append([x1_t, x2_t, x3_t, y_t, v_t])
            
            # Next step
            x1_t, x2_t, x3_t = x1_tp1, x2_tp1, x3_tp1
        
        # Return Results
        return np.array(sim_results,dtype=np.float)
    
    def simulate_noisy_chua_circuit(self,steps=1000,noise_var=[0.01,0.5,0.1,0.1,0.05]):
        # Simulate System without noise
        sim_out = self.simulate_chua_circuit(steps=steps)
        
        # Add noise
        noise = np.random.randn(*sim_out.shape)
        noise_scaled = noise * noise_var
        noisy_sim_out = sim_out + noise_scaled
        
        # Return results
        return noisy_sim_out

    def plot_example(self):
        # Simulate System without and with noise
        out = self.simulate_chua_circuit(steps=10000)
        outn = self.simulate_noisy_chua_circuit(steps=10000)

        # Plot Chua
        plt.figure()
        ax1=plt.subplot(2,1,1)
        plt.plot(out[:,[0,1,2]])
        plt.legend(['x1','x2','x3'])
        plt.grid()
        plt.subplot(2,1,2,sharex=ax1)
        plt.plot(out[:,[3,4]])
        plt.legend(['y','z'])
        plt.grid()
        plt.xlabel('simulations step k [-]')
        
        # Plot Noisy Chua
        plt.figure()
        ax1=plt.subplot(2,1,1)
        plt.plot(outn[:,[0,1,2]])
        plt.legend(['x1','x2','x3'])
        plt.grid()
        plt.subplot(2,1,2,sharex=ax1)
        plt.plot(outn[:,[3,4]])
        plt.legend(['y','z'])
        plt.grid()
        plt.xlabel('simulations step k [-]')
    
    def simulate_example(self,steps=10000,noisy=False):
        if noisy is False:
            out = self.simulate_chua_circuit(steps=steps)
        else:
            out = self.simulate_noisy_chua_circuit(steps=steps)
        return out
        

#cc = chua_circuit()
#cc.plot_example()
#data = cc.simulate_example()
#X = data[:,[0,1,2]]
#Y = data[:,[3]]
#Z = data[:,[4]]

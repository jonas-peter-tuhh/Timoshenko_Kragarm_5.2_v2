"""
Created on Mon Aug 29 17:20:05 2022
@author: Jonas Peter
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np
import torch.autograd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(1, 5)
        self.hidden_layer2 = nn.Linear(5, 15)
        self.hidden_layer3 = nn.Linear(15, 50)
        self.hidden_layer4 = nn.Linear(50, 50)
        self.hidden_layer5 = nn.Linear(50, 50)
        self.hidden_layer6 = nn.Linear(50, 25)
        self.hidden_layer7 = nn.Linear(25, 15)
        self.output_layer = nn.Linear(15, 3)

    def forward(self, x):  # ,p,px):
        inputs = x
        layer1_out = torch.tanh(self.hidden_layer1(inputs))
        layer2_out = torch.tanh(self.hidden_layer2(layer1_out))
        layer3_out = torch.tanh(self.hidden_layer3(layer2_out))
        layer4_out = torch.tanh(self.hidden_layer4(layer3_out))
        layer5_out = torch.tanh(self.hidden_layer5(layer4_out))
        layer6_out = torch.tanh(self.hidden_layer6(layer5_out))
        layer7_out = torch.tanh(self.hidden_layer7(layer6_out))
        output = self.output_layer(layer7_out)
        return torch.unsqueeze(output.reshape(-1),1)

# Hyperparameter
learning_rate = 0.01


net = Net()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss()  # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

#Der Scheduler sorgt dafür, dass die Learning Rate auf einem Plateau mit dem factor multipliziert wird
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, verbose=True, factor= 0.8)

# Definition der Parameter des statischen Ersatzsystems

Lb = 5#float(input('Länge des Kragarms [m]: '))
EI = 20#float(input('EI des Balkens [10^6 kNcm²]: '))
A = 100#float(input('Querschnittsfläche des Balkens [cm²]: '))
G = 80#float(input('Schubmodul des Balkens [GPa]: '))
LFS = 1#int(input('Anzahl Streckenlasten: '))
K = 5 / 6  # float(input(' Schubkoeffizient '))
Ln = np.zeros(LFS)
Lq = np.zeros(LFS)
s = [None] * LFS

for i in range(LFS):
    # ODE als Loss-Funktion, Streckenlast
    Ln[i] = 0#float(input('Länge Einspannung bis Anfang der ' + str(i + 1) + '. Streckenlast [m]: '))
    Lq[i] = 5#float(input('Länge der ' + str(i + 1) + '. Streckenlast [m]: '))
    s[i] = "x"#input(str(i + 1) + '. Streckenlast eingeben: ')

def h(x, j):
    return eval(s[j])

#Netzwerk System 1
def f(x, net):
    net_out = net(x)
    phi = net_out[0::3]
    phi_x = torch.autograd.grad(phi, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(phi))[0]
    phi_xx = torch.autograd.grad(phi_x, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(phi))[0]
    phi_xxx = torch.autograd.grad(phi_xx, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(phi))[0]
    ode = 0
    for i in range(LFS):
        ode += phi_xxx + h(x - Ln[i], i)/EI * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode

def g(x, net):
    net_out = net(x)
    ode = 0
    gamma = net_out[1::3]
    gamma_x = torch.autograd.grad(gamma, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(gamma))[0]
    for i in range(LFS):
        ode += gamma_x-h(x - Ln[i], i)/(K*A*G) * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])
    return ode

def t(x,net):
    net_out = net(x)
    ode = 0
    phi = net_out[0::3]
    gamma = net_out[1::3]
    v = net_out[2::3]
    v_x = torch.autograd.grad(v, x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v))[0]
    ode += phi+gamma-v_x
    return ode

x = np.linspace(0, Lb, 1000)
qx = np.zeros(1000)
for i in range(LFS):
    qx = qx + (h(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1) - Ln[i], i).cpu().detach().numpy()).squeeze() * (x <= (Ln[i] + Lq[i])) * (x >= Ln[i])

Q0 = integrate.cumtrapz(qx, x, initial=0)
#Q0 = Q(0) = int(q(x)), über den ganzen Balken
qxx = qx * x
#M0 = M(0) = int(q(x)*x), über den ganzen Balken
M0 = integrate.cumtrapz(qxx, x, initial=0)
#Die nächsten Zeilen bis Iterationen geben nur die Biegelinie aus welche alle 10 Iterationen refreshed wird während des Lernens, man kann also den Lernprozess beobachten
y1 = net(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1)) #+ net_S(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device), 1))
fig = plt.figure()
plt.grid()
ax1 = fig.add_subplot()
ax1.set_xlim([0, Lb])
ax1.set_ylim([-30, 0])
#ax2.set_
net_out_plot = y1.cpu().detach().numpy()
line1, = ax1.plot(x, net_out_plot[2::3])

iterations = 5000
for epoch in range(iterations):
    optimizer.zero_grad()  # to make the gradients zero
    x_bc = np.linspace(0, Lb, 5000)
    # linspace x Vektor zwischen 0 und 1, 500 Einträge gleichmäßiger Abstand
    # Zufällige Werte zwischen 0 und 1
    pt_x_bc = torch.unsqueeze(Variable(torch.from_numpy(x_bc).float(), requires_grad=True).to(device), 1)
    # unsqueeze wegen Kompatibilität
    pt_zero = Variable(torch.from_numpy(np.zeros(1)).float(), requires_grad=False).to(device)

    x_collocation = np.random.uniform(low=0.0, high=Lb, size=(1000 * int(Lb), 1))
    all_zeros = np.zeros((1000 * int(Lb), 1))

    pt_x_collocation = Variable(torch.from_numpy(x_collocation).float(), requires_grad=True).to(device)
    pt_all_zeros = Variable(torch.from_numpy(all_zeros).float(), requires_grad=False).to(device)
    ode_phi = f(pt_x_collocation, net)
    ode_gamma = g(pt_x_collocation, net)
    ode_v = t(pt_x_collocation, net)

    # Randbedingungen
    net_bc_out = net(pt_x_bc)

    # Netzwerkausgabewerte berechnen
    phi = net_bc_out[0::3]
    gamma = net_bc_out[1::3]
    v = net_bc_out[2::3]

    #für phi:
    phi_x = torch.autograd.grad(phi, pt_x_bc, create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones_like(phi))[0]
    phi_xx = torch.autograd.grad(phi_x, pt_x_bc, create_graph=True, retain_graph=True,
                                grad_outputs=torch.ones_like(phi))[0]
    phi_bc1 = phi_x[0] + M0[-1]/EI
    phi_bc2 = phi[0]
    phi_bc3 = phi_xx[0] - Q0[-1]/EI
    phi_bc4 = phi_xx[-1]
    phi_bc5 = phi_x[-1]

    #für gamma:
    gamma_bc1 = gamma[0] + Q0[-1]/(K*A*G)
    gamma_bc2 = gamma[-1]

    #für v:
    v_bc1 = v[0]


    #Alle e's werden gegen 0-Vektor (pt_zero) optimiert.
    #BC Error
    mse_bc_phi = mse_cost_function(phi_bc1, pt_zero) + mse_cost_function(phi_bc2, pt_zero) + mse_cost_function(phi_bc3, pt_zero) + mse_cost_function(phi_bc4, pt_zero) + mse_cost_function(phi_bc5, pt_zero)
    mse_bc_gamma = mse_cost_function(gamma_bc1, pt_zero) + mse_cost_function(gamma_bc2, pt_zero)
    mse_bc_v = mse_cost_function(v_bc1, pt_zero)
    #ODE Error
    mse_ode_phi = mse_cost_function(ode_phi, pt_all_zeros)
    mse_ode_gamma = mse_cost_function(ode_gamma, pt_all_zeros)
    mse_ode_v = mse_cost_function(ode_v, pt_all_zeros)

    loss = mse_bc_phi + mse_ode_phi+mse_bc_gamma + mse_ode_gamma+mse_bc_v + mse_ode_v


    loss.backward()
    optimizer.step()
    scheduler.step(loss)
    with torch.autograd.no_grad():
        if epoch % 10 == 9:
            print(epoch, "Traning Loss:", loss.data)
            plt.grid()
            net_out  = net(torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=False).to(device),1)).cpu().detach().numpy()
            line1.set_ydata(net_out[2::3])
            fig.canvas.draw()
            fig.canvas.flush_events()
##
pt_x = torch.unsqueeze(Variable(torch.from_numpy(x).float(), requires_grad=True).to(device), 1)

pt_u_out = net(pt_x)
v_out = pt_u_out[2::3]
gamma_out = pt_u_out[1::3]

v_out_x = torch.autograd.grad(v_out, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_out))[0]
v_out_xx = torch.autograd.grad(v_out_x, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_out))[0]

gamma_out_x = torch.autograd.grad(gamma_out, pt_x, create_graph=True, retain_graph=True, grad_outputs=torch.ones_like(v_out))[0]

v_out_x = v_out_x.cpu().detach().numpy()
v_out_xx = v_out_xx.cpu().detach().numpy()

gamma_out_x = gamma_out_x.cpu().detach().numpy()

#Vges Kompatibilität Numpy Array
net_out_cpu = pt_u_out.cpu()
net_out = net_out_cpu.detach()
net_out_np = net_out.numpy()

v_out = net_out_np[2::3]
gamma_out = net_out_np[1::3]
fig = plt.figure()


plt.subplot(3, 2, 1)
plt.title('$v_{ges}$ Auslenkung')
plt.xlabel('')
plt.ylabel('[cm]')
#v = v
plt.plot(x, v_out)
#plt.plot(x, (-1/24 *x**4-np.sin(x)+(Q0[-1]-1)/6 *x**3 - M0[-1]/2 * x**2 +x)/EI - (0.5*x**2 - np.sin(x) - (Q0[-1]-1)*x)/(K*A*G))
plt.plot(x, (-1/120 *x**5 + 25/12 * x**3 - 41.67/2 * x**2)/EI + (1/6 * x**3 - 12.5*x)/(K*A*G))
plt.grid()

plt.subplot(3, 2, 3)
plt.title('$\phi$ Neigung')
plt.xlabel('')
plt.ylabel('$10^{-2}$')
#phi = v' - gamma
plt.plot(x, (v_out_x + gamma_out_x))
#plt.plot(x, (-1/6 *x**3-np.cos(x)+(Q0[-1]-1)/2 * x**2 - M0[-1]*x+1)/EI)
plt.plot(x, (-1/24 * x**4 +25/4 *x**2 - 41.67*x)/EI)
plt.grid()

plt.subplot(3, 2, 5)
plt.title('$\kappa$ Krümmung')
plt.xlabel('Meter')
plt.ylabel('$(10^{-4})$[1/cm]')
#kappa = v'' - gamma'
plt.plot(x, (v_out_xx - gamma_out_x))
#plt.plot(x, (-0.5*x**2+np.sin(x)+(Q0[-1]-1)*x-M0[-1])/EI)
plt.plot(x, (-1/6 * x**3 + 12.5 * x - 41.67)/EI)
plt.grid()

plt.subplot(3, 2, 2)
plt.title('Schubwinkel $\gamma$')
plt.xlabel('')
plt.ylabel('$(10^{-2})$')
#gamma = gamma
plt.plot(x, (gamma_out))
plt.plot(x, (0.5*x**2 - 12.5)/(K*A*G))
#plt.plot(x, (x-np.cos(x)-(Q0[-1]-1))/(K*A*G))
plt.grid()


#plt.subplot(3, 2, 4)
#plt.title('q(x) Test')
#plt.xlabel('')
#plt.ylabel('$kN$')
#plt.plot(x, (-w_xxx))
#plt.plot(x, x)
#plt.plot(x, 1+np.sin(x))
#plt.grid()



plt.show()
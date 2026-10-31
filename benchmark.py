import torch
import torch.nn as nn
import torch.optim as optim
from time import perf_counter
from torchdiffeq import odeint
from statistics import mean

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class Duffing(nn.Module): # a equação de Duffing

    def __init__(self, alpha, beta, delta, gamma, omega):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.omega = omega

    def forward(self, t, y):

        x = y[..., 0]
        v = y[..., 1]

        dxdt = v

        dvdt = (
            -self.delta * v
            -self.alpha * x
            -self.beta * x**3
            +self.gamma * torch.cos(self.omega * t)
        )

        return torch.stack([dxdt, dvdt], dim=-1)

class ODEFunc(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(ODEFunc, self).__init__()
        layers = []
        in_dim = 3
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, t, state):
        x = state[..., 0]
        v = state[..., 1]
        t = torch.ones_like(x) * t
        inp = torch.stack([x, v, t], dim=-1)
        dxdt = v
        dvdt = self.net(inp).squeeze(-1) # chat sugeriu squeeze
        return torch.stack([dxdt, dvdt], dim=-1)

# constantes
gamma = 0.3
alpha = -1.0
beta = 1.0
omega = 1.2
delta = 0.2
# valores iniciais
x0 = 0
v0 = 0.2
# y0 = torch.stack([torch.tensor([x0]), torch.tensor([v0])])
y0 = torch.tensor([x0,v0])

# treinamento
intervalo = [0,100]
N = 100 # nº de pontos de treinamento
fxy = Duffing(alpha, beta, delta, gamma, omega)
h = (intervalo[1] - intervalo[0])/N
t = torch.linspace(intervalo[0], intervalo[1], N)
sol = odeint(fxy, y0, t)
x = sol[:,0]
v = sol[:,1]

# passando pro cuda
y0 = y0.to(device)
x = x.to(device)
v = v.to(device)
t = t.to(device)

best_params = {
     "num_layers": 3,
     "dim": 115,
     "lr": 0.002066949712723738
}

n_epochs = 500
previsoes_x = []
previsoes_v = []
erros = []
hidden_dims = best_params["dim"]
num_layers = best_params["num_layers"]
lr = best_params["lr"]
metodos = [
    "euler",
    "midpoint",
    "heun3",
    "rk4",
    "dopri5",
    "dopri8",
]

def marcar_tempo(device):
  if device == "cuda":
    torch.cuda.synchronize()
  return perf_counter()

def benchmark(metodo, best_params, n_trials):

  tempos = []
  for trial in n_trials:
    func = ODEFunc(hidden_dim=best_params["dim"],
                  num_layers=best_params["num_layers"]).to(device)
    optimizer = optim.Adam(func.parameters(),
                          lr=best_params["lr"])

    tempo_total_inicio = marcar_tempo(device)

    for epoch in range(n_epochs):
      # inicio_epoca = marcar_tempo(device)

      optimizer.zero_grad()
      pred = odeint(func, y0, t, method=metodo)
      pred_x = pred[:,0]
      pred_v = pred[:,1]

      loss_x = nn.MSELoss()(pred_x, x)
      loss_v = nn.MSELoss()(pred_v, v)
      loss = loss_x + loss_v
      loss.backward()
      optimizer.step()

    tempo_total_fim = marcar_tempo(device)
    tempo_total_treinamento = tempo_total_fim - tempo_total_inicio
    tempos.append(tempo_total_fim - tempo_total_inicio)

    print(f"Treinamento {trial} de {metodo}: {tempo_total_treinamento:.6f}s")
  media = mean(tempos)
  print(f"Concluído: média {media:.6f}s")
  return tempos, media

# main
n_trials = 20
tempos_metodo = []
media_metodos = []
for metodo in metodos:
   tempos, media = benchmark(metodo, best_params, n_trials)
   tempos_metodo.append(tempos)
   media_metodos.append(media)

print(tempos_metodo)
print(media_metodos)
import torch
from torch import nn
from torch.distributions import MultivariateNormal
from torch.nn import functional as F

class FCN(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super().__init__()
        # actor

        self.fc1 = nn.Linear(state_dim, n_latent_var)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(n_latent_var, n_latent_var)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(n_latent_var, 5)
        self.drop3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(5, n_latent_var)
        self.drop4 = nn.Dropout(0.5)
        self.fc5 = nn.Linear(n_latent_var, n_latent_var * 2)
        self.drop5 = nn.Dropout(0.5)
        self.fc6 = nn.Linear(n_latent_var * 2, n_latent_var)
        self.drop6 = nn.Dropout(0.5)
        self.action_layer = nn.Linear(n_latent_var, action_dim)
        self.xavier_init()

    def xavier_init(self):
        nn.init.xavier_uniform_(self.fc1.weight, 5/3)
        nn.init.xavier_uniform_(self.fc2.weight, 5/3)
        nn.init.xavier_uniform_(self.fc3.weight, 5/3)
        nn.init.xavier_uniform_(self.fc4.weight, 5/3)
        nn.init.xavier_uniform_(self.fc5.weight, 5/3)
        nn.init.xavier_uniform_(self.fc6.weight, 5/3)
        nn.init.uniform_(self.action_layer.weight, -0.1, 0.1)
        # nn.init.uniform_(self.value_layer.weight, -0.1, 0.1)


    def forward(self, state):
        x = self.fullyLayer(state)
        pred = self.action_layer(x)
        #print(pred.shape)
        a = pred[0]
        b = pred[1]**2
        c = pred[2]**3
        d = pred[3]**4
        e = pred[4]**5

        #print(a.shape, b.shape)
        f = torch.sub(b, a)#torch.cat((a, b), dim=-1)
        g = torch.sub(c, d)#torch.cat((c, d), dim=-1)
        h = torch.add(f, g)#torch.cat((f, g), dim=-1)
        #print(h.shape)
        return torch.sub(h, e)#torch.cat((h, e), dim=-1)

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)

    def fullyLayer(self, x):
        x = self.drop1(F.tanh(self.fc1(x)))
        x = self.drop2(F.tanh(self.fc2(x)))
        x = self.drop3(F.tanh(self.fc3(x)))
        return self.drop4(F.tanh(self.fc4(x))) #x = self.drop4(F.tanh(self.fc4(x)))
        #x = self.drop5(F.tanh(self.fc5(x)))
        #return self.drop6(F.tanh(self.fc6(x)))
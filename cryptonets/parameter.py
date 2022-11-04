import torch
import model
import pickle

net = model.CryptoNets()
weights = torch.load('cryptonets.pth')
net.load_state_dict(weights)

params = {}
i = 1
for _, prm in enumerate(net.state_dict()):
    params[prm] = net.state_dict()[prm].detach().numpy()
    print(str(i) + "  " + prm)
    print("shape: " + str(params[prm].shape))
    i += 1

with open('cryptonets_param.pkl', 'wb') as f:
    pickle.dump(params, f, -1)
import torch
import torch.nn as nn

class Dreamer(nn.Module):
    def __init__(
            self,
            input_dims : int,
            output_dims : int, 
            action_space : int,
            ):
        super(Dreamer, self).__init__()
        self.world_model = nn.Sequential(
            # Fill in something here for a model the Sequential is just a stand-in
        )

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.action_space = action_space

        self.actor = DenseConnections(
            input_dims,
            action_space,
            action_model = True
        )

        self.critic = DenseConnections(
            input_dims,
            output_dims,
            action_model = False
        )

    def find_lambda_returns(
            self,
            ):
        
        return
    
    def find_loss(self):

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr =8e-5)

        actor_loss = -self.find_lambda_returns()
        self.actor_optimizer.zero_grad()
        actor_loss.backwards()
        self.actor_optimizer.step()



        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=8e-5)
        critic_distribution = 
        critic_loss = 
        # Use KL-Loss instead of MSE for some reason
        # Actor loss is the negative of the predicted returns
        # Value loss is the "KL" loss between the predicted value and the actual value 
        return # Return the world model loss, actor loss, critic loss
    
    def train(self):

        return
    
    def find_predicted_returns(
        self,
    ):
        return

class DenseConnections(nn.Module):
    def __init__(self, input_dims : int, output_dims : int, mid_dims :int = 300, action_model = False):
        super(DenseConnections, self).__init__()
        self.l1 = nn.Linear(input_dims, mid_dims)
        self.l2 = nn.Linear(mid_dims, mid_dims)
        self.action_model = action_model

        if self.action_model:
            self.std = nn.Linear(mid_dims, output_dims)
            self.mean = nn.Linear(mid_dims, output_dims)
        else:
            self.l3 = nn.Linear(mid_dims, output_dims)


    def forward(self, input : torch.Tensor):
        x = nn.ELU(self.l1(input))
        x = nn.ELU(self.l2(x))
        if self.action_model:
            std = self.std(x)
            mean = self.mean(x)

            eps = torch.randn_like(std)

            x = nn.Tanh(mean + std * eps)
        else:
            x = self.l3(x)

        return x
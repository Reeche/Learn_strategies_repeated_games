import unittest
import torch
import numpy as np

from actor_critic import Actor, Critic

# todo: need to tidy up the tests
class TestModel(unittest.TestCase):

    def test_critic(self):
        Actor_obj = Actor(1, 16, 4)
        Critic_obj = Critic(4, 16, 1)
        # critic_optimizer = optim.SGD(Critic_obj.parameters(), lr=C_learning_rate)

        y = Actor_obj.forward(torch.FloatTensor([1]))
        # Forward Propagation
        y_pred = Critic_obj.forward(y)
        self.assertTrue(len(y_pred) == 1)

        # Compute and print loss
    #     loss = criterion(y_pred, (torch.tensor([[4]], dtype=torch.float)))
    #
    #     # Zero the gradients
    #     critic_optimizer.zero_grad()
    #
    #     # perform a backward pass (backpropagation)
    #     loss.backward(retain_graph=True)
    #
    #     # Update the parameters
    #     critic_optimizer.step()
        # #for param_tensor in Critic_obj.state_dict():
        # #    print(param_tensor, "\t", Critic_obj.state_dict()[param_tensor])
        # torch.save(Critic_obj.state_dict(), "checkpoint/test.pth")

    def test_actor(self):
        Actor_obj = Actor(1, 16, 4)
        Critic_obj = Critic(4, 16, 1)
        # actor_optimizer = optim.SGD(Actor_obj.parameters(), lr=0.1, momentum=0.5)

        # Forward Propagation
        y = Actor_obj.forward(torch.FloatTensor([1]))
        self.assertTrue(len(y) == 4)
        #y_pred = torch.sum(y)
        #y_pred = Critic_obj.forward(y)


        # # Compute and print loss
        # loss = criterion(y_pred, (torch.tensor([[5]], dtype=torch.float)))
        #
        # # Zero the gradients
        # actor_optimizer.zero_grad()
        #
        # # perform a backward pass (backpropagation)
        # loss.backward(retain_graph=True)
        #
        # # Update the parameters
        # actor_optimizer.step()



# def test_load_trained_model():
#     Critic_obj = Critic(input_size_C, hidden_size, output_size_C)
#     model.load_state_dict(torch.load("checkpoint/test.pth"))
#     model.eval()
#     print("------------------------")
#     for param_tensor in model.state_dict():
#         print(param_tensor, "\t", model.state_dict()[param_tensor])
#
# #test_load_trained_model()
#
# #input_size_A, hidden_size, output_size_A, input_size_C, output_size_C = 10, 128, 4, 4, 1
# #criterion = nn.MSELoss()

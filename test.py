def test_critic(z):
    Actor_obj = Actor(input_size_A, hidden_size, output_size_A)
    Critic_obj = Critic(input_size_C, hidden_size, output_size_C)
    critic_optimizer = optim.SGD(Critic_obj.parameters(), lr=C_learning_rate)

    y = Actor_obj.forward(z)
    for _ in range(100):
        # Forward Propagation
        y_pred = Critic_obj.forward(y)
        #print("y_pred", y_pred)

        # Compute and print loss
        loss = criterion(y_pred, (torch.tensor([[4]], dtype=torch.float)))

        # Zero the gradients
        critic_optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward(retain_graph=True)

        # Update the parameters
        critic_optimizer.step()
    #for param_tensor in Critic_obj.state_dict():
    #    print(param_tensor, "\t", Critic_obj.state_dict()[param_tensor])
    torch.save(Critic_obj.state_dict(), "checkpoint/test.pth")

#test_critic(get_input())


def test_load_trained_model():
    Critic_obj = Critic(input_size_C, hidden_size, output_size_C)
    model.load_state_dict(torch.load("checkpoint/test.pth"))
    model.eval()
    print("------------------------")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor])

#test_load_trained_model()

#input_size_A, hidden_size, output_size_A, input_size_C, output_size_C = 10, 128, 4, 4, 1
#criterion = nn.MSELoss()

def test_actor(z):
    Actor_obj = Actor(input_size_A, hidden_size, output_size_A)
    Critic_obj = Critic(input_size_C, hidden_size, output_size_C)
    actor_optimizer = optim.SGD(Actor_obj.parameters(), lr=0.1, momentum=0.5)

    for _ in range(100):
        # Forward Propagation
        y = Actor_obj.forward(z)
        #y_pred = torch.sum(y)
        y_pred = Critic_obj.forward(y)
        print("y_pred", y_pred)

        # Compute and print loss
        loss = criterion(y_pred, (torch.tensor([[5]], dtype=torch.float)))

        # Zero the gradients
        actor_optimizer.zero_grad()

        # perform a backward pass (backpropagation)
        loss.backward(retain_graph=True)

        # Update the parameters
        actor_optimizer.step()

#test_actor(get_input())
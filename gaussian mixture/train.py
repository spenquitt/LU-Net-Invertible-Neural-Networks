from functions import log_likelihood
from tqdm import tqdm
from visuals import add_batch_loss
from torch import nn
from model import register_activation_hooks

"""training of gaussian data"""
def training_routine_gaussian(model, device, train_loader, optimizer, epoch, batch_size):
    model.train()
    train_loss = 0
    i = 1
    
    """outputs of L-layer are needed for the loss function"""
    layers = []
    for j in range(len(model.intermediate_lu_blocks)):
        if j % 2 != 0:
            layers.append("intermediate_lu_blocks.{}".format(j))
    layers.append("final_lu_block.1")
    
    for k in tqdm(range(int(len(train_loader) / batch_size))):
        saved_layers = register_activation_hooks(model, layers_to_save=layers)
        inputs = train_loader[k * batch_size : k * batch_size + batch_size]
        inputs = inputs.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs)
        loss = log_likelihood(output, model, saved_layers)
        train_loss += loss
        _, layer_size = output.shape
        add_batch_loss(epoch, i, loss.item() / (batch_size * layer_size))
        print("batch loss: " + str(loss.item() / (batch_size * layer_size)))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=1)
        optimizer.step()
        i += 1

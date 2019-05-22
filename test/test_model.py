import torch
from torch import nn, optim

def train_model(model, train_features, train_labels, v_f, v_l, epochs = 4, batch_size =  512, learning_rate = 0.001, print_every = 100):
  """
  Train your model with dropout. Make sure to clip your gradients.
  Print the training loss, validation loss, and validation accuracy for every 100 steps.
  """ 
  #print_every = 1#100 # for small data !!!!!!!!!!!!!!!!!!!!!!! TODO Recovery

  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate)
  model.train()

  val_losses = []
  accuracy = []

  for epoch in range(epochs):
      print('Starting epoch {}'.format(epoch + 1))

      steps = 0
      for text_batch, labels in load_data(
              train_features, train_labels, batch_size=batch_size, sequence_length=20, shuffle=True):
          steps += 1
          hidden = model.init_hidden(labels.shape[0]) #pass

          # Set Device
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          text_batch, labels = text_batch.to(device), labels.to(device)
          for each in hidden:
              each.to(device)

          # Train Model
          hidden = tuple([each.data for each in hidden])
          model.zero_grad()
          output, hidden = model(text_batch, hidden)
          loss = criterion(output.squeeze(), labels)
          loss.backward()
          clip = 5
          nn.utils.clip_grad_norm_(model.parameters(), clip)
          optimizer.step()

          # Accumulate loss
          val_losses.append(loss.item())

          correct_count = 0.0
          if steps % print_every == 0:
              model.eval()

              # Get validation loss
              val_h = model.init_hidden(batch_size)
              val_losses = []
              model.eval()
              for inputs, labels in load_data(v_f, v_l, batch_size=batch_size, sequence_length=20, shuffle=True):
                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                if(torch.cuda.is_available()):
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = model(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())
              
              # Calculate accuracy
              ps = torch.exp(output)
              top_p, top_class = ps.topk(1, dim=1)

              correct_count += torch.sum(top_class.squeeze()== labels)

              label_count = len(labels)
              #print(label_count)
              #print(correct_count.cpu().numpy())

              correct_count_num = correct_count.cpu().numpy()
              #print(correct_count_num/label_count)

              accuracy.append(correct_count_num/label_count)

              # Print metrics
              print("Epoch: {}/{}...".format(epoch+1, epochs),
                   "Step: {}...".format(steps),
                   "Collect Count: {}".format(correct_count),
                   "Total Count: {}".format(len(labels)),
                   "Loss: {:.6f}...".format(loss.item()),
                   "Loss Avg: {:.6f}".format(np.mean(val_losses)),
                   #"Accuracy: {:.2f}".format((100*correct_count_num/len(labels))),
                   "Accuracy: {:.2f}".format(correct_count_num/label_count),
                   "Accuracy Avg: {:.2f}".format(np.mean(accuracy))
                   )

              model.train()
  print("Last Loss Avg: {:.6f}".format(np.mean(val_losses)))
  print("Last Accuracy Avg: {:.2f}".format(np.mean(accuracy)))

  #torch.save(model.state_dict(), "./model.pth")
  torch.save(model, "./model.torch")

####################
from sentiment.model import load_data
"""
def test_model(model, train_features, train_labels, epochs = 4, batch_size =  512, learning_rate = 0.001, print_every = 100):
  #Train your model with dropout. Make sure to clip your gradients.
  #Print the training loss, validation loss, and validation accuracy for every 100 steps.
  collect = 0
  total = 0
  for text_batch, labels in load_data(
          train_features, train_labels, batch_size=batch_size, sequence_length=20, shuffle=True):

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_batch, labels = text_batch.to(device), labels.to(device)

    # Test Model
    with torch.no_grad():
      outputs = model(text_batch)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Accuracy: {:.2f} %%'.format(100 * float(correct/total)))

"""
  

# Get test data loss and accuracy
def test_model(model, train_features, train_labels, epochs = 4, batch_size =  512):
  test_losses = [] # track loss
  num_correct = 0

  # init hidden state
  h = model.init_hidden(batch_size)

  criterion = nn.NLLLoss()

  model.eval()
  # iterate over test data
  for inputs, labels in load_data(
          train_features, train_labels, batch_size=batch_size, sequence_length=20, shuffle=True):

      # Creating new variables for the hidden state, otherwise
      # we'd backprop through the entire training history
      h = tuple([each.data for each in h])

      if(torch.cuda.is_available()):
          inputs, labels = inputs.cuda(), labels.cuda()

      # get predicted outputs
      output, h = model(inputs, h)

      # calculate loss
      test_loss = criterion(output.squeeze(), labels)#.float())
      test_losses.append(test_loss.item())

      # convert output probabilities to predicted class (0 or 1)
      #pred = torch.round(output.squeeze())  # rounds to the nearest integer

      # compare predictions to true label
      #correct_tensor = pred.eq(labels.view_as(pred))#.float().view_as(pred))
      #correct = np.squeeze(correct_tensor.numpy()) if not torch.cuda.is_available() else np.squeeze(correct_tensor.cpu().numpy())
      #num_correct += np.sum(correct)


  # -- stats! -- ##
  # avg test loss
  print("Test loss: {:.3f}".format(np.mean(test_losses)))

  # accuracy over all test data
  #test_acc = num_correct/len(test_loader.dataset)
  #print("Test accuracy: {:.3f}".format(test_acc))
  #return test_acc

train_features, train_labels, tf, tl, vf, vl = split_data(token_ids, sentiments, vocab)
test_model(model, train_features, train_labels)

import cdsw
cdsw.track_metric("Accuracy",test_acc)


model_filename = "model.torch"
vocab_filename = "vocab.pickle"
cdsw.track_file(model_filename)
cdsw.track_file(vocab_filename)


cdsw.track_metric("Loss",)


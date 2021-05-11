import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        ''' Initialize the layers of this model.'''
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        # embedding layer that turns words into a vector of a specified size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        # the linear layer that maps the hidden state output dimension 
        # to the number of tags we want as output, tagset_size (in this case this is 3 tags)
        self.fc1 = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        cap_embedding = self.embedding(captions[:,:-1])
        #Add the features a another part of the sequence length so caption_features.shape == (10, 12, 256)
        features = features.unsqueeze(dim=1)
        start_vec = torch.cat((features, cap_embedding),dim=1)
        output, hidden = self.lstm(start_vec)
        return self.fc1(output)


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        caption = []
        
        for i in range(max_len):
            
            output, states = self.lstm(inputs, states) 
            score = self.fc1(output.squeeze(1))
            val, reduced = score.max(1)
            caption.append(reduced.item())
            #for next iteration
            inputs = self.embedding(reduced)
            inputs = inputs.unsqueeze(1) 
        return caption

# class FeedbackClassifier(object):
#     def train(self, train_loader):
#         raise NotImplementedError

#     @torch.no_grad()
#     def evaluate(self, val_loader):
#         raise NotImplementedError

#     def predict(self, text):
#         raise NotImplementedError


# class FastTextClassifier(FeedbackClassifier):
#     def __init__(self, opt):
#         self.opt = opt
#         self.model = FastText(
#             vocab_size=self.opt['vocab_size'],
#             embedding_dim=self.opt['embedding_dim'],
#             output_dim=self.opt['output_dim']
#         )
#         self.nlp = spacy.load('en')
        
#         # Prepare optimizer/loss
#         self.optimizer = optim.Adam(self.model.parameters())
#         self.criterion = nn.BCEWithLogitsLoss()
        
#         # Move to gpu (if applicable)
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         self.model = self.model.to(self.device)
#         self.criterion = self.criterion.to(self.device)
    
#     def train(self, train_loader):
#         # Build vocab
#         self.TEXT = data.Field(tokenize='spacy', 
#             preprocessing=self._generate_bigrams)
#         self.TEXT.build_vocab(train_loader, max_size=self.opt['vocab_size'], 
#             vectors="glove.6B.100d")

#         # LABEL = data.LabelField(tensor_type=torch.FloatTensor)
#         # LABEL.build_vocab(train_loader)

#         # Load pretrained embeddings
#         pretrained_embeddings = self.TEXT.vocab.vectors
#         self.model.embedding.weight.data.copy_(pretrained_embeddings)

#         # Train loop
#         self.model.train()

#         for epoch in range(opt['num_epochs']):
#             epoch_loss = 0
#             for batch in train_loader:
#                 self.optimizer.zero_grad()

#                 predictions = self.model(batch.text).squeeze(1)
#                 loss = self.criterion(predictions, batch.label)
#                 loss.backward()
#                 self.optimizer.step()
                
#                 epoch_loss += loss.item()
#         return epoch_loss / len(train_loader)
    
#     @torch.no_grad()
#     def evaluate(self, val_loader):
#         epoch_loss = 0
#         self.model.eval()

#         for batch in val_loader:
#             predictions = self.model(batch.text).squeeze(1)
#             loss = self.criterion(predictions, batch.label)            
#             epoch_loss += loss.item()
#         return epoch_loss / len(val_loader)

#     def predict(self, text):
#         tokenized = [tok.text for tok in self.nlp.tokenizer(text)]
#         indexed = [self.TEXT.vocab.stoi[t] for t in tokenized]
#         tensor = torch.LongTensor(indexed).to(self.device)
#         tensor = tensor.unsqueeze(1)
#         prediction = F.sigmoid(self.model(tensor))
#         return prediction.item()        


#     @staticmethod
#     def _generate_bigrams(x):
#         """Appends bigrams to the end of an ordered token list"""
#         bigrams = set(zip(*[x[i:] for i in range(2)]))
#         for bigram in bigrams:
#             x.append(' '.join(bigram))
#         return x


# class FastText(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, output_dim):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.fc = nn.Linear(embedding_dim, output_dim)    

#     def forward(self, x):
#         # x: [sent_len, batch_size]
#         embedded = self.embedding(x)
#         # embedded: [sent_len, batch_size, emedding_dim]
#         embedded = embedded.permute(1, 0, 2)
#         # embedded: [batch_size, sent_len, emedding_dim]
#         pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
#         # pooled: [batch_size, embedding_dim]
#         return self.fc(pooled)
#         # returned: [batch_size, output_dim]



# class KeywordClassifier(FeedbackClassifier):
#     def train(self, *args, **kwargs):
#         pass

#     def predict(self, text):
#         if any(w in text[:-1].lower().split() for w in ['yes', 'correct', 'right']):
#             return 1
#         elif any(w in text[:-1].lower().split() for w in ['no', 'incorrect', 'wrong', 'sorry,']):
#             return -1
#         else:
#             return 0

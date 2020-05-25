from torch_lr_finder import LRFinder


class LRFinderClass:
	def __init__(self,model, optimizer, criterion, device):
		self.model, self.optimizer, self.criterion, self.device  = model, optimizer, criterion, device
		
	def getBestLr_where_lowLoss(self,train_loader,end_lr=100, num_iter=100, step_mode="exp"):
		lr_finder = LRFinder(model=self.model, optimizer=self.optimizer, criterion=self.criterion, device=self.device)
		lr_finder.range_test(train_loader=train_loader, end_lr=end_lr, num_iter=num_iter, step_mode=step_mode)
		lr_finder.plot() # to inspect the loss-learning rate graph
		lr_finder.reset() # to reset the model and optimizer to their initial state

		print('Min loss value is : {} \nMin LR value is   : {}'.format(min(lr_finder.history['loss']),format(min(lr_finder.history['lr']),'.10f')))
		print('Min loss observed at index : {}'.format(lr_finder.history['loss'].index(min(lr_finder.history['loss']))))
		print('so corresponding LR value at that index is : {}'.format(lr_finder.history['lr'][lr_finder.history['loss'].index(min(lr_finder.history['loss']))]))
		return(min(lr_finder.history['loss']),lr_finder.history['lr'][lr_finder.history['loss'].index(min(lr_finder.history['loss']))])
'''
model = model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001,momentum=0.2)
#optimizer = optim.Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
'''


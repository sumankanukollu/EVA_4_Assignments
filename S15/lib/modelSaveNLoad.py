from os.path import join

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    homepath = r'/content/drive/My Drive/EVA4/S15/modelWeights/'
    # load check point
    checkpoint = torch.load(os.path.join(homepath,'checkpoint',checkpoint_fpath))
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
    
    
    
    
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    homepath = r'/content/drive/My Drive/EVA4/S15/modelWeights/'
    f_path = join(homepath,'checkpoint',checkpoint_path)
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = join(homepath,'best_model',best_model_path)
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)
class colorplate:
    """
    Color in HTML format for plotting 
    """
    red = "#D23918" # luoshenzhu
    blue = "#2E59A7" # qunqing
    yellow = "#E5A84B" # huanghe liuli
    cyan = "#5DA39D" # er lv
    black = "#151D29" # lanjian

def plot_loss(history, save_file:None, dpi = 200):
    """
    Plot the loss evolution during training
    Args: 
        history     : A dictionary contains loss evolution in list
        save_file   : Path to save the figure, if None, then just show the plot
        dpi         : The dpi for save the file 

    Returns:
        A fig for loss 

    """
    import matplotlib.pyplot as plt 
    import utils.plt_rc_setup
    from utils.plot import colorplate as cc 

    fig, axs = plt.subplots(1,1,figsize=(12,5))
    
    axs.semilogy(history["train_loss"],lw = 2, c = cc.blue)
    if len(history["val_loss"]) != 0 :
        axs.semilogy(history["val_loss"],lw = 2, c = cc.red)
    axs.set_xlabel("Epoch")
    axs.set_ylabel("MSE Loss") 
    
    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight', dpi= dpi)


def plot_signal(test_data, Preds, save_file:None, dpi = 200):
    """
    Plot the temproal evolution of prediction and test data
    Args: 
        test_data   : A numpy array of test data 
        Preds       : A numpy array of prediction data
        
        save_file   : Path to save the figure, if None, then just show the plot
        dpi         : The dpi for save the file 

    Returns:
        A fig for temporal dynamic of ground truth and predictions on test data

    """
    import sys
    
    try: 
        test_data.shape == Preds.shape 
    except:
        print("The prediction and test data must have same shape!")
        sys.exit()
    
    import matplotlib.pyplot as plt 
    import utils.plt_rc_setup
    from utils.plot import colorplate as cc 

    Nmode = min(test_data.shape[0],test_data.shape[-1])

    fig, axs = plt.subplots(Nmode,1,figsize=(16,2.5* Nmode),sharex=True)
    
    for i, ax in enumerate(axs):
        ax.plot(test_data[:,i],c = cc.black,lw = 1.5)
        ax.plot(Preds[:,i],c = cc.blue,lw = 1.5)
        ax.set_ylabel(f"M{i+1}")
        axs[-1].set_xlabel("t",fontsize=20)
    if save_file != None:
        plt.savefig(save_file, bbox_inches='tight', dpi= dpi)
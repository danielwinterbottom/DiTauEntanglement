import matplotlib.pyplot as plt

def plot_loss(loss_values, val_loss_values, output_dir='nn_plots'):
    plt.figure()
    plt.plot(range(1, len(loss_values)+1), loss_values, label='train loss')
    plt.plot(range(1, len(val_loss_values)+1), val_loss_values, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(f'{output_dir}/loss_vs_epoch_live.pdf')
    plt.close()
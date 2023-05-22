import matplotlib.pyplot as plt

def plt_comparison(ref, other, cmap='bone', clim=(0, 1), titles=['', ''], metricname=None, metric=None, zoom=None):
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    axzoom0 = fig.add_axes([0.34, 0.05, 0.14, 0.26], zorder=1)
    axzoom1 = fig.add_axes([0.835, 0.05, 0.14, 0.26], zorder=1)
    
    axs[0].imshow(ref, vmin=clim[0], vmax=clim[1], cmap=cmap)
    axs[0].axis('tight')
    axs[0].axis('off')
    axs[0].set_title(titles[0], fontsize=30, fontweight='bold')
    axzoom0.imshow(ref[zoom[0]:zoom[1], zoom[2]:zoom[3]], vmin=clim[0], vmax=clim[1], cmap=cmap)
    axzoom0.axis('tight')
    axzoom0.axis('off')

    axs[1].imshow(other, vmin=clim[0], vmax=clim[1], cmap=cmap)
    axs[1].axis('tight')
    axs[1].axis('off')
    axs[1].set_title(titles[1] + ('' if metric is None else f' ({metricname}: {metric(ref, other):.1f})'), 
                     fontsize=30, fontweight='bold')
    axzoom1.imshow(other[zoom[0]:zoom[1], zoom[2]:zoom[3]], vmin=clim[0], vmax=clim[1], cmap=cmap)
    axzoom1.axis('tight')
    axzoom1.axis('off')

    plt.tight_layout()
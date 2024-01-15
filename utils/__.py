"""
Chaotic Nature Analysis Using Poinare Map
"""

import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np 
# https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67



def t_zero_cross(data,direction_positive = True):
    zero_crossings = np.where( np.diff(np.signbit(data)) )
    if (direction_positive):
        positive = np.where( np.diff(data) > 0 )
    else:
        positive = np.where( np.diff(data) < 0 )

    crossings_idx = np.intersect1d(zero_crossings, positive)
    crossings_idx_next = crossings_idx +1

    x_values = crossings_idx - data[list(crossings_idx)]/(data[list(crossings_idx_next)] - data[list(crossings_idx)])


    return crossings_idx, x_values


def PDF(ax,x,y,xmin,xmax,ymin,ymax,color):

    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:50j, ymin:ymax:50j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    cset = ax.contour(xx, yy, f, colors=color)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(xmin, xmax)
    ax.grid(visible=True,markevery=1,color='gainsboro', zorder=1)

def completePDFmatrixFigure(series, positive_direction, plane, fig=None, axs=None, color='k'):
    lims=1
    nmodes = series.shape[1]

    if fig is None:
        fig, axs = plt.subplots(nmodes, nmodes, figsize=(25, 25))

    zeros, x_cross = t_zero_cross(series[:,plane],positive_direction)

    intersections = np.zeros((zeros.shape[0],nmodes))
    for axis in range(0,nmodes):
        intersections[:,axis] = np.interp(x_cross, np.arange(series.shape[0]), series[:,axis])


    for i in range(0,nmodes):
        for j in range(0,nmodes):
            if(i==j or j==plane or i==plane or j>i):
                axs[i,j].set_visible(False)
                continue
            PDF(axs[i,j], intersections[:,i], intersections[:,j],-lims,lims,-lims,lims,color)
            #axs[i,j].set_title('Section r{}=0'.format(ax+1))
            axs[i,j].set_xlabel('$r_{}$'.format(i), fontsize="large")
            axs[i,j].set_ylabel('$r_{}$'.format(j), fontsize="large")

    fig.tight_layout()
    return fig, axs


def detailPDFfigure(series, positive_direction, plane, fig=None, axs=None, color='k', i=1, j=1, model='',dpi= 100):
    lims=1
    nmodes = series.shape[1]

    if fig is None:
        fig, axs = plt.subplots(1, 1, figsize=(3, 3), dpi=dpi)

    zeros, x_cross = t_zero_cross(series[:,plane],positive_direction)

    intersections = np.zeros((zeros.shape[0],nmodes))
    for axis in range(0,nmodes):
        intersections[:,axis] = np.interp(x_cross, np.arange(series.shape[0]), series[:,axis])


    PDF(axs, intersections[:,i], intersections[:,j],-lims,lims,-lims,lims,color)
    axs.text(0.08, 0.08, '$r_{}=0$'.format(plane+1), fontsize=10,
                transform=axs.transAxes, bbox=dict(facecolor='white', alpha=0.4))

    axs.set_xlabel('$r_{10}$', fontsize="large") #OJO
    axs.set_ylabel('$r_{}$'.format(j+1), fontsize="large")

    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig, axs
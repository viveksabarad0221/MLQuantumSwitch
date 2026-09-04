import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def barplot_3d(figure, matrix, z_label: str, num_qubit=2, sub_row=1, sub_col=1, sub_idx=1,
               azim_view=-63.5, elev_view=23., dist_view=8,
               color='twilight_shifted', ticksize=18, xtick_rotation=0, ytick_rotation=0, barwidth=1):
    """
    This function generates a 3D-bar plot. The function is specialized on plotting qubit density matrices.
    :param figure: Figure object the plot will be assigned to.
    :param matrix: matrix to visualize as numpy array
    :param z_label: Label of the z-axis.
                    if z_label = 'r', the real part label is attached.
                    if z_label = 'i', the imaginary part label is attached.
                    else, the label can be customized.
    :param num_qubit: Number of qubits in the density matrix.
    :param sub_row: subplot row.
    :param sub_col: subplot column.
    :param sub_idx: subplot index
    :param azim_view: azimuthal viewing angle.
    :param elev_view: elevation viewing angle.
    :param dist_view: viewing distance.
    :param color: colormap of the bar pülot.
    :param ticksize: tick size of x- and y-axis.
    :param xtick_rotation: rotation angle of the x-ticks.
    :param ytick_rotation: rotation angle of the y-ticks.
    :param barwidth: width of the bars in interval (0, 1).
    :return: axis object of the figure.
    """
    assert num_qubit in [1, 2], 'Currently the function only provides two or four dimensional plots, sorry!'

    regular_path = r'C:\Users\richt\Fonts\TeX-Gyre-Heros\texgyreheros-regular.otf'
    bold_path = r'C:\Users\richt\Fonts\TeX-Gyre-Heros\texgyreheros-bold.otf'

    font_regular = fm.FontProperties(fname=regular_path)
    font_bold = fm.FontProperties(fname=bold_path)
    x = np.arange(1, len(matrix[:, 0]) + 1, 1, dtype=float)
    y = np.copy(x)
    xpos, ypos = np.meshgrid(x, y)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)
    dx = barwidth * np.ones_like(zpos)
    dy = dx.copy()
    dz = matrix.flatten()

    colorMap = plt.cm.get_cmap(color)
    ax = figure.add_subplot(sub_row, sub_col, sub_idx, projection='3d')

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colorMap((dz + 1) / 2))
    if num_qubit == 1:
        tick = np.array(['H', 'V'])
        ticks_loc = np.arange(1.5, 3.5, 1)
    elif num_qubit == 2:
        tick = np.array(['HH', 'HV', 'VH', 'VV'])
        ticks_loc = np.arange(1, 5, 1)
    else:
        print(f'Method is not available for {num_qubit} qubit.')
        return
    plt.xticks(ticks_loc, tick, rotation=xtick_rotation, size=ticksize)
    plt.yticks(np.arange(2, 6, 1), tick, rotation=ytick_rotation, size=ticksize)
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax.azim = azim_view
    ax.elev = elev_view
    ax.dist = dist_view
    ax.set_zlim3d(-0.2, 0.6)
    ax.zaxis.set_rotate_label(False)
    if z_label == 'r':
        z_label = r'$\Re[\rho]$'
    elif z_label == 'i':
        z_label = r'$\Im[\rho]$'
    else:
        pass

    #plt.xticks(fontsize=30, fontproperties=font_bold)
    #plt.yticks(fontsize=30, fontproperties=font_bold)
    #ax.yaxis.set_tick_params(labelsize=25, fontproperties=font_bold)
    #ax.xaxis.set_tick_params(labelsize=25, fontproperties=font_bold)
    #ax.zaxis.set_tick_params(labelsize=25, fontproperties=font_bold)
    ax.yaxis.set_ticklabels(labels=tick, fontproperties=font_regular, size=20)
    ax.xaxis.set_ticklabels(labels=tick, fontproperties=font_regular, size=20)
    ax.zaxis.set_ticklabels(labels=[-1, -0.5, 0, 0.5, 1], fontproperties=font_regular, size=20)
    ax.set_zlabel(z_label, fontsize=20, labelpad=15)
    # change tick label fontsize
    return ax

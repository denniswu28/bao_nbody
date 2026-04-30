import numpy as np
from scipy.integrate import quad
import yaml
import os


def comoving_distance(z, h, Omega_m, Omega_de=None, w0=-1.0, wa=0.0):
    # comoving distance via numerical integration; CPL dark energy
    if Omega_de is None:
        Omega_de = 1 - Omega_m

    c_over_H0 = 2997.9   # Mpc/h

    def E(zp):
        a = 1 / (1 + zp)
        Omega_de_z = Omega_de * a**(-3*(1 + w0 + wa)) * np.exp(-3 * wa * (1 - a))
        return np.sqrt(Omega_m * (1 + zp)**3 + Omega_de_z)

    chi, _ = quad(lambda zp: 1.0 / E(zp), 0, z, limit=100)
    return c_over_H0 * chi


def angular_diameter_distance(z, h, Omega_m, **kwargs):
    return comoving_distance(z, h, Omega_m, **kwargs) / (1 + z)


def hubble_z(z, h, Omega_m, Omega_de=None, w0=-1.0, wa=0.0):
    if Omega_de is None:
        Omega_de = 1 - Omega_m
    a = 1 / (1 + z)
    Omega_de_z = Omega_de * a**(-3*(1 + w0 + wa)) * np.exp(-3 * wa * (1 - a))
    E = np.sqrt(Omega_m * (1 + z)**3 + Omega_de_z)
    return 100.0 * E   # H0 = 100h km/s/Mpc


def load_config(fname):
    with open(fname, 'r') as f:
        return yaml.safe_load(f)


def plot_pk_comparison(k_list, Pk_list, labels, colors=None, title='',
                       fname=None, k_range=(1e-2, 0.4)):
    import matplotlib.pyplot as plt

    if colors is None:
        colors = [f'C{i}' for i in range(len(k_list))]

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                              gridspec_kw={'height_ratios': [2, 1]})

    # use first curve as reference for ratio panel
    k_ref, Pk_ref = k_list[0], Pk_list[0]
    from scipy.interpolate import interp1d
    Pk_ref_interp = interp1d(k_ref, Pk_ref, bounds_error=False, fill_value=np.nan)

    for k, Pk, label, color in zip(k_list, Pk_list, labels, colors):
        axes[0].loglog(k, Pk, label=label, color=color, alpha=0.8)

        Pk_interp = interp1d(k, Pk, bounds_error=False, fill_value=np.nan)
        k_common = np.logspace(np.log10(k_range[0]), np.log10(k_range[1]), 200)
        ratio = Pk_interp(k_common) / Pk_ref_interp(k_common)
        axes[1].semilogx(k_common, ratio, color=color, alpha=0.8)

    axes[1].axhline(1, color='k', lw=0.8, ls='--')
    axes[0].set_ylabel(r'$P(k)$ [$(h^{-1}$Mpc$)^3$]')
    axes[0].legend(fontsize='small')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(title)
    axes[0].set_xlim(k_range)

    axes[1].set_xlabel(r'$k$ [$h$/Mpc]')
    axes[1].set_ylabel(f'Ratio to {labels[0]}')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0.5, 1.5)

    plt.tight_layout()
    if fname:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(fname, dpi=150)
    return fig


def plot_density_slices(snapshots, L, fname=None):
    import matplotlib.pyplot as plt

    n = len(snapshots)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for ax, snap in zip(axes, snapshots):
        delta_proj = snap['delta'].sum(axis=2)
        vmax = np.percentile(np.abs(delta_proj), 99)
        im = ax.imshow(delta_proj, cmap='RdBu_r', origin='lower',
                       vmin=-vmax, vmax=vmax,
                       extent=[-L/2, L/2, -L/2, L/2])
        ax.set_title(f"z = {snap['z']:.2f}")
        ax.set_xlabel(r'$x$ [Mpc/$h$]')
        ax.set_ylabel(r'$y$ [Mpc/$h$]')
        plt.colorbar(im, ax=ax, label=r'$\delta$ (projected)')

    plt.tight_layout()
    if fname:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        fig.savefig(fname, dpi=150)
    return fig


def make_animation(snapshot_dir, output_fname, L, fps=5):
    import glob
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    try:
        import imageio_ffmpeg
        matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    snap_files = sorted(glob.glob(os.path.join(snapshot_dir, 'snap_*.h5')))
    if not snap_files:
        print("no HDF5 snapshots found for animation.")
        return

    # load all density fields
    frames = []
    for f in snap_files:
        with h5py.File(f, 'r') as hf:
            delta = hf['delta'][:]
            z = float(hf.attrs['z'])
        # project a thin slab along z for contrast
        slab = delta.shape[2] // 8
        delta_proj = delta[:, :, :slab].mean(axis=2)
        frames.append((delta_proj, z))

    print(f"rendering animation: {len(frames)} frames at {fps} fps")

    # one color scale for all frames
    vmax_global = max(np.percentile(np.abs(f[0]), 99.5) for f in frames)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(np.log10(np.clip(frames[0][0] + 1, 1e-2, None)),
                   cmap='inferno', origin='lower',
                   extent=[-L/2, L/2, -L/2, L/2],
                   vmin=-0.5, vmax=np.log10(1 + vmax_global))
    plt.colorbar(im, ax=ax, label=r'$\log_{10}(1 + \delta)$')
    title = ax.set_title(f'z = {frames[0][1]:.2f}', fontsize=16, fontweight='bold')
    ax.set_xlabel(r'$x$ [Mpc/$h$]', fontsize=12)
    ax.set_ylabel(r'$y$ [Mpc/$h$]', fontsize=12)

    def update(i):
        delta_proj, z = frames[i]
        data = np.log10(np.clip(delta_proj + 1, 1e-2, None))
        im.set_data(data)
        title.set_text(f'z = {z:.2f}')
        return [im, title]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000//fps, blit=False)

    os.makedirs(os.path.dirname(output_fname), exist_ok=True)
    writer = FFMpegWriter(fps=fps, extra_args=['-pix_fmt', 'yuv420p'])
    anim.save(output_fname, writer=writer, dpi=150)
    plt.close(fig)
    print(f"animation saved: {output_fname}")


def pk_error_gaussian(Pk, nmodes):
    # gaussian P(k) variance: sigma = sqrt(2/N_modes) * P
    return np.sqrt(2.0 / nmodes) * np.abs(Pk)

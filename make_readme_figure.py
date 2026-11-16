"""Render the README preview from committed checkpoints; no synthetic flow data."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from taylor_green import PeriodicMLP, derivative, exact


def main():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    n = 128
    axis = torch.linspace(0, 2*np.pi, n)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    z = torch.stack((x, y, torch.full_like(x, 2.0)), -1).reshape(-1, 3)
    truth = exact(z, .1).detach().numpy().reshape(n, n, 3)
    fields = [(truth, 2*np.sin(x.numpy())*np.sin(y.numpy())*np.exp(-.4))]
    for name in ("pinn_seed0", "hard_ic_seed0"):
        checkpoint = torch.load(f"results/{name}/after_lbfgs.pt", weights_only=True)
        cfg = checkpoint["config"]
        net = PeriodicMLP(cfg["width"], cfg["depth"], cfg["horizon"], cfg.get("hard_ic", False))
        net.load_state_dict(checkpoint["state_dict"])
        q = z.clone().requires_grad_(True)
        pred = net(q)
        omega = derivative(pred[:, 1], q)[:, 0]-derivative(pred[:, 0], q)[:, 1]
        fields.append((pred.detach().numpy().reshape(n, n, 3), omega.detach().numpy().reshape(n, n)))

    with plt.rc_context({"font.family":"DejaVu Sans", "font.size":10,
                         "text.color":"#e5eef7", "axes.labelcolor":"#aabbcf",
                         "xtick.color":"#aabbcf", "ytick.color":"#aabbcf",
                         "axes.edgecolor":"#33485f", "axes.facecolor":"#142339",
                         "figure.facecolor":"#0b1525", "axes.titlecolor":"#e5eef7"}):
        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 3, left=.055, right=.965, bottom=.08, top=.79, hspace=.34, wspace=.30)
        fig.text(.055,.935,"PINN RELIABILITY EXPERIMENTS",fontsize=23,weight="bold")
        fig.text(.055,.888,"An exact flow. A learned solution. An error we can measure.",fontsize=14,color="#b0c3d6")
        fig.text(.055,.846,"2D Taylor–Green  /  ν = 0.1  /  t = 2  /  2,403 parameters  /  physics-only PINN training",fontsize=10,color="#8ba7bf")
        titles=["EXACT REFERENCE", "SOFT INITIAL CONDITION · SEED 0", "EXACT INITIAL CONDITION · SEED 0"]
        for col,((field,omega),title) in enumerate(zip(fields,titles)):
            ax=fig.add_subplot(gs[0,col])
            im=ax.imshow(omega,origin="lower",extent=(0,2*np.pi,0,2*np.pi),cmap="RdBu_r",vmin=-1.35,vmax=1.35)
            ax.streamplot(axis.numpy(),axis.numpy(),field[:,:,0],field[:,:,1],density=.85,color="#172334",linewidth=.65,arrowsize=.65)
            ax.set_title(title,loc="left",fontsize=10,pad=12)
            ax.set_xticks([0,np.pi,2*np.pi],["0","π","2π"]);ax.set_yticks([0,np.pi,2*np.pi],["0","π","2π"])
            ax.set_xlabel("x");ax.set_ylabel("y")
            cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03);cb.set_label("Vorticity ∂v/∂x − ∂u/∂y",fontsize=9)
        ax=fig.add_subplot(gs[1,0])
        for prefix,color,label in [("pinn","#f7aa86","Soft IC"),("hard_ic","#6bd3ce","Exact IC")]:
            for seed in (0,1):
                d=json.loads(Path(f"results/{prefix}_seed{seed}/metrics.json").read_text())
                values=d["stages"][-1]["metrics"]["velocity_rel_l2_by_time"]
                ax.plot(d["times"],100*np.array(values),color=color,ls="-" if seed==0 else "--",lw=2,label=f"{label}, seed {seed}")
        ax.set(title="Velocity error across time",xlabel="Time t",ylabel="Relative L2 error (%)")
        ax.grid(alpha=.12);ax.legend(frameon=False,fontsize=8,labelcolor="#dce9f5")
        for col in (1,2):
            ax=fig.add_subplot(gs[1,col])
            error=np.linalg.norm(fields[col][0][:,:,:2]-truth[:,:,:2],axis=2)
            im=ax.imshow(error,origin="lower",extent=(0,2*np.pi,0,2*np.pi),cmap="magma",vmin=0,vmax=error.max())
            ax.set_title("Absolute velocity error · own color scale",loc="left",fontsize=10,pad=12)
            ax.set_xticks([0,np.pi,2*np.pi],["0","π","2π"]);ax.set_yticks([0,np.pi,2*np.pi],["0","π","2π"])
            ax.set_xlabel("x");ax.set_ylabel("y")
            cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.03);cb.set_label("‖velocity − exact velocity‖₂",fontsize=9)
        fig.text(.055,.025,"Exploratory two-seed comparison · Same base MLP, points and iteration limits · Streamlines are instantaneous · No uncertainty guarantee",fontsize=9,color="#9fb4c8")
        Path("docs/assets").mkdir(parents=True,exist_ok=True)
        fig.savefig("docs/assets/taylor-green-preview.png",dpi=170,bbox_inches="tight",pad_inches=.25)
        plt.close(fig)


if __name__ == "__main__":
    main()

from TFM import *



# generate a batch of B synthetic datasets by sampling from Gaussian ER-DAGs; in
# each, the target is a node drawn uniformly at random and n_cov of the remaining
# nodes are the covariates, so the model sees parents, children and unrelated nodes
def create_dataset(B, n, n_cov):
    n_train, n_test = 100, 10

    # random upper-triangular adjacency (acyclic by construction)
    adj = torch.triu(torch.bernoulli(torch.full((B, n, n), 0.5)), diagonal=1)
    # edge weights, zero where there's no edge
    weights = adj * torch.empty(B, n, n).uniform_(0.5, 1.5)

    # linear SEM: x = (I - W)^-1 @ noise, with a per-variable noise scale
    mixing = torch.linalg.inv(torch.eye(n) - weights)
    noise_sd = torch.empty(B, 1, n).uniform_(0.2, 0.5)
    noise = torch.randn(B, n_train + n_test, n) * noise_sd
    data = noise @ mixing.transpose(1, 2)

    # one random permutation of the nodes per dataset: the first node is the
    # target, the next n_cov are the covariates
    perm = torch.rand(B, n).argsort(dim=1)
    x = torch.take_along_dim(data, perm[:, None, 1:1+n_cov], dim=2)  # (B, R, n_cov)
    y = torch.take_along_dim(data, perm[:, None, :1], dim=2).squeeze(-1)  # (B, R)
    x_train, y_train = x[:, :n_train], y[:, :n_train]
    x_test, y_test = x[:, n_train:], y[:, n_train:]
    return x_train, y_train, x_test, y_test


# training loop; each step averages the loss over a batch of batch_size fresh
# datasets from create_dataset, to tame the gradient variance of a single-task
# step. resumes from the checkpoint in RESULTS when there is one, else starts fresh
def train(d_model, max_col, n_iter, batch_size, n_col_train, n_cov):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = RESULTS / "tfm.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        model = TFM(d_model=ckpt["d_model"], max_col=ckpt["max_col"])
        model.load_state_dict(ckpt["state_dict"])
        loss_curve = ckpt["loss_curve"]
        print(f"resuming from {ckpt_path} after {len(loss_curve)} iterations")
    else:
        model = TFM(d_model=d_model, max_col=max_col)
        loss_curve = []

    # move before building the optimizer, so it holds the device parameters
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    print_every = max(1, n_iter // 20)
    for i in range(n_iter):
        x_train, y_train, x_test, y_test = [
            t.to(device) for t in create_dataset(batch_size, n_col_train, n_cov)]
        optimizer.zero_grad()
        logits, y_t_mean, y_t_sd = model(x_train, y_train, x_test)
        test_logits = logits[:, x_train.shape[1]:]  # (B, M, n_bins), unlabeled rows only

        # bin true test y (same standardize+sigmoid map as in forward) into model.n_bins bins
        p_test = torch.sigmoid((y_test-y_t_mean)/y_t_sd)  # (B, M); stats are (B, 1)
        bin_idx = (p_test * model.n_bins).long().clamp(0, model.n_bins - 1)

        # every dataset has M test rows, so this equals the mean of per-dataset losses
        loss = ce_loss(test_logits.flatten(0, 1), bin_idx.flatten())
        loss.backward()
        optimizer.step()
        loss_curve.append(loss.item())
        if (i + 1) % print_every == 0 or i == n_iter - 1:
            print(f"iter {i}: loss={loss.item():.4f}")
            save_run(model, loss_curve)  # checkpoint, so an interrupt keeps progress
    return loss_curve


# persist the trained weights alongside the loss curve behind them
def save_run(model, loss_curve):
    import matplotlib.pyplot as plt

    RESULTS.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "d_model": model.d_model,
                "max_col": model.max_col,
                "loss_curve": loss_curve},
               RESULTS / "tfm.pt")

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(loss_curve, lw=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("cross-entropy loss")
    fig.tight_layout()
    fig.savefig(RESULTS / "learning_curve.pdf")
    plt.close(fig)


# illustration check: in-context predictions on $y = x + \epsilon$, using the
# weights written by save_run, with the context given once and then duplicated
def illustrate():
    import matplotlib.pyplot as plt
    import seaborn as sns

    ckpt = torch.load(RESULTS / "tfm.pt", weights_only=True, map_location="cpu")
    model = TFM(d_model=ckpt["d_model"], max_col=ckpt["max_col"])
    model.load_state_dict(ckpt["state_dict"])

    n_train, n_test = 100, 10
    x = torch.randn(n_train + n_test, 1)
    y = x.squeeze(-1) + torch.randn(n_train + n_test)
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    def scatter(ax, x_tr, y_tr, title):
        preds = model.predict(x_tr[None], y_tr[None], x_test[None])[0]  # batch of one
        sns.scatterplot(x=x_tr.squeeze(-1).numpy(), y=y_tr.numpy(),
                        label="train", alpha=.3, ax=ax)
        sns.scatterplot(x=x_test.squeeze(-1).numpy(), y=y_test.numpy(),
                        label="true", ax=ax)
        sns.scatterplot(x=x_test.squeeze(-1).numpy(), y=preds.numpy(),
                        label="pred", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True, sharey=True)
    scatter(axes[0], x_train, y_train, f"{n_train} context rows")
    # same pairs, each appearing twice
    scatter(axes[1], x_train.repeat(2, 1), y_train.repeat(2),
            f"{2*n_train} context rows (each pair twice)")
    fig.tight_layout()
    fig.savefig(RESULTS / "illustration.pdf")
    plt.show()


# probability the model puts on each cell of an (x, y) grid, given one context
# (x_ctx, y_ctx) with the grid's x values as test rows; saved to RESULTS / fname
def plot_predictive_grid(model, x_ctx, y_ctx, fname):
    import matplotlib.pyplot as plt

    M = 100  # number of x grid points, which are the test rows
    x_grid = torch.linspace(-2, 2, M)
    y_grid = torch.linspace(-2, 2, 10)

    with torch.inference_mode():
        # a batch of one context, with the grid points as its test rows
        logits, y_t_mean, y_t_sd = model(x_ctx[None], y_ctx[None], x_grid[None, :, None])  # logits is (B, N+M, n_bins)
        probs = model.sm_classify(logits[0, len(x_ctx):])  # (M, n_bins)

    # each grid y lands in one bin under the same standardize+sigmoid map as forward
    bin_idx = (torch.sigmoid((y_grid-y_t_mean.item())/y_t_sd.item()) * model.n_bins).long()
    bin_idx = bin_idx.clamp(0, model.n_bins - 1)
    density = probs[:, bin_idx].T  # (len(y_grid), M); row j holds y_grid[j]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    # origin='lower' puts the most negative x and y at the bottom left
    im = ax.imshow(density.numpy(), origin="lower", aspect="auto",
                   extent=(-2., 2., -2., 2.))
    ax.scatter(x_ctx.squeeze(-1).numpy(), y_ctx.numpy(), s=8,
               c="w", edgecolor="k", lw=.3, label="train")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{len(x_ctx)} context rows")
    fig.colorbar(im, ax=ax, label="predicted probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / fname)
    plt.show()


# predictive grid for 100 in-context train points from $y = x + \epsilon$,
# drawn from a fixed seed so visualize_doubling conditions on the same points
def visualize():
    ckpt = torch.load(RESULTS / "tfm.pt", weights_only=True, map_location="cpu")
    model = TFM(d_model=ckpt["d_model"], max_col=ckpt["max_col"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    g = torch.Generator().manual_seed(0)
    n_train = 100
    x_train = torch.randn(n_train, 1, generator=g)
    y_train = x_train.squeeze(-1) + torch.randn(n_train, generator=g)
    plot_predictive_grid(model, x_train, y_train, "predictive_grid.pdf")


if __name__ == '__main__':
    RESULTS = Path(__file__).parent / "results"
    
    train(d_model=16, max_col=3, 
          n_iter=200, batch_size=400, n_col_train=5, n_cov=3)
    illustrate()
    visualize()
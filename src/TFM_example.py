import torch
import torch.nn as nn


class TFM(nn.Module):
    def __init__(self, d_model, max_col):
        """
        Args:
            d_model (int): embedding dimension
            max_col (int): maximal number of columns
        """
        super().__init__()
        # some constants
        self.d_model = d_model
        self.max_col = max_col
        self.n_bins = 4
        # embedding layers; projects each individual entry up
        self.embed_X = nn.Linear(1, d_model)
        self.embed_Y = nn.Linear(1, d_model)
        # attention within rows (between cols);
        self.row_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=2, 
                dim_feedforward=64, 
                dropout=0.,
                batch_first=True
        ), num_layers=1, enable_nested_tensor=False)
        # attention within cols (between rows)
        self.col_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=2,
                dim_feedforward=64,
                dropout=0.,
                batch_first=True
        ), num_layers=1, enable_nested_tensor=False)
        # out projection
        self.d_model_proj = nn.Linear(d_model, 1)
        self.col_proj = nn.Linear(max_col, self.n_bins) # split Y into 100 bins
        self.sm_classify = nn.Softmax(-1)
        
    def forward(self, x_train, y_train, x_test):
        """
            x_train (N, C)
            y_train (N,)
            x_test  (M, C)
        """
        # constants for convenience
        N, C = x_train.shape
        M, _ = x_test.shape
        # simply assume it's the same for all
        device = x_train.device
        dtype = x_train.dtype

        # standardize features
        x_t_mean = x_train.mean(axis=0, keepdims=True)
        x_t_sd = x_train.std(axis=0, keepdims=True)
        x_train_std = (x_train-x_t_mean)/x_t_sd
        x_test_std = (x_test-x_t_mean)/x_t_sd

        # standardize y,then squash to (0,1) using sigmoid so bin edges have a
        # consistent meaning across draws
        y_t_mean = y_train.mean()
        y_t_sd = y_train.std()
        y_train_mm = torch.sigmoid((y_train-y_t_mean)/y_t_sd)

        ## embed x and pad to max_col 
        # concat train and test
        x_cat = torch.cat([x_train_std, x_test_std], dim=0)
        # embed
        x_embd = self.embed_X(x_cat.unsqueeze(-1))  # (N+M, max_col, d_model)
        # pad to c_max columns to be flexible in number of columns
        x_pad = torch.zeros(size=(N+M, self.max_col-C, self.d_model), 
                            dtype=dtype, device=device)
        # x is vector of all features, train and test
        x = torch.cat([x_embd, x_pad], dim=1)

        ## embed y_train and pad for test
        # embed
        y_embd = self.embed_Y(y_train_mm.unsqueeze(1))  # shape (N, d_model)
        # put zeros for the unlabeled observations
        y_pad = torch.zeros(size=(M, self.d_model), 
                            dtype=y_train.dtype, device=y_train.device)
        # y is vector of all targets, train and test (test just zeros)
        y = torch.cat([y_embd, y_pad], dim=0).unsqueeze(1)

        ## apply row transformer (between cols) with mask so only real cols are attended to
        # first dim is treated as batch, so we have a seq. of max_col elements
        # yes, we are using col_mask for the row_transformer. think about it!
        col_mask = torch.zeros(size=(self.max_col,self.max_col), 
                               dtype=torch.bool, device=device)
        col_mask[:, C:] = True  # mask padded columns
        row_transformed = self.row_transformer(x, mask=col_mask)

        # add x and y; 
        # note that we'll be using the batch dim as an actual dim!!!
        flavored = (row_transformed + y)  # still (N+M, max_col, d_model)

        ### row transformer with mask so only train obs are attended to
        # yes, we are using row_mask for the col_transformer. think about it!
        row_transformed_T = flavored.transpose(1, 0)  # crucial!
        row_mask = torch.zeros(size=(N+M, N+M), dtype=torch.bool, 
                               device=row_transformed.device)
        row_mask[:, N:] = True  # mask test rows
        col_transformed = self.col_transformer(row_transformed_T, mask=row_mask)

        # return to normal tabular order (N+M, C_max, d_model)
        transformed = col_transformed.transpose(1,0)

        ### out projection, starting from (N+M, C_max, d_model)
        mod_proj = self.d_model_proj(transformed).squeeze(-1)  # (N+M, C_max)
        logits = self.col_proj(mod_proj)  # (N+M, self.n_bins), raw (pre-softmax)

        # return logits and standardization stats
        return logits, y_t_mean, y_t_sd


    @torch.inference_mode()
    def predict(self, x_train, y_train, x_test):
        """ same as forward, but in inference and eval mode, and only returns test points """
        was_training = self.training
        self.eval()
        logits, y_t_mean, y_t_sd = self(x_train, y_train, x_test)
        probs = self.sm_classify(logits)
        # bin centers in (0,1), mapped back through the inverse sigmoid (logit)
        # and un-standardized to the original y scale
        bin_centers_mm = (torch.arange(self.n_bins, device=logits.device) + 0.5) / self.n_bins
        bin_centers = torch.logit(bin_centers_mm) * y_t_sd + y_t_mean
        out_exp = probs @ bin_centers
        if was_training:
            self.train()
        return out_exp[len(x_train):]  # return test predictions



# generate synthetic datasets by sampling from Gaussian ER-DAGs
def create_dataset(n):
    n_train, n_test = 100, 10

    # random upper-triangular adjacency (acyclic by construction)
    adj = torch.triu(torch.bernoulli(torch.full((n, n), 0.5)), diagonal=1)
    # edge weights, zero where there's no edge
    weights = adj * torch.empty(n, n).uniform_(0.5, 1.5)

    # linear SEM: x = (I - W)^-1 @ noise
    mixing = torch.linalg.inv(torch.eye(n) - weights)
    noise = torch.randn(n_train + n_test, n)
    data = noise @ mixing.T

    # last variable (a sink node) as target, rest as features
    x, y = data[:, :-1], data[:, -1]
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]
    return x_train, y_train, x_test, y_test


# training loop; each step averages the loss over accum_steps fresh draws
# from create_dataset, to tame the gradient variance of a single-task step
def train(model, n_iter, accum_steps, n_col_train):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    ce_loss = nn.CrossEntropyLoss()
    print_every = max(1, n_iter // 10)
    for i in range(n_iter):
        optimizer.zero_grad()
        losses = []
        for _ in range(accum_steps):
            x_train, y_train, x_test, y_test = create_dataset(n_col_train)
            logits, y_t_mean, y_t_sd = model(x_train, y_train, x_test)
            test_logits = logits[len(x_train):]  # need loss only on unlabeled rows

            # bin true test y (same standardize+sigmoid map as in forward) into model.n_bins bins
            p_test = torch.sigmoid((y_test-y_t_mean)/y_t_sd)
            bin_idx = (p_test * model.n_bins).long().clamp(0, model.n_bins - 1)

            losses.append(ce_loss(test_logits, bin_idx))

        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        if (i + 1) % print_every == 0:
            print(f"iter {i}: loss={loss.item():.4f}")


# illustration check: in-context predictions on $y = x + \epsilon$
def illustrate(model):
    n_train, n_test = 100, 10
    x = torch.randn(n_train + n_test, 1)
    y = x.squeeze(-1) + torch.randn(n_train + n_test)
    x_train, y_train = x[:n_train], y[:n_train]
    x_test, y_test = x[n_train:], y[n_train:]

    preds = model.predict(x_train, y_train, x_test)

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.scatterplot(x=x_train.squeeze(-1).numpy(), y=y_train.numpy(), label="train", alpha=0.3)
    sns.scatterplot(x=x_test.squeeze(-1).numpy(), y=y_test.numpy(), label="true")
    sns.scatterplot(x=x_test.squeeze(-1).numpy(), y=preds.numpy(), label="pred")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    tfm = TFM(d_model=16, max_col=4)
    train(tfm, n_iter=100, accum_steps=100, n_col_train=4)
    illustrate(tfm)
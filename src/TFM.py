from pathlib import Path

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
        self.n_bins = 10
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
        self.col_proj = nn.Linear(d_model, self.n_bins) # split Y into 100 bins
        self.sm_classify = nn.Softmax(-1)
        
    def forward(self, x_train, y_train, x_test):
        """
            x_train (B, N, C)
            y_train (B, N,)
            x_test  (B, M, C)
        """
        # constants for convenience
        B, N, C = x_train.shape
        B, M, _ = x_test.shape
        # simply assume it's the same for all
        device = x_train.device
        dtype = x_train.dtype

        # standardize features
        x_t_mean = x_train.mean(axis=1, keepdim=True)
        x_t_sd = x_train.std(axis=1, keepdim=True)
        x_train_std = (x_train-x_t_mean)/x_t_sd
        x_test_std = (x_test-x_t_mean)/x_t_sd

        # standardize y,then squash to (0,1) using sigmoid so bin edges have a
        # consistent meaning across draws
        y_t_mean = y_train.mean(axis=1, keepdim=True)
        y_t_sd = y_train.std(axis=1, keepdim=True)
        y_train_mm = torch.sigmoid((y_train-y_t_mean)/y_t_sd)

        ## embed x and pad to max_col 
        # concat train and test
        x_cat = torch.cat([x_train_std, x_test_std], dim=1)
        # embed
        x_embd = self.embed_X(x_cat.unsqueeze(-1))  # (B, N+M, max_col, d_model)
        # pad to c_max columns to be flexible in number of columns
        x_pad = torch.zeros(size=(B, N+M, self.max_col-C, self.d_model), 
                            dtype=dtype, device=device)
        # x is vector of all features, train and test
        x = torch.cat([x_embd, x_pad], dim=2)

        ## embed y_train and pad for test
        # embed
        y_embd = self.embed_Y(y_train_mm.unsqueeze(-1))  # shape (N, d_model)
        # put zeros for the unlabeled observations
        y_pad = torch.zeros(size=(B, M, self.d_model), 
                            dtype=y_train.dtype, device=y_train.device)
        # y is vector of all targets, train and test (test just zeros)
        y = torch.cat([y_embd, y_pad], dim=1).unsqueeze(-2)  # add col dim

        ## apply row transformer (between cols) with mask so only real cols are attended to
        # first dim is treated as batch, so we have a seq. of max_col elements
        # yes, we are using col_mask for the row_transformer. think about it!
        col_mask = torch.zeros(size=(self.max_col, self.max_col), 
                               dtype=torch.bool, device=device)
        col_mask[:, C:] = True  # mask padded columns
        # flatten batch and row dims into one so the transformer accepts it, then unflatten result
        row_transformed = self.row_transformer(x.flatten(0,1), mask=col_mask)
        row_transformed = row_transformed.unflatten(0, (B, N+M))

        # add x and y; 
        # note that we'll be using the batch dim as an actual dim!!!
        flavored = (row_transformed + y)  # still (N+M, max_col, d_model)

        ### row transformer with mask so only train obs are attended to
        # yes, we are using row_mask for the col_transformer. think about it!
        row_transformed_T = flavored.transpose(2, 1)  # swap row and cols
        row_mask = torch.zeros(size=(N+M, N+M), dtype=torch.bool, 
                               device=row_transformed.device)
        row_mask[:, N:] = True  # mask test rows
        # flatten batch and col dims into one so the transformer accepts it, then unflatten result
        col_transformed = self.col_transformer(row_transformed_T.flatten(0,1), mask=row_mask)
        col_transformed = col_transformed.unflatten(0, (B,self.max_col))

        # return to normal tabular order (B, N+M, C_max, d_model)
        transformed = col_transformed.transpose(2, 1)

        ### out projection, starting from (B, N+M, C_max, d_model)
        col_summed = transformed[:,:,:C].sum(axis=2)  # (B, N+M, d_model)
        logits = self.col_proj(col_summed)  # (B, N+M, self.n_bins), pre-softmax

        # return logits and standardization stats
        return logits, y_t_mean, y_t_sd


    @torch.inference_mode()
    def predict(self, x_train, y_train, x_test):
        """ Same as forward but in inference and eval mode, and return test points only. """
        was_training = self.training
        self.eval()
        logits, y_t_mean, y_t_sd = self(x_train, y_train, x_test)
        probs = self.sm_classify(logits)
        # bin centers in (0,1), mapped back through the inverse sigmoid (logit)
        # and un-standardized to the original y scale
        bin_centers_mm = (torch.arange(self.n_bins, device=logits.device) + 0.5) / self.n_bins  # aranges bin centers in [0,1]
        bin_centers = torch.logit(bin_centers_mm) * y_t_sd + y_t_mean  # (B, n_bins)
        out_exp = (probs * bin_centers.unsqueeze(1)).sum(-1)  # (B, N+M)
        if was_training:
            self.train()
        return out_exp[:, x_train.shape[1]:]  # return test predictions
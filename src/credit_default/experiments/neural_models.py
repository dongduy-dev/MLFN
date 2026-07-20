import torch
import torch.nn as nn

class SharedStaticBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(15, 16),
            nn.ReLU(),
            nn.Dropout(0.20)
        )
        
    def forward(self, x_static):
        return self.encoder(x_static)

class SharedFusionHead(nn.Module):
    def __init__(self, fused_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, 1)
        )
        
    def forward(self, x_fused):
        return self.head(x_fused)

class GRUSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_branch = SharedStaticBranch()
        self.rnn = nn.GRU(input_size=3, hidden_size=32, num_layers=1, bidirectional=False, batch_first=True)
        self.fusion = SharedFusionHead(32 + 16)
        
    def forward(self, x_temporal, x_static):
        # x_temporal: (batch, 6, 3)
        static_emb = self.static_branch(x_static)
        out, h_n = self.rnn(x_temporal)
        temporal_emb = h_n[-1]
        fused = torch.cat([temporal_emb, static_emb], dim=1)
        return self.fusion(fused)

class GRUDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_branch = SharedStaticBranch()
        self.rnn = nn.GRU(input_size=3, hidden_size=64, num_layers=2, dropout=0.25, bidirectional=False, batch_first=True)
        self.fusion = SharedFusionHead(64 + 16)
        
    def forward(self, x_temporal, x_static):
        static_emb = self.static_branch(x_static)
        out, h_n = self.rnn(x_temporal)
        temporal_emb = h_n[-1]
        fused = torch.cat([temporal_emb, static_emb], dim=1)
        return self.fusion(fused)

class LSTMDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_branch = SharedStaticBranch()
        self.rnn = nn.LSTM(input_size=3, hidden_size=64, num_layers=2, dropout=0.25, bidirectional=False, batch_first=True)
        self.fusion = SharedFusionHead(64 + 16)
        
    def forward(self, x_temporal, x_static):
        static_emb = self.static_branch(x_static)
        out, (h_n, c_n) = self.rnn(x_temporal)
        temporal_emb = h_n[-1]
        fused = torch.cat([temporal_emb, static_emb], dim=1)
        return self.fusion(fused)

class Conv1DSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_branch = SharedStaticBranch()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.fusion = SharedFusionHead(32 + 16)
        
    def forward(self, x_temporal, x_static):
        # x_temporal: (batch, 6, 3) -> (batch, channels, timesteps)
        x_temporal = x_temporal.transpose(1, 2)
        static_emb = self.static_branch(x_static)
        temporal_emb = self.conv(x_temporal).squeeze(-1)
        fused = torch.cat([temporal_emb, static_emb], dim=1)
        return self.fusion(fused)

class Conv1DDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_branch = SharedStaticBranch()
        self.conv = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fusion = SharedFusionHead(64 + 16)
        
    def forward(self, x_temporal, x_static):
        x_temporal = x_temporal.transpose(1, 2)
        static_emb = self.static_branch(x_static)
        temporal_emb = self.conv(x_temporal).squeeze(-1)
        fused = torch.cat([temporal_emb, static_emb], dim=1)
        return self.fusion(fused)

class Conv1DMultiScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.static_branch = SharedStaticBranch()
        
        self.branch_a = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.branch_b = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        
        self.fusion = SharedFusionHead(64 + 16)
        
    def forward(self, x_temporal, x_static):
        x_temporal = x_temporal.transpose(1, 2)
        static_emb = self.static_branch(x_static)
        
        out_a = self.branch_a(x_temporal).squeeze(-1)
        out_b = self.branch_b(x_temporal).squeeze(-1)
        temporal_emb = torch.cat([out_a, out_b], dim=1)
        
        fused = torch.cat([temporal_emb, static_emb], dim=1)
        return self.fusion(fused)

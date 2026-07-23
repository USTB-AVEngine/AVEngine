"""Local v4_3 Progressive Refinement architecture for this experiment.

Derived from the sibling Spatial repository's
``v4_3_new_IPD_Enhancer/model/model_TPEech_progressive_refinement.py``.
AVEngine-specific development must use this local copy so the dedicated model
subdirectory is self-contained.
"""

# 方案C: 渐进式特征精炼
# 理论基础：人类听觉处理是层次化的，从基础特征提取到特征整合再到高层语义理解
# 当前concat相当于"一锅烩"，缺乏层次性。本方案通过渐进式精炼实现层次化处理

import torch.nn.functional as F
from torch import nn
import torch
import laion_clap
import librosa
import warnings
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention

warnings.filterwarnings('ignore')

class GlobalLayerNorm(nn.Module):
    '''
       Calculate Global Layer Normalization
       dim: (int or list or torch.Size) –
          input shape from an expected input of size
       eps: a value added to the denominator for numerical stability.
       elementwise_affine: a boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).
    '''

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x-mean)**2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = self.weight*(x-mean)/torch.sqrt(var+self.eps)+self.bias
            else:
                x = (x-mean)/torch.sqrt(var+self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    '''
       Calculate Cumulative Layer Normalization
       dim: you want to norm dim
       elementwise_affine: learnable per-element affine parameters
    '''

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8)

    def forward(self, x):
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
           x = x.permute(0, 2, 3, 1).contiguous()
           # N x K x S x C == only channel norm
           x = super().forward(x)
           # N x C x K x S
           x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    if norm == 'gln':
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == 'cln':
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == 'ln':
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)

class Encoder(nn.Module):
    '''
       Conv-Tasnet Encoder part
       kernel_size: the length of filters
       out_channels: the number of filters
    '''

    def __init__(self, kernel_size=2, out_channels=64):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=out_channels,
                                kernel_size=kernel_size, stride=kernel_size//2, groups=1, bias=False)

    def forward(self, x):
        """
          Input:
              x: [B, T], B is batch size, T is times
          Returns:
              x: [B, C, T_out], T_out is the number of time steps
        """
        # 检查输入
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Encoder input abnormal: nan=", torch.isnan(x).any().item(), "inf=", torch.isinf(x).any().item())

        # B x T -> B x 1 x T
        x = torch.unsqueeze(x, dim=1)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("After unsqueeze abnormal: nan=", torch.isnan(x).any().item(), "inf=", torch.isinf(x).any().item())

        # B x 1 x T -> B x C x T_out
        x = self.conv1d(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("After conv1d abnormal: nan=", torch.isnan(x).any().item(), "inf=", torch.isinf(x).any().item())
            print("conv1d weight nan:", torch.isnan(self.conv1d.weight).any().item(), "inf:", torch.isinf(self.conv1d.weight).any().item())

        # ReLU 激活
        x = F.relu(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("After relu abnormal: nan=", torch.isnan(x).any().item(), "inf=", torch.isinf(x).any().item())

        return x


class CueEncoder(nn.Module):
    '''
        Cue encoder for ours
    '''

    def __init__(self, CLAP):
        super(CueEncoder, self).__init__()
        self.CLAP = CLAP

    def forward(self, cue):
        with torch.no_grad():
            self.CLAP.requires_grad_(False)
            cue_emb = self.CLAP.get_audio_embedding_from_data(x = cue, use_tensor=True)

            return cue_emb


class TextEncoder(nn.Module):
    '''
        Cue encoder for ours
    '''

    def __init__(self, CLAP):
        super(TextEncoder, self).__init__()
        self.CLAP = CLAP

    def forward(self, cue):
        with torch.no_grad():
            self.CLAP.requires_grad_(False)
            cue_emb = self.CLAP.get_text_embedding(x = cue, use_tensor=True)

            return cue_emb


class Decoder(nn.ConvTranspose1d):
    '''
        Decoder of the TasNet
        This module can be seen as the gradient of Conv1d with respect to its input.
        It is also known as a fractionally-strided convolution
        or a deconvolution (although it is not an actual deconvolution operation).
    '''

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: [B, N, L]
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 3/4D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM', norm='ln',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = select_norm(norm, out_channels, 4)
        self.inter_norm = select_norm(norm, out_channels, 4)
        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        self.inter_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)


    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)

        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class Dual_Path_RNN(nn.Module):
    '''
       Implementation of the Dual-Path-RNN model
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
            num_layers: number of Dual-Path-Block
            K: the length of chunk
            num_spks: the number of speakers
    '''

    def __init__(self,device, in_channels, out_channels, hidden_channels,
                 rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=2):
        super(Dual_Path_RNN, self).__init__()
        self.device = device
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        # self.norm = select_norm(norm, in_channels, 3)
        # self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_rnn = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_rnn.append(Dual_RNN_Block(out_channels, hidden_channels,
                                     rnn_type=rnn_type, norm=norm, dropout=dropout,
                                     bidirectional=bidirectional))

        # self.conv2d = nn.Conv2d(
        #     out_channels, out_channels*num_spks, kernel_size=1)
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
         # gated output layer
        self.output = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                    nn.Tanh()
                                    )
        self.output_gate = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1),
                                         nn.Sigmoid()
                                         )

        self.text_downsample = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )


        # ----Cross attention from X-SepFormer----
        self.W_QE = nn.Linear(out_channels, out_channels)
        self.W_QZ = nn.Linear(out_channels, out_channels)

        self.W_KE = nn.Linear(out_channels, out_channels)
        self.W_KZ = nn.Linear(out_channels, out_channels)

        self.W_VE = nn.Linear(out_channels, out_channels)
        self.W_VZ = nn.Linear(out_channels, out_channels)

        self.attn = ScaledDotProductAttention(d_model=out_channels,
                                              d_k=out_channels,
                                              d_v=out_channels,
                                              h=1).to(self.device)

    def forward(self, x, text):
        '''
           x: [B, N, L]

        '''
        # to fuse layer
        # # [B, N, L]
        # x = self.norm(x)
        # # [B, N, L]
        # x = self.conv1d(x)
        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        text = self.text_downsample(text)
        text = text.unsqueeze(2).unsqueeze(3)
        text = text.expand(-1, -1, x.shape[2], x.shape[3])
        text = text.permute(0, 2, 3, 1)

        tap_list = []  # 收集每一层 dual_rnn 输出（over_add 前的 4D 特征）[B,N,K,S]

        # [B, N*spks, K, S]
        for i in range(self.num_layers):
            x = self.dual_rnn[i](x)
            tap_list.append(x)  # 保存该层的 4D 特征（未 over_add）

            if i % 2 == 0:
                # -------------- 尝试使用cross-attn来进行记忆提醒 ----------------
                # [bs, N, K, S] -> [bs, K, S, N]
                x = x.permute(0, 2, 3, 1)


                # x, text has same dimension
                bs, K, S, N = x.shape


                Q_pc = self.W_QE(text.reshape(bs*K*S, N)) + self.W_QZ(x.reshape(bs*K*S, N))
                K_pc = self.W_KE(text.reshape(bs*K*S, N)) + self.W_KZ(x.reshape(bs*K*S, N))
                V_pc = self.W_VE(text.reshape(bs*K*S, N)) + self.W_VZ(x.reshape(bs*K*S, N))

                Q_pc = Q_pc.reshape(bs, K*S, N)
                K_pc = K_pc.reshape(bs, K*S, N)
                V_pc = V_pc.reshape(bs, K*S, N)
                # skip-connection
                attn = self.attn(Q_pc, K_pc, V_pc).reshape(bs, K, S, N)
                x = x + attn

                # [bs, K, S, N] -> [bs, N, K, S]
                x = x.permute(0, 3, 1, 2)

                # # -------------- 尝试使用FiLM来进行记忆提醒 ----------------
                # # [bs, N, K, S]
                # x = self.FiLM_list[i](x, text) + x


        x = self.prelu(x)
        # x = self.conv2d(x) # 此处升维
        # [B*spks, N, K, S]
        # B, _, K, S = x.shape
        # x = x.view(B*self.num_spks,-1, K, S)
        # [B*spks, N, L]
        x = self._over_add(x, gap)
        # 下面是简单一维卷积，维度不变
        x = self.output(x)*self.output_gate(x)
        # [spks*B, N, L]
        x = self.end_conv1x1(x)
        # [B*spks, N, L] -> [B, spks, N, L]
        # _, N, L = x.shape
        # x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)
        # [spks, B, N, L]
        # x = x.transpose(0, 1)

        return x, tap_list, gap

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type()).to(self.device)
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type()).to(self.device)
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class FiLM(nn.Module):
    def __init__(self, dim_in=512, hidden_dim=256):
        super(FiLM, self).__init__()
        self.beta = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
            )
        self.gamma = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
            )
        # self.gamma = nn.Linear(dim_in, hidden_dim)

    def forward(self, hidden_state, embed):
        return self.gamma(embed).unsqueeze(-1) * hidden_state + self.beta(embed).unsqueeze(-1)


class FuseLayer(nn.Module):
    def __init__(self,device, in_channels, out_channels, bottle_neck, norm='ln', fuse_type="BERT"):
        super(FuseLayer, self).__init__()
        self.device = device
        self.fuse_type = fuse_type
        self.norm_audio = select_norm(norm, in_channels, 3)

        self.cue_down_sample2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.film1 = FiLM(512, in_channels) # 512为CLAP输出文本的维度
        self.layer_norm1 = nn.GroupNorm(1, in_channels, eps=1e-8)
        self.film2 = FiLM(in_channels, 128)
        self.layer_norm2 = nn.GroupNorm(1, 128, eps=1e-8)
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d1 = nn.Conv1d(in_channels, 128, 1, bias=False)
        self.conv1d2 = nn.Conv1d(128, out_channels, 1, bias=False)

        self.out_channels = out_channels

    def forward(self, x, cue):
        # 有些东西从之前的模块搬过来了，详情看搬过来的那部分的注释
        # [bs, dim, seq_len_a]
        x = self.norm_audio(x)
        # [bs, 256, seq_len]
        x = self.norm(x)
        # [bs, 256, seq_len]
        x = self.film1(x, cue) + x
        x = self.layer_norm1(x)
        # [bs, 128, sqe_len]
        x = self.conv1d1(x)
        # [bs, 128]
        cue = self.cue_down_sample2(cue)
        # [bs, 128, seq_len]
        x = self.film2(x, cue) + x
        x = self.layer_norm2(x)
        # [bs, 64, seq_len]
        x = self.conv1d2(x)
        return x


class LogMelGccExtractor(nn.Module):
    def __init__(self, fs, nfft, hopsize, mel_bins, window, fmin):
        super(LogMelGccExtractor, self).__init__()
        self.fs = fs
        self.nfft = nfft
        self.hopsize = hopsize
        self.mel_bins = mel_bins
        self.window = window
        self.fmin = fmin

        # 初始化 Mel 滤波器
        mel_filters = librosa.filters.mel(sr=fs, n_fft=nfft, n_mels=mel_bins, fmin=fmin)
        self.melW = torch.from_numpy(mel_filters).float()

    def gcc_phat(self, sig, refsig):
        """
        GCC-PHAT 的 PyTorch 实现，保留 batch 维度。

        输入：
            sig (torch.Tensor): 输入信号，形状 [B, T] 或 [T]
            refsig (torch.Tensor): 参考信号，形状 [B, T] 或 [T]
        输出：
            gcc_phat (torch.Tensor): GCC-PHAT 特征，形状 [B, n_frames, mel_bins]
        """
        # 输入处理：确保张量为 [B, T]
        if sig.dim() == 1:
            sig = sig.unsqueeze(0)  # [T] -> [1, T]
        elif sig.dim() == 3:  # [B, T, 1] -> [B, T]
            sig = sig.squeeze(-1)

        if refsig.dim() == 1:
            refsig = refsig.unsqueeze(0)
        elif refsig.dim() == 3:
            refsig = refsig.squeeze(-1)

        # 计算 STFT 参数
        ncorr = 2 * self.nfft - 1
        n_fft = int(2 ** torch.ceil(torch.log2(torch.tensor(ncorr, dtype=torch.float))))
        hop_length = self.hopsize
        window = torch.hann_window(n_fft).to(sig.device) if self.window == 'hann' else torch.ones(n_fft).to(sig.device)

        # STFT 计算
        Px = torch.stft(
            sig,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            pad_mode='reflect',
            return_complex=True
        )  # [B, freq_bins, time_frames]

        Px_ref = torch.stft(
            refsig,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            center=True,
            pad_mode='reflect',
            return_complex=True
        )  # [B, freq_bins, time_frames]

        # 计算频域互相关 R
        R = Px * torch.conj(Px_ref)  # [B, freq_bins, time_frames]
        B, freq_bins, n_frames = R.shape

        # 批量计算 GCC-PHAT
        R = R.permute(0, 2, 1)  # [B, n_frames, freq_bins]
        phase = torch.exp(1j * torch.angle(R))  # [B, n_frames, freq_bins]
        cc = torch.fft.irfft(phase, n=n_fft)  # [B, n_frames, n_fft]
        gcc_phat = torch.cat((cc[..., -self.mel_bins//2:], cc[..., :self.mel_bins//2]), dim=-1)  # [B, n_frames, mel_bins]

        return gcc_phat

    def gcc_phat_stft(self, Px, Px_ref):
        """


        """

        # 计算频域互相关 R
        R = Px * torch.conj(Px_ref)  # [B, freq_bins, time_frames]
        B, freq_bins, n_frames = R.shape

        # 批量计算 GCC-PHAT
        R = R.permute(0, 2, 1)  # [B, n_frames, freq_bins]
        phase = torch.exp(1j * torch.angle(R))  # [B, n_frames, freq_bins]
        cc = torch.fft.irfft(phase, n=self.nfft)  # [B, n_frames, n_fft]
        gcc_phat = torch.cat((cc[..., -self.mel_bins//2:], cc[..., :self.mel_bins//2]), dim=-1)  # [B, n_frames, mel_bins]

        return gcc_phat


# GCC-Phat
class GCC_MLP(nn.Module):
    def __init__(self, device):
        super(GCC_MLP, self).__init__()
        self.device = device

        # gcc_phat (b,t,f)
        self.fc1 = nn.Sequential(
            nn.Linear(96,128),
            nn.BatchNorm1d(251),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(251,256),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(256,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.MLP = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),

            nn.Linear(256, 180),
            nn.Sigmoid()
        )

    def forward(self, gcc):
        x = self.fc1(gcc)
        B,T,F = x.shape
        x = self.fc2(x.permute(0,2,1)).reshape(B,F)
        doa = self.MLP(x)

        return doa


# ------------------------------
# Helper: Depthwise Separable Conv (轻量平滑)
# ------------------------------
class DWSeparableConv2d(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

# ------------------------------
# 新版 IPD Enhancer：Gated FiLM 融合（语义 -> 声学）
# Inputs:
#   target_mag: [B, F, T]
#   mix_ipd:    [B, F, T]   (角度, -pi..pi)
#   ild:        [B, F, T]
#   E:          [B, K, F, T]  (来自 TSE 的 EIE 堆叠; 可为 None 或 K=0)
# Output:
#   ipd_enh:    [B, F, T]
# ------------------------------
class IPDEnhancerGatedFiLM(nn.Module):
    def __init__(self, ipd_extra_channels=0, ac_channels=64, sem_channels=64, film_reduction=4):
        super().__init__()
        self.ipd_extra_channels = ipd_extra_channels
        # 声学分支输入: [target_mag, cos(ipd), sin(ipd), ild] => 4 通道
        self.ac_head = nn.Sequential(
            nn.Conv2d(4, ac_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ac_channels),
            nn.ReLU(inplace=True),
        )
        self.ac_smooth = DWSeparableConv2d(ac_channels, kernel_size=3, padding=1)

        # 语义分支（可选）: 输入通道 K = ipd_extra_channels
        if ipd_extra_channels > 0:
            self.sem_head = nn.Sequential(
                nn.Conv2d(ipd_extra_channels, sem_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(sem_channels),
                nn.ReLU(inplace=True),
            )
            # Gated FiLM: 语义 -> (gamma,beta) + 空间门 M
            hidden = max(sem_channels // film_reduction, 8)
            self.film_gamma = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(sem_channels, hidden, kernel_size=1), nn.ReLU(inplace=True),
                nn.Conv2d(hidden, ac_channels, kernel_size=1),
            )
            self.film_beta = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(sem_channels, hidden, kernel_size=1), nn.ReLU(inplace=True),
                nn.Conv2d(hidden, ac_channels, kernel_size=1),
            )
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(sem_channels, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )
        else:
            # 无语义分支时，用恒等门控
            self.sem_head = None
            self.film_gamma = None
            self.film_beta  = None
            self.spatial_gate = None

        # 从融合后的表征上预测单位圆的相位增量 [Δc, Δs]
        self.out_head = nn.Conv2d(ac_channels, 2, kernel_size=1, bias=True)

    def forward(self, target_mag, mix_ipd, ild, E=None):
        """
        target_mag: [B, F, T]
        mix_ipd:    [B, F, T]
        ild:        [B, F, T]
        E:          [B, K, F, T] or None
        """
        B, F, T = target_mag.shape
        cos_ipd = torch.cos(mix_ipd)
        sin_ipd = torch.sin(mix_ipd)

        # 声学分支: [B, 4, F, T] -> [B, C_ac, F, T]
        ac_in = torch.stack([target_mag, cos_ipd, sin_ipd, ild], dim=1)
        A_ac = self.ac_head(ac_in)
        A_ac = self.ac_smooth(A_ac)  # 轻量局部平滑

        # 语义分支（若有）: [B, K, F, T] -> FiLM + 门控 -> 融合
        if self.sem_head is not None and E is not None and E.numel() > 0:
            A_sem = self.sem_head(E)            # [B, C_sem, F, T]
            gamma = self.film_gamma(A_sem)      # [B, C_ac, 1, 1]
            beta  = self.film_beta(A_sem)       # [B, C_ac, 1, 1]
            M     = self.spatial_gate(A_sem)    # [B, 1, F, T]

            F_fused = A_ac + A_ac               # 残差起点（等价放大一倍的 A_ac，可改为 A_ac 本身）
            F_fused = (1 + M) * (gamma * F_fused + beta)
        else:
            # 退化为仅声学分支
            F_fused = A_ac

        # 预测 Δc, Δs 并做单位圆修正 -> 增强后的 IPD
        delta = self.out_head(F_fused)          # [B, 2, F, T]
        c0 = cos_ipd.unsqueeze(1)               # [B, 1, F, T]
        s0 = sin_ipd.unsqueeze(1)               # [B, 1, F, T]
        c1 = c0 + delta[:, :1, :, :]
        s1 = s0 + delta[:, 1:, :, :]

        ipd_enh = torch.atan2(s1.squeeze(1), c1.squeeze(1))  # [B, F, T]
        return ipd_enh


# ----------------------------------------------------------------------------
# 模块：空间线索提取（ILD / Group-Delay IPD / GCC-PHAT from IPD）
# 说明：
#   - 不依赖额外输入：只用你前面已经传进来的 mix_mag_ch1/ch2 和 mix_ipd
#   - GCC-PHAT 直接用 IPD 构造复谱相位，做 irfft 获得时延谱（不需要复数 STFT）
#   - 输出维度：
#       ILD           : [B, F, T]（F=513）
#       GD-IPD        : [B, F, T]
#       GCC-embed     : [B, T, E]（E = gcc_embed_dim，默认 128）
# ----------------------------------------------------------------------------
class SpatialCues(nn.Module):
    def __init__(self, n_fft=1024, gcc_lag_bins=129, gcc_embed_dim=128, eps=1e-8):
        super().__init__()
        assert gcc_lag_bins % 2 == 1, "gcc_lag_bins 建议用奇数，方便居中裁剪"
        self.n_fft = n_fft
        self.gcc_lag_bins = gcc_lag_bins
        self.gcc_embed_dim = gcc_embed_dim
        self.eps = eps

        # 将 GCC 的 (lag_bins) 压到一个定长嵌入维度（每一帧一个向量）
        self.gcc_proj = nn.Linear(gcc_lag_bins, gcc_embed_dim)

    @torch.no_grad()
    def _center_crop(self, x, L):
        """
        从最后一维中心裁剪到长度 L
        x: [*, N]
        return: [*, L]
        """
        N = x.shape[-1]
        assert L <= N, "裁剪长度 L 不能大于原长度 N"
        start = (N - L) // 2
        end = start + L
        return x[..., start:end]

    def ild(self, mag_l, mag_r):
        """
        ILD = log((mag_l + eps) / (mag_r + eps))
        输入: [B, F, T]
        输出: [B, F, T]
        """
        return torch.log((mag_l + self.eps) / (mag_r + self.eps))

    def group_delay_ipd(self, ipd):
        """
        沿频率轴对 IPD 做差分（中心差分近似）并用 atan2 包回 [-pi,pi]
        输入: [B, F, T]
        输出: [B, F, T]
        """
        # 前向差分，并保持相位连续性（用复数相乘等价的角度差更稳）
        d = ipd[:, 1:, :] - ipd[:, :-1, :]
        d = torch.atan2(torch.sin(d), torch.cos(d))  # wrap 到 [-pi, pi]
        # 在频率轴补一行零，使得输出恢复到 F 维
        d = F.pad(d, (0, 0, 1, 0))  # pad 在频率维前面补 1 行
        return d

    def gcc_from_ipd(self, ipd):
        """
        由 IPD 构造 GCC-PHAT：phase-only 逆傅里叶
        输入: ipd [B, F, T]
        输出: gcc_embed [B, T, E]
        """
        # 调整为 [B, T, F] 以便在频率轴做 irfft
        ipd_btf = ipd.permute(0, 2, 1)  # [B, T, F]
        # 构造单位幅度的复数谱：exp(j*ipd)
        Z = torch.polar(torch.ones_like(ipd_btf), ipd_btf)  # complex64/complex32
        # 在频率轴做 irfft，得到时延域互相关：[B, T, n_fft]
        cc = torch.fft.irfft(Z, n=self.n_fft, dim=-1)  # 实数
        # 中心裁剪到指定 lag_bins（对称，含 0 时延）
        cc_cropped = self._center_crop(cc, self.gcc_lag_bins)  # [B, T, L]
        # 线性投影到固定嵌入维（每一帧一个向量）
        gcc_embed = self.gcc_proj(cc_cropped)  # [B, T, E]
        return gcc_embed

    def forward(self, mix_mag_ch1, mix_mag_ch2, mix_ipd):
        """
        输入:
            mix_mag_ch1, mix_mag_ch2: [B, F, T]
            mix_ipd: [B, F, T]
        输出:
            ild:      [B, F, T]
            gd_ipd:   [B, F, T]
            gcc_emb:  [B, T, E]
        """
        ild = self.ild(mix_mag_ch1, mix_mag_ch2)                 # [B, F, T]
        gd_ipd = self.group_delay_ipd(mix_ipd)                   # [B, F, T]
        gcc_emb = self.gcc_from_ipd(mix_ipd)                     # [B, T, E]
        return ild, gd_ipd, gcc_emb


# ----------------------------------------------------------------------------
# 模块2: TCN 核心模块 (与之前相同)
# ----------------------------------------------------------------------------
class TemporalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.PReLU()
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x
        y = self.conv(x)
        y = y.permute(0, 2, 1)
        y = self.norm(y)
        y = y.permute(0, 2, 1)
        y = self.act(y)
        if self.res_conv is not None:
            residual = self.res_conv(residual)
        return y + residual

class TCNStack(nn.Module):
    def __init__(self, in_ch, channels, n_layers=6, kernel_size=3):
        super().__init__()
        layers = []
        current_ch = in_ch
        for i in range(n_layers):
            dilation = 2**(i % 4)
            layers.append(TemporalConvBlock(current_ch, channels, kernel_size=kernel_size, dilation=dilation))
            current_ch = channels
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----------------------------------------------------------------------------
# 方案C: 渐进式特征精炼
# 理论基础：人类听觉处理是层次化的，从基础特征提取到特征整合再到高层语义理解
# 当前concat相当于"一锅烩"，缺乏层次性。本方案通过渐进式精炼实现层次化处理
# 改进：每层都包含 depthwise conv（局部上下文）→ SE 通道权重 → residual
# ----------------------------------------------------------------------------
class FeatureRefinementBlock(nn.Module):
    """
    渐进式特征精炼模块（Stage-wise）
    每层结构：Depthwise Conv → SE 通道权重 → Residual
    让每一层都略微改变统计特性，形成真正的 stage 感觉
    """
    def __init__(self, channels, reduction_ratio=4, kernel_size=3, dilation=1):
        super().__init__()
        reduced_channels = max(channels // reduction_ratio, 8)

        # 1. Depthwise Conv: 捕获局部上下文（轻量级）
        # 使用分组卷积实现 depthwise，groups=channels 表示每个通道独立卷积
        pad = (kernel_size - 1) * dilation // 2
        self.depthwise_conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=channels,  # depthwise: 每个通道独立卷积
            bias=False
        )
        self.dw_norm = nn.BatchNorm1d(channels)
        self.dw_act = nn.PReLU()

        # 2. SE 通道权重: 全局通道注意力
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid()
        )

        # 3. 可选的 pointwise conv（进一步混合通道信息，可选）
        # 如果不需要可以跳过，这里保留以增强表达能力
        self.pointwise_conv = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.pw_norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        """
        输入: [B, C, T]
        输出: [B, C, T]

        流程：
        1. Depthwise Conv: 捕获局部时间上下文
        2. SE: 全局通道注意力
        3. Residual: 保持信息流
        """
        residual = x

        # Stage 1: Depthwise Conv（局部上下文）
        x = self.depthwise_conv(x)  # [B, C, T]
        x = self.dw_norm(x)
        x = self.dw_act(x)

        # Stage 2: SE 通道权重（全局通道注意力）
        # Squeeze
        squeezed = self.squeeze(x).squeeze(-1)  # [B, C]
        # Excitation
        excited = self.excitation(squeezed)  # [B, C]
        # Scale
        excited = excited.unsqueeze(-1)  # [B, C, 1]
        x = x * excited  # [B, C, T]

        # Stage 3: Pointwise Conv（可选，进一步混合通道）
        x = self.pointwise_conv(x)
        x = self.pw_norm(x)

        # Residual connection
        x = x + residual  # [B, C, T]

        return x

# ----------------------------------------------------------------------------
# 方案C: 集成渐进式特征精炼的最终 DoA 模型（仅保留 ILD）
# ----------------------------------------------------------------------------
class DoA_TCN_GRU_with_Progressive_Refinement(nn.Module):
    def __init__(self,
                 target_mag_dim=513,
                 enhanced_ipd_dim=513,
                 output_dim=180,
                 tcn_channels=128,
                 tcn_layers=6,
                 gru_hidden=128,
                 input_time_frames=251,
                 output_time_frames=75,
                 # === 保留参数 ===
                 ipd_extra_channels=0,  # = K (来自 TSE 的 EIE 通道数)
                 # === 渐进式精炼参数 ===
                 refinement_channels=256,  # 精炼模块的通道数
                 num_refinement_layers=2,   # 精炼层数
                 refinement_reduction=4     # SE模块的reduction ratio
                 ):
        super().__init__()

        # 新版 IPD 增强器（语义->声学 Gated FiLM；EIE 通道=ipd_extra_channels）
        self.ipd_extra_channels = ipd_extra_channels
        self.ipd_enhancer = IPDEnhancerGatedFiLM(
            ipd_extra_channels=ipd_extra_channels,
            ac_channels=64, sem_channels=64, film_reduction=4
        )

        # 计算基础特征维度: target_mag + enhanced_ipd + ild
        base_feat_dims = [target_mag_dim, enhanced_ipd_dim, target_mag_dim]
        self.feat_dim = sum(base_feat_dims)

        # 第一阶段：基础concat (保持当前方式)
        self.concat_proj = nn.Linear(self.feat_dim, refinement_channels)

        # 第二阶段：渐进式特征精炼
        # 每层使用不同的 dilation，让感受野逐渐扩大，形成真正的 stage 层次
        self.refinement_blocks = nn.ModuleList([
            FeatureRefinementBlock(
                refinement_channels,
                refinement_reduction,
                kernel_size=3,
                dilation=2**i  # 第 i 层使用 dilation=2^i，感受野逐渐扩大
            )
            for i in range(num_refinement_layers)
        ])

        # 主干：TCN -> GRU
        self.tcn = TCNStack(refinement_channels, tcn_channels, n_layers=tcn_layers)
        self.gru = nn.GRU(tcn_channels, gru_hidden, batch_first=True, bidirectional=True)

        # 时间对齐：251 -> 75
        self.time_align = nn.Linear(input_time_frames, output_time_frames)

        # 输出头
        self.fc_out = nn.Linear(gru_hidden * 2, output_dim)  # [B, 75, 180]
        self.fc_card = nn.Linear(gru_hidden * 2, 3)          # [B, 75, 3]

    @staticmethod
    def compute_ild(mix_mag_ch1, mix_mag_ch2, eps=1e-8):
        # ILD = log((|S1|+eps)/( |S2|+eps ))
        return torch.log( (mix_mag_ch1 + eps) / (mix_mag_ch2 + eps) )

    def forward(self, target_mag, mix_mag_ch1, mix_mag_ch2, mix_ipd, dprnn_stack=None):
        """
        target_mag:   [B, F, T]  —— 分离器的目标幅度 (左声道)
        mix_mag_ch1:  [B, F, T]
        mix_mag_ch2:  [B, F, T]
        mix_ipd:      [B, F, T]  —— 原始 IPD (角度)
        dprnn_stack:  [B, K, F, T] or None —— EIE 堆叠 (语义分支)
        """
        # 1) 计算 ILD
        ild = self.compute_ild(mix_mag_ch1, mix_mag_ch2)  # [B, F, T]

        # 2) 新版 IPD 增强（Gated FiLM 融合语义）
        enhanced_ipd = self.ipd_enhancer(
            target_mag=target_mag,
            mix_ipd=mix_ipd,
            ild=ild,
            E=dprnn_stack
        )  # [B, F, T]

        # 3) 第一阶段：基础特征concat
        concat_list = [
            target_mag.permute(0, 2, 1),     # [B, T, F]
            enhanced_ipd.permute(0, 2, 1),   # [B, T, F]
            ild.permute(0, 2, 1)             # [B, T, F]
        ]

        fused_feats = torch.cat(concat_list, dim=-1)  # [B, T, feat_dim]

        # 4) 投影到统一维度
        x = self.concat_proj(fused_feats)  # [B, T, refinement_channels]
        x = x.permute(0, 2, 1)             # [B, C, T]

        # 5) 第二阶段：渐进式特征精炼
        for refinement_block in self.refinement_blocks:
            x = refinement_block(x)  # [B, C, T]

        # 6) TCN + GRU 主干
        x = self.tcn(x)
        x = x.permute(0, 2, 1)               # [B, T, C]
        gru_out, _ = self.gru(x)             # [B, 251, 2H]

        # 7) 时间对齐：251 -> 75
        gru_out_permuted = gru_out.permute(0, 2, 1)       # [B, 2H, 251]
        aligned_out = self.time_align(gru_out_permuted)   # [B, 2H, 75]
        aligned_out_permuted = aligned_out.permute(0, 2, 1)  # [B, 75, 2H]

        # 8) 输出
        doa_logits = self.fc_out(aligned_out_permuted)    # [B, 75, 180]
        doa_prob   = torch.sigmoid(doa_logits)

        card_logits = self.fc_card(aligned_out_permuted)  # [B, 75, 3]
        card_prob   = torch.softmax(card_logits, dim=-1)

        return doa_prob, card_prob


class TPEech_Progressive_Refinement(nn.Module):
    '''
       model of TPEech network with Progressive Feature Refinement
       方案C: 渐进式特征精炼
    '''
    def __init__(self, in_channels, out_channels, hidden_channels, gpuid,
                 kernel_size=2, rnn_type='LSTM', norm='ln', dropout=0,
                 bidirectional=False, num_layers=4, K=200, num_spks=2,
                 max_text_length=512, fuse_type='BERT', clap_checkpoint=None):
        super(TPEech_Progressive_Refinement,self).__init__()
        if isinstance(gpuid, list):
            self.device = torch.device(f'cuda:{gpuid[0]}')
        elif isinstance(gpuid, int):
            self.device = torch.device(f'cuda:{gpuid}')
        elif isinstance(gpuid, torch.device):
            self.device = gpuid
        else:
            raise("Error for gpuid!")
        self.gpuid = gpuid
        self.fuse_type = fuse_type

        # ----- CLAP ------
        if clap_checkpoint is None:
            raise ValueError("clap_checkpoint is required")
        self.CLAP = laion_clap.CLAP_Module(enable_fusion=False, device=self.device)
        self.CLAP.load_ckpt(str(clap_checkpoint))
        for param in self.CLAP.parameters():
            param.requires_grad = False

        # -----cue encoder------
        self.text_encoder = TextEncoder(self.CLAP)
        self.cue_encoder = CueEncoder(self.CLAP)
        self.cue_fc = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        self.text_fc = nn.Sequential(
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.fuse_layer = FuseLayer(self.device, in_channels, out_channels, out_channels, fuse_type=self.fuse_type) # bottle_neck 暂时和out_channels 一样

        self.separation = Dual_Path_RNN(self.device, in_channels, out_channels, hidden_channels,
                 rnn_type=rnn_type, norm=norm, dropout=dropout,
                 bidirectional=bidirectional, num_layers=num_layers, K=K, num_spks=num_spks)
        self.num_spks = num_spks
        self._feature = None

        self.audio_extractor = LogMelGccExtractor(
            fs=16000,
            nfft=1024,
            hopsize=256,
            mel_bins=96,
            window='hann',
            fmin=0
        )
        self.gcc_mlp = GCC_MLP(self.device)

        self.encoder_mlp = nn.Sequential(
            nn.Conv1d(513, 256, kernel_size=1),
            nn.ReLU()
        )


        self.decoder_mlp = nn.Sequential(
            nn.Conv1d(256, 513, kernel_size=1),
            nn.ReLU()
        )

        # 取最后 K 条 tap（建议先用 K=2）；把 DPRNN 的通道 N 投到频点 513
        self.dprnn_tap_k = 2
        self.dprnn_tap_to_spec = nn.ModuleList([
            nn.Conv1d(out_channels, 513, kernel_size=1) for _ in range(self.dprnn_tap_k)
        ])

        self.chunk_size = 64000
        self.n_fft = 1024
        self.hop_length = self.n_fft // 4

        self.doa_estimator = DoA_TCN_GRU_with_Progressive_Refinement(
            target_mag_dim=513,
            enhanced_ipd_dim=513,
            output_dim=180,
            tcn_channels=256,
            tcn_layers=8,
            gru_hidden=256,
            input_time_frames=251,
            output_time_frames=75,
            ipd_extra_channels=self.dprnn_tap_k
        )


    def forward(self, x, cue, text):
        """
        TPEech 模型的前向传播。

        Args:
            x (torch.Tensor): 双声道输入音频波形，形状为 [Batch, Length, 2]
            cue (torch.Tensor): 引导音频片段，用于 CueEncoder
            text (list or torch.Tensor): 引导文本，用于 TextEncoder

        Returns:
            tuple: (
                torch.Tensor: 分离出的目标音频波形, [Batch, Length, 2],
                torch.Tensor: DoA 估计概率, [Batch, Time_frames, 180]
            )
        """
        # 1. STFT: 将时域信号转换为频域表示
        # x_stft 是幅度谱，形状: [bs, 2, 513, 251]
        # mix_phase 是相位谱，形状: [bs, 2, 513, 251]
        x_stft, mix_phase = self.compute_stft(x)

        # 2. 准备 DoA 模块所需的原始混合信号特征
        mix_mag_ch1 = x_stft[:, 0, :, :]  # 混合信号左声道幅度 [bs, 513, 251]
        mix_mag_ch2 = x_stft[:, 1, :, :]  # 混合信号右声道幅度 [bs, 513, 251]

        # 计算原始混合信号的 IPD
        ipd = mix_phase[:, 0, :, :] - mix_phase[:, 1, :, :]
        ipd = torch.atan2(torch.sin(ipd), torch.cos(ipd)) # 限制在 [-pi, pi]

        # 3. 目标声音提取 (TSE) 流程
        # 编码器 MLP
        encoded_ch1 = self.encoder_mlp(mix_mag_ch1)  # [bs, 256, 251]
        encoded_ch2 = self.encoder_mlp(mix_mag_ch2)  # [bs, 256, 251]

        # Cue 和 Text 编码与融合
        cue_emb = self.cue_encoder(cue)      # [bs, 512]
        cue_emb = self.cue_fc(cue_emb)       # [bs, 256]
        if torch.is_tensor(text):
            if text.ndim != 2 or text.shape != (x.shape[0], 512):
                raise ValueError(
                    "cached CLAP text embedding must have shape [batch, 512]"
                )
            text_emb = text.to(device=x.device, dtype=x.dtype)
        else:
            text_emb = self.text_encoder(text)   # [bs, 512]
        text_emb = self.text_fc(text_emb)    # [bs, 256]

        # 融合引导信息 (这里使用拼接)
        fused_guidance = torch.cat([cue_emb, text_emb], dim=-1) # [bs, 512]

        # FuseLayer: 将引导信息融合到音频特征中
        feature1 = self.fuse_layer(encoded_ch1, fused_guidance)
        feature2 = self.fuse_layer(encoded_ch2, fused_guidance)

        # 分离网络 (Dual_Path_RNN)
        est_mask1, taps1, gap1 = self.separation(feature1, fused_guidance)
        est_mask2, taps2, gap2 = self.separation(feature2, fused_guidance)

        # 应用掩码
        separated_encoded1 = est_mask1 * encoded_ch1
        separated_encoded2 = est_mask2 * encoded_ch2

        # === 从 DPRNN 抽取最后 K 层（over_add 之前的特征），映射到 [B, F=513, T]，并堆叠成 [B, K, F, T]
        dprnn_specs = []
        for j in range(self.dprnn_tap_k):
            # 取"最后 K 层"的第 j 条（保持层的先后顺序）
            f1_4d = taps1[-self.dprnn_tap_k + j]   # [B,N,K,S] 未 over_add
            f2_4d = taps2[-self.dprnn_tap_k + j]   # [B,N,K,S]

            # 用 DPRNN 自身的 over_add 还原到时间轴 [B,N,L]
            f1 = self.separation._over_add(f1_4d, gap1)
            f2 = self.separation._over_add(f2_4d, gap2)

            # N -> F=513；左右路径做均值，得到 [B,513,T]
            s1 = self.dprnn_tap_to_spec[j](f1)
            s2 = self.dprnn_tap_to_spec[j](f2)
            s  = 0.5 * (s1 + s2)

            # 与混合谱时间帧对齐（T 可能是 251）
            Tmix = mix_mag_ch1.shape[-1]
            if s.shape[-1] > Tmix:
                s = s[..., :Tmix]
            elif s.shape[-1] < Tmix:
                s = F.pad(s, (0, Tmix - s.shape[-1]))

            dprnn_specs.append(s)

        # [B, K, F, T]
        dprnn_stack = torch.stack(dprnn_specs, dim=1)


        # 解码器 MLP，得到分离出的目标声音幅度谱
        # 这是 DoA 模块需要的关键输入之一: target_mag
        target_mag_ch1 = self.decoder_mlp(separated_encoded1)  # [bs, 513, 251]
        target_mag_ch2 = self.decoder_mlp(separated_encoded2)  # [bs, 513, 251]

        # 4. 调用 DoA 估计模块
        # 我们使用分离后的左声道幅度谱作为目标声源的主要参考
        doa, card_prob = self.doa_estimator(
            target_mag_ch1, mix_mag_ch1, mix_mag_ch2, ipd, dprnn_stack
        )


        # 5. 重建时域音频信号
        # 将分离出的双声道幅度谱组合起来
        separated_mag = torch.stack((target_mag_ch1, target_mag_ch2), dim=1) # [bs, 2, 513, 251]

        # 使用原始混合信号的相位进行 iSTFT 重建
        # 这是一个常见的做法，也可以尝试估计目标相位以获得更好效果
        reconstructed_audio = self.compute_istft(separated_mag, mix_phase) # [bs, L, 2]

        return reconstructed_audio, doa, card_prob

    def compute_stft(self, audio):
        """使用PyTorch计算多声道STFT，支持批量维度"""
        # audio: (bs, time, channels) 或 (time, channels) 或 (time,)
        # 输出: magnitude, phase 为 (bs, channels, freq, time) 或 (channels, freq, time)

        audio_tensor = torch.from_numpy(audio).float() if not isinstance(audio, torch.Tensor) else audio
        has_batch = audio_tensor.ndim == 3

        if audio_tensor.ndim == 1:  # 单声道
            audio_tensor = audio_tensor.unsqueeze(0)  # (1, time)
        elif audio_tensor.ndim == 2:  # 多声道 (time, channels)
            audio_tensor = audio_tensor.T  # (channels, time)
        elif audio_tensor.ndim == 3:  # 批量多声道 (bs, time, channels)
            audio_tensor = audio_tensor.transpose(1, 2)  # (bs, channels, time)
            bs, channels, time = audio_tensor.shape
            audio_tensor = audio_tensor.reshape(bs * channels, time)  # (bs * channels, time)

        stft = torch.stft(
            audio_tensor,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft).to(audio_tensor.device),
            return_complex=True
        )  # (bs * channels, freq, time) 或 (channels, freq, time)

        # 恢复批量维度
        if has_batch:
            bs = audio.shape[0]
            channels = audio.shape[2] if audio.ndim == 3 else 1
            stft = stft.view(bs, channels, stft.shape[1], stft.shape[2])  # (bs, channels, freq, time)

        magnitude = stft.abs()
        phase = stft.angle()

        if not has_batch:
            magnitude = magnitude.squeeze(0) if magnitude.shape[0] == 1 else magnitude
            phase = phase.squeeze(0) if phase.shape[0] == 1 else phase

        return magnitude, phase

    def compute_istft(self, magnitude, phase):
        """使用PyTorch计算多声道iSTFT，支持批量维度"""
        has_batch = magnitude.ndim == 4
        if not has_batch:
            magnitude = magnitude.unsqueeze(0)  # (1, channels, freq, time)
            phase = phase.unsqueeze(0)

        bs, channels, freq, time = magnitude.shape
        stft_matrix = magnitude * torch.exp(1j * phase)  # (bs, channels, freq, time)
        stft_matrix = stft_matrix.view(bs * channels, freq, time)

        audio = torch.istft(
            stft_matrix,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft).to(magnitude.device),
            length=self.chunk_size
        )  # (bs * channels, time)

        audio = audio.view(bs, channels, -1).transpose(1, 2)  # (bs, time, channels)
        if not has_batch:
            audio = audio.squeeze(0)
        if audio.shape[-1] == 1 and audio.ndim > 1:
            audio = audio.squeeze(-1)
        return audio

    def get_feature(self):
        return self._feature

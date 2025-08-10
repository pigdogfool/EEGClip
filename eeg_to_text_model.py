import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BertTokenizer
from transformers.modeling_outputs import BaseModelOutput
import math
import numpy as np
from typing import List, Union, Tuple, Optional


class CNNPatchExtractor(nn.Module):
    """Two-layer CNN to convert EEG signals into patches"""
    
    def __init__(self, input_channels=66, patch_dim=256):
        super(CNNPatchExtractor, self).__init__()
        
        # First CNN layer - 调整参数增加patch数量
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=128, 
            kernel_size=32, 
            stride=12,  # 减小stride从16->12，增加输出长度
            padding=8
        )
        self.bn1 = nn.BatchNorm1d(128)
        
        # Second CNN layer - 进一步调整获得更多patches
        self.conv2 = nn.Conv1d(
            in_channels=128, 
            out_channels=patch_dim, 
            kernel_size=16, 
            stride=6,   # 减小stride从8->6，增加输出长度
            padding=4
        )
        self.bn2 = nn.BatchNorm1d(patch_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, 66, 4000)
        
        # First CNN layer
        x = self.conv1(x)  # (batch_size, 128, ~250)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second CNN layer
        x = self.conv2(x)  # (batch_size, patch_dim, ~31)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Transpose to (batch_size, seq_len, patch_dim) for transformer
        x = x.transpose(1, 2)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # pe shape: (max_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class EEGTransformerEncoder(nn.Module):
    """Transformer encoder for EEG patches with self-attention"""
    
    def __init__(self, patch_dim=256, num_heads=8, num_layers=6, 
                 ff_dim=1024, dropout=0.1):
        super(EEGTransformerEncoder, self).__init__()
        
        self.patch_dim = patch_dim
        self.pos_encoding = PositionalEncoding(patch_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=patch_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_key_padding_mask=None):
        # x shape: (batch_size, seq_len, patch_dim)
        
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply transformer encoder with masking
        encoded = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        return encoded


class EEGToTextModel(nn.Module):
    """Complete EEG-to-text model with CNN, Transformer Encoder, and BART decoder"""
    
    def __init__(self, bart_model_name="fnlp/bart-base-chinese", 
                 patch_dim=256, freeze_decoder=True,
                 target_channels=66, target_length=4000):
        super(EEGToTextModel, self).__init__()
        
        # 目标维度设置
        self.target_channels = target_channels
        self.target_length = target_length
        
        # CNN patch extractor
        self.patch_extractor = CNNPatchExtractor(
            input_channels=target_channels, 
            patch_dim=patch_dim
        )
        
        # Transformer encoder
        self.transformer_encoder = EEGTransformerEncoder(
            patch_dim=patch_dim,
            num_heads=8,
            num_layers=6
        )
        
        # Load BART-Chinese model
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        
        # Multi-layer projection to gradually transform from EEG features to BART's hidden size
        bart_hidden_size = self.bart.config.d_model
        hidden_dim = 512  # 中间维度
        
        self.projection = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),  # 256 -> 512
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, bart_hidden_size),  # 512 -> 768
            nn.LayerNorm(bart_hidden_size)  # 添加LayerNorm稳定训练
        )
        
        # Freeze decoder weights if specified
        if freeze_decoder:
            for param in self.bart.model.decoder.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(0.1)
    
    def process_batch_text(self, text_batch: List[str], tokenizer, max_length: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理批量文本数据，填充到相同长度
        
        Args:
            text_batch: 文本列表
            tokenizer: 文本tokenizer
            max_length: 最大文本长度(32个字符)
            
        Returns:
            input_ids: (batch_size, max_length)
            attention_mask: (batch_size, max_length)
        """
        encodings = tokenizer(
            text_batch,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        )
        return encodings['input_ids'], encodings['attention_mask']
        
    def process_batch_eeg(self, eeg_batch: Union[List[np.ndarray], List[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
        """
        智能处理批量EEG数据，自动识别shape并进行填充
        
        Args:
            eeg_batch: 可以是以下格式之一:
                      - List[np.ndarray]: 每个元素形状为 (channels, time_points)
                      - List[torch.Tensor]: 每个元素形状为 (channels, time_points)  
                      - torch.Tensor: 形状为 (batch_size, channels, time_points)
        
        Returns:
            processed_eeg: (batch_size, target_channels, target_length)
            original_lengths: 每个样本的原始时间长度（用于内部mask计算）
        """
        # 如果输入已经是标准tensor格式，直接返回
        if isinstance(eeg_batch, torch.Tensor) and len(eeg_batch.shape) == 3:
            batch_size, channels, time_points = eeg_batch.shape
            if channels == self.target_channels and time_points == self.target_length:
                return eeg_batch, [time_points] * batch_size
        
        # 处理list格式的输入
        if isinstance(eeg_batch, list):
            batch_size = len(eeg_batch)
            processed_eeg = torch.zeros(batch_size, self.target_channels, self.target_length)
            original_lengths = []
            
            for i, eeg in enumerate(eeg_batch):
                # 转换为tensor
                if isinstance(eeg, np.ndarray):
                    eeg = torch.from_numpy(eeg).float()
                elif not isinstance(eeg, torch.Tensor):
                    eeg = torch.tensor(eeg).float()
                
                current_channels, current_length = eeg.shape
                original_lengths.append(current_length)
                
                # 处理通道数不匹配的情况
                if current_channels != self.target_channels:
                    if current_channels < self.target_channels:
                        # 填充通道（用零填充）
                        padding_channels = torch.zeros(self.target_channels - current_channels, current_length)
                        eeg = torch.cat([eeg, padding_channels], dim=0)
                    else:
                        # 截断通道（取前target_channels个）
                        eeg = eeg[:self.target_channels, :]
                
                # 处理时间长度不匹配的情况
                if current_length > self.target_length:
                    # 截断时间序列
                    processed_eeg[i] = eeg[:, :self.target_length]
                    original_lengths[i] = self.target_length
                else:
                    # 填充时间序列
                    processed_eeg[i, :, :current_length] = eeg
            
            return processed_eeg, original_lengths
            
        # 处理单个tensor输入但需要调整维度的情况
        elif isinstance(eeg_batch, torch.Tensor):
            if len(eeg_batch.shape) == 3:
                batch_size, current_channels, current_length = eeg_batch.shape
                processed_eeg = torch.zeros(batch_size, self.target_channels, self.target_length)
                original_lengths = [current_length] * batch_size
                
                for i in range(batch_size):
                    eeg = eeg_batch[i]
                    
                    # 处理通道数
                    if current_channels != self.target_channels:
                        if current_channels < self.target_channels:
                            padding_channels = torch.zeros(self.target_channels - current_channels, current_length)
                            eeg = torch.cat([eeg, padding_channels], dim=0)
                        else:
                            eeg = eeg[:self.target_channels, :]
                    
                    # 处理时间长度
                    if current_length > self.target_length:
                        processed_eeg[i] = eeg[:, :self.target_length]
                        original_lengths[i] = self.target_length
                    else:
                        processed_eeg[i, :, :current_length] = eeg
                
                return processed_eeg, original_lengths
        
        raise ValueError(f"不支持的EEG数据格式: {type(eeg_batch)}")
    
    def calculate_patch_lengths(self, original_lengths: List[int]) -> List[int]:
        """
        根据原始EEG长度计算经过CNN后的patch数量
        
        Args:
            original_lengths: 原始EEG时间长度列表
            
        Returns:
            patch_lengths: 每个样本经过CNN后的patch数量
        """
        patch_lengths = []
        
        for length in original_lengths:
            # 确保不超过最大长度
            effective_length = min(length, self.target_length)
            
            # 第一层CNN: kernel=32, stride=12, padding=8
            after_conv1 = (effective_length + 16 - 32) // 12 + 1
            
            # 第二层CNN: kernel=16, stride=6, padding=4  
            after_conv2 = (after_conv1 + 8 - 16) // 6 + 1
            
            # 确保至少有1个patch
            patch_length = max(1, after_conv2)
            patch_lengths.append(patch_length)
        
        return patch_lengths

    def create_eeg_mask(self, patch_lengths: List[int], max_patches: int) -> torch.Tensor:
        """根据patch长度创建EEG mask"""
        batch_size = len(patch_lengths)
        mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)
        
        for i, length in enumerate(patch_lengths):
            if length < max_patches:
                mask[i, length:] = True
                
        return mask
    
    
    def forward(self, eeg_data, text_data=None, tokenizer=None):
        """
        前向传播：处理EEG-文本配对数据的监督学习
        """
        # 智能处理EEG数据
        processed_eeg, original_lengths = self.process_batch_eeg(eeg_data)
        patch_lengths = self.calculate_patch_lengths(original_lengths)
        
        # Extract patches from EEG
        eeg_patches = self.patch_extractor(processed_eeg)  # (batch_size, seq_len, patch_dim)
        seq_len = eeg_patches.size(1)
        
        # Create EEG padding mask
        eeg_mask = self.create_eeg_mask(patch_lengths, seq_len)
        eeg_mask = eeg_mask.to(processed_eeg.device)
        
        # Apply transformer encoder with masking
        encoded_eeg = self.transformer_encoder(
            eeg_patches, 
            src_key_padding_mask=eeg_mask
        )
        
        # Project to BART's hidden dimension
        projected_eeg = self.projection(encoded_eeg)
        projected_eeg = self.dropout(projected_eeg)
        
        # 处理文本数据
        if text_data is not None and tokenizer is not None:
            # 使用提供的文本数据，填充到32个字符
            target_ids, attention_mask = self.process_batch_text(text_data, tokenizer, max_length=32)
            target_ids = target_ids.to(processed_eeg.device)
            attention_mask = attention_mask.to(processed_eeg.device)
        
            # Prepare inputs for BART decoder
            decoder_input_ids = target_ids[:, :-1]  # Exclude last token for causal LM
            labels = target_ids[:, 1:]  # Exclude first token (shift for causal LM)
            
            # Create attention mask for decoder input (causal mask)
            if attention_mask is not None:
                decoder_attention_mask = attention_mask[:, :-1]
                label_mask = attention_mask[:, 1:]
                # Set ignored tokens to -100 for loss calculation
                labels = labels.masked_fill(~label_mask.bool(), -100)
            else:
                decoder_attention_mask = None
            
            # Create proper encoder outputs object for BART
            encoder_outputs = BaseModelOutput(
                last_hidden_state=projected_eeg,
                hidden_states=None,
                attentions=None
            )
            
            # Use full BART model with encoder outputs - this automatically calculates loss
            outputs = self.bart(
                encoder_outputs=encoder_outputs,
                attention_mask=~eeg_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
                return_dict=True
            )
            
            return outputs
        else:
            # Inference mode: 只返回EEG编码用于生成
            return projected_eeg
    
    def generate(self, eeg_data, max_length=32, num_beams=4, 
                 early_stopping=True, **generation_kwargs):
        """从EEG数据生成对应的文本"""
        
        with torch.no_grad():
            processed_eeg, original_lengths = self.process_batch_eeg(eeg_data)
            patch_lengths = self.calculate_patch_lengths(original_lengths)
            
            # Extract patches from EEG
            eeg_patches = self.patch_extractor(processed_eeg)
            seq_len = eeg_patches.size(1)
            
            # Create EEG padding mask
            eeg_mask = self.create_eeg_mask(patch_lengths, seq_len)
            eeg_mask = eeg_mask.to(processed_eeg.device)
            
            # Apply transformer encoder with masking
            encoded_eeg = self.transformer_encoder(
                eeg_patches, 
                src_key_padding_mask=eeg_mask
            )
            
            # Project to BART's hidden dimension
            projected_eeg = self.projection(encoded_eeg)
            projected_eeg = self.dropout(projected_eeg)
            
            # Create proper encoder outputs object for BART generation
            encoder_outputs = BaseModelOutput(
                last_hidden_state=projected_eeg,
                hidden_states=None,
                attentions=None
            )
            
            generated_ids = self.bart.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=~eeg_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=early_stopping,
                pad_token_id=self.bart.config.pad_token_id,
                eos_token_id=self.bart.config.eos_token_id,
                **generation_kwargs
            )
            
            return generated_ids


def create_model(bart_model_name="fnlp/bart-base-chinese", 
                patch_dim=256, freeze_decoder=True,
                target_channels=66, target_length=4000):
    """Factory function to create the EEG-to-text model"""
    
    model = EEGToTextModel(
        bart_model_name=bart_model_name,
        patch_dim=patch_dim,
        freeze_decoder=freeze_decoder,
        target_channels=target_channels,
        target_length=target_length
    )
    
    return model


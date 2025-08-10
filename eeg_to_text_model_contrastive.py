import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BertTokenizer, BertModel
from transformers.modeling_outputs import BaseModelOutput
import math
import numpy as np


class CNNPatchExtractor(nn.Module):
    """Two-layer CNN to convert EEG signals into patches"""
    
    def __init__(self, input_channels=66, patch_dim=256):
        super(CNNPatchExtractor, self).__init__()
        
        # First CNN layer
        self.conv1 = nn.Conv1d(
            in_channels=input_channels, 
            out_channels=128, 
            kernel_size=32, 
            stride=12,
            padding=8
        )
        self.bn1 = nn.BatchNorm1d(128)
        
        # Second CNN layer
        self.conv2 = nn.Conv1d(
            in_channels=128, 
            out_channels=patch_dim, 
            kernel_size=16, 
            stride=6,
            padding=4
        )
        self.bn2 = nn.BatchNorm1d(patch_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
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
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class EEGTransformerEncoder(nn.Module):
    """Transformer encoder for EEG patches"""
    
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
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        encoded = self.transformer_encoder(
            x, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        return encoded


class TextEncoder(nn.Module):
    """文本编码器，使用BART编码器提取文本特征"""
    
    def __init__(self, bart_model_name="fnlp/bart-base-chinese", hidden_dim=768):
        super(TextEncoder, self).__init__()
        
        # ✅ 使用BART编码器而不是BERT
        bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.bart_encoder = bart_model.model.encoder
        self.hidden_dim = hidden_dim
        
        # 获取BART的配置
        self.config = bart_model.config
        
        print(f"🔧 文本编码器使用: {bart_model_name}")
        print(f"   BART编码器隐藏维度: {self.config.d_model}")
        
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        
        Returns:
            text_features: (batch_size, hidden_dim) - 句子级别的特征
        """
        if attention_mask is None:
            # 创建attention_mask：非pad_token的位置为1，pad_token的位置为0
            attention_mask = (input_ids != self.config.pad_token_id).long()
        
        # ✅ 使用BART编码器
        encoder_outputs = self.bart_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 获取编码器的输出
        last_hidden_state = encoder_outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        # ✅ 使用attention mask进行平均池化
        if attention_mask is not None:
            # 扩展attention_mask的维度以匹配hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            
            # 对有效token进行加权平均
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)  # (batch_size, hidden_dim)
            sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)  # (batch_size, hidden_dim)
            sentence_features = sum_embeddings / sum_mask  # (batch_size, hidden_dim)
        else:
            # 如果没有mask，直接平均
            sentence_features = torch.mean(last_hidden_state, dim=1)
        
        return sentence_features


class ContrastiveEEGTextModel(nn.Module):
    """对比学习的EEG-文本模型"""
    
    def __init__(self, 
                 bart_model_name="fnlp/bart-base-chinese",
                 patch_dim=256,
                 feature_dim=768,
                 target_channels=66, 
                 target_length=4000,
                 temperature=0.07):
        super(ContrastiveEEGTextModel, self).__init__()
        
        self.target_channels = target_channels
        self.target_length = target_length
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.patch_dim = patch_dim
        print(f"🏗️  初始化模型组件...")
        
        # EEG处理组件
        print(f"  初始化CNN特征提取器...")
        self.patch_extractor = CNNPatchExtractor(
            input_channels=target_channels, 
            patch_dim=patch_dim
        )
        self.eeg_encoder = EEGTransformerEncoder(
            patch_dim=patch_dim,
            num_heads=8,
            num_layers=6
        )
        
        # ✅ 初始化文本编码器并立即冻结
        self.text_encoder = TextEncoder(
            bart_model_name=bart_model_name,
            hidden_dim=feature_dim
        )
        
        # ❄️ 立即冻结文本编码器的所有参数
        print("❄️ 永久冻结文本编码器参数...")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        frozen_text_params = sum(p.numel() for p in self.text_encoder.parameters())
        print(f"🔒 已冻结文本编码器参数: {frozen_text_params:,}")

        # 投影层
        self.eeg_projection = nn.Sequential(
            nn.Linear(patch_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )

        self.text_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
        
        # ✅ 初始化BART生成模型（用于阶段2）
        print(f"  初始化BART生成模型...")
        bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
        self.bart = bart_model
        
        # ❄️ 立即冻结BART编码器（我们用EEG编码器替代）
        print("❄️ 永久冻结BART编码器...")
        for param in self.bart.model.encoder.parameters():
            param.requires_grad = False
        
        frozen_bart_encoder_params = sum(p.numel() for p in self.bart.model.encoder.parameters())
        print(f"🔒 已冻结BART编码器参数: {frozen_bart_encoder_params:,}")

    def configure_for_stage1(self):
        """配置阶段1训练参数：训练EEG编码器，冻结BART解码器"""
        print("🔧 配置阶段1参数状态...")
        
        # ✅ 冻结BART的所有参数
        for param in self.bart.parameters():
            param.requires_grad = False
        print("🔒 冻结BART解码器")
        
        # ✅ 解冻EEG编码器参数
        for param in self.patch_extractor.parameters():
            param.requires_grad = True
        print(f"🔓 解冻CNN特征提取器: {sum(p.numel() for p in self.patch_extractor.parameters()):,} 参数")
        
        for param in self.eeg_encoder.parameters():
            param.requires_grad = True
        print(f"🔓 解冻EEG Transformer编码器: {sum(p.numel() for p in self.eeg_encoder.parameters()):,} 参数")
        
        for param in self.eeg_projection.parameters():
            param.requires_grad = True
        print(f"🔓 解冻EEG投影层: {sum(p.numel() for p in self.eeg_projection.parameters()):,} 参数")
        
        # ❄️ 确保文本编码器保持永久冻结
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        print(f"❄️ 确认文本编码器永久冻结: {sum(p.numel() for p in self.text_encoder.parameters()):,} 参数")
        
        # ✅ 文本投影层需要训练
        for param in self.text_projection.parameters():
            param.requires_grad = True
        print(f"🔓 解冻文本投影层: {sum(p.numel() for p in self.text_projection.parameters()):,} 参数")
        
        # ✅ 统计阶段1预期参数
        stage1_params = (
            sum(p.numel() for p in self.patch_extractor.parameters()) +
            sum(p.numel() for p in self.eeg_encoder.parameters()) +
            sum(p.numel() for p in self.eeg_projection.parameters()) +
            sum(p.numel() for p in self.text_projection.parameters())
        )
        print(f"📊 阶段1预期可训练参数总计: {stage1_params:,}")
    
    def configure_for_stage2(self, freeze_decoder=False, freeze_eeg_encoder=False):
        """配置阶段2训练参数"""
        print("🔧 配置阶段2训练参数...")
        
        if freeze_decoder:
            print("🔒 冻结BART解码器...")
            # 冻结BART解码器和LM头
            for param in self.bart.model.decoder.parameters():
                param.requires_grad = False
            for param in self.bart.lm_head.parameters():
                param.requires_grad = False
            
            print("🔓 解冻EEG编码器组件...")
            # 解冻编码器
            for param in self.patch_extractor.parameters():
                param.requires_grad = True
            for param in self.eeg_encoder.parameters():
                param.requires_grad = True
            for param in self.eeg_projection.parameters():
                param.requires_grad = True
                
        elif freeze_eeg_encoder:
            print("🔒 冻结EEG编码器组件...")
            # 冻结EEG编码器
            for param in self.patch_extractor.parameters():
                param.requires_grad = False
            for param in self.eeg_encoder.parameters():
                param.requires_grad = False
            for param in self.eeg_projection.parameters():
                param.requires_grad = False
                
            print("🔓 解冻BART解码器...")
            # 解冻BART解码器和LM头
            for param in self.bart.model.decoder.parameters():
                param.requires_grad = True
            for param in self.bart.lm_head.parameters():
                param.requires_grad = True
                
        else:
            print("🔓 解冻EEG编码器和BART解码器进行联合训练...")
            # 解冻EEG编码器
            for param in self.patch_extractor.parameters():
                param.requires_grad = True
            for param in self.eeg_encoder.parameters():
                param.requires_grad = True
            for param in self.eeg_projection.parameters():
                param.requires_grad = True
            # 解冻BART解码器和LM头
            for param in self.bart.model.decoder.parameters():
                param.requires_grad = True
            for param in self.bart.lm_head.parameters():
                param.requires_grad = True
        
        # ❄️ 确保BART编码器永远保持冻结（因为我们用EEG编码器替代）
        print("❄️ 确认BART编码器永久冻结...")
        for param in self.bart.model.encoder.parameters():
            param.requires_grad = False
                
        # ❄️ 确保文本编码器的所有组件永远保持冻结
        print("❄️ 确认文本编码器永久冻结...")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        # ❄️ 在阶段2中也冻结文本投影层（保持预训练特征）
        print("❄️ 冻结文本投影层（保持预训练特征）...")
        for param in self.text_projection.parameters():
            param.requires_grad = False
        
        # ✅ 验证冻结状态
        frozen_text_params = sum(p.numel() for p in self.text_encoder.parameters())
        frozen_text_proj_params = sum(p.numel() for p in self.text_projection.parameters())
        frozen_bart_encoder_params = sum(p.numel() for p in self.bart.model.encoder.parameters())
        
        print(f"❄️ 验证冻结状态:")
        print(f"   文本编码器: {frozen_text_params:,} 参数 (永久冻结)")
        print(f"   文本投影层: {frozen_text_proj_params:,} 参数 (阶段2冻结)")
        print(f"   BART编码器: {frozen_bart_encoder_params:,} 参数 (永久冻结)")
        
        # ✅ 统计阶段2预期参数
        if freeze_eeg_encoder:
            stage2_params = (
                sum(p.numel() for p in self.bart.model.decoder.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.bart.lm_head.parameters() if p.requires_grad)
            )
        elif freeze_decoder:
            stage2_params = (
                sum(p.numel() for p in self.patch_extractor.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.eeg_encoder.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.eeg_projection.parameters() if p.requires_grad)
            )
        else:
            stage2_params = (
                sum(p.numel() for p in self.patch_extractor.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.eeg_encoder.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.eeg_projection.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.bart.model.decoder.parameters() if p.requires_grad) +
                sum(p.numel() for p in self.bart.lm_head.parameters() if p.requires_grad)
            )
        print(f"📊 阶段2预期可训练参数总计: {stage2_params:,}")
    
    def get_trainable_parameters_stage2(self, freeze_decoder=False, freeze_eeg_encoder=False):
        """获取阶段2的可训练参数配置"""
        param_groups = []
        
        if freeze_decoder and freeze_eeg_encoder:
            # 错误配置：两者都冻结
            print("❌ 错误配置：编码器和解码器都被冻结!")
            return []
            
        elif freeze_decoder:
            # 冻结BART解码器，只训练EEG编码器
            param_groups = [
                {'params': [p for p in self.patch_extractor.parameters() if p.requires_grad], 
                 'lr': 5e-5, 'name': 'patch_extractor'},
                {'params': [p for p in self.eeg_encoder.parameters() if p.requires_grad], 
                 'lr': 5e-5, 'name': 'eeg_encoder'},
                {'params': [p for p in self.eeg_projection.parameters() if p.requires_grad], 
                 'lr': 1e-4, 'name': 'eeg_projection'},
            ]
            
        elif freeze_eeg_encoder:
            # ✅ 冻结EEG编码器，只训练BART解码器
            param_groups = [
                # ✅ 只训练BART解码器和LM头
                {'params': [p for p in self.bart.model.decoder.parameters() if p.requires_grad], 
                 'lr': 1e-4, 'name': 'bart_decoder'},
                {'params': [p for p in self.bart.lm_head.parameters() if p.requires_grad], 
                 'lr': 2e-4, 'name': 'bart_lm_head'},
            ]
            
        else:
            # 联合训练EEG编码器和BART解码器
            param_groups = [
                # EEG编码器参数
                {'params': [p for p in self.patch_extractor.parameters() if p.requires_grad], 
                 'lr': 2e-5, 'name': 'patch_extractor'},
                {'params': [p for p in self.eeg_encoder.parameters() if p.requires_grad], 
                 'lr': 2e-5, 'name': 'eeg_encoder'},
                {'params': [p for p in self.eeg_projection.parameters() if p.requires_grad], 
                 'lr': 5e-5, 'name': 'eeg_projection'},
                
                # ✅ 只有BART解码器和LM头参与训练
                {'params': [p for p in self.bart.model.decoder.parameters() if p.requires_grad], 
                 'lr': 1e-4, 'name': 'bart_decoder'},
                {'params': [p for p in self.bart.lm_head.parameters() if p.requires_grad], 
                 'lr': 2e-4, 'name': 'bart_lm_head'},
            ]
        
        # 过滤掉空的参数组
        filtered_groups = []
        for group in param_groups:
            if len(group['params']) > 0:
                filtered_groups.append(group)
                print(f"✅ 参数组 '{group['name']}': {len(group['params'])} 个参数, lr={group['lr']}")
        
        # ❄️ 验证文本编码器永久冻结
        text_encoder_trainable = any(p.requires_grad for p in self.text_encoder.parameters())
        if text_encoder_trainable:
            print("❌ 警告：检测到文本编码器中有可训练参数！")
        else:
            print("✅ 确认：文本编码器所有参数已永久冻结")
        
        return filtered_groups

    def process_batch_eeg(self, eeg_batch):
        """处理EEG数据"""
        # 🔧 获取模型所在的设备
        device = next(self.parameters()).device
        
        if isinstance(eeg_batch, list):
            batch_size = len(eeg_batch)
            # 🔧 在正确的设备上创建tensor
            processed_eeg = torch.zeros(batch_size, self.target_channels, self.target_length, device=device)
            original_lengths = []
            
            for i, eeg in enumerate(eeg_batch):
                if isinstance(eeg, np.ndarray):
                    eeg = torch.from_numpy(eeg).float()
                
                # 🔧 确保输入EEG在正确设备上
                eeg = eeg.to(device)
                
                current_channels, current_length = eeg.shape
                original_lengths.append(current_length)
                
                # 处理通道数
                if current_channels != self.target_channels:
                    if current_channels < self.target_channels:
                        # 🔧 在正确设备上创建padding
                        padding_channels = torch.zeros(self.target_channels - current_channels, current_length, device=device)
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
        else:
            # 如果已经是tensor，确保在正确设备上
            batch_size = eeg_batch.size(0)
            eeg_batch = eeg_batch.to(device)
            return eeg_batch, [self.target_length] * batch_size
    
    def calculate_patch_lengths(self, original_lengths):
        """计算patch长度"""
        patch_lengths = []
        for length in original_lengths:
            effective_length = min(length, self.target_length)
            after_conv1 = (effective_length + 16 - 32) // 12 + 1
            after_conv2 = (after_conv1 + 8 - 16) // 6 + 1
            patch_length = max(1, after_conv2)
            patch_lengths.append(patch_length)
        return patch_lengths

    def create_eeg_mask(self, patch_lengths, max_patches):
        """创建EEG mask"""
        # 🔧 获取模型所在的设备
        device = next(self.parameters()).device
        
        batch_size = len(patch_lengths)
        # 🔧 在正确设备上创建mask
        mask = torch.zeros(batch_size, max_patches, dtype=torch.bool, device=device)
        
        for i, length in enumerate(patch_lengths):
            if length < max_patches:
                mask[i, length:] = True
                
        return mask
    
    def encode_eeg(self, eeg_data):
        """编码EEG数据到特征空间"""
        processed_eeg, original_lengths = self.process_batch_eeg(eeg_data)
        patch_lengths = self.calculate_patch_lengths(original_lengths)
        
        # 提取patches
        eeg_patches = self.patch_extractor(processed_eeg)
        seq_len = eeg_patches.size(1)
        
        # 创建mask
        eeg_mask = self.create_eeg_mask(patch_lengths, seq_len)
        eeg_mask = eeg_mask.to(processed_eeg.device)
        
        # Transformer编码
        encoded_eeg = self.eeg_encoder(
            eeg_patches, 
            src_key_padding_mask=eeg_mask
        )
        
        # ✅ 移除平均池化步骤，直接返回序列特征
        return encoded_eeg, eeg_mask, patch_lengths

    def encode_text(self, text_batch, tokenizer, max_length=32):
        """编码文本数据到特征空间"""
        # ✅ 使用BART tokenizer而不是BERT tokenizer
        encodings = tokenizer(
            text_batch,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask'] 
        
        # 移动到正确的设备
        device = next(self.text_encoder.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        # ✅ 使用BART编码器编码文本
        sentence_features = self.text_encoder(
            input_ids, 
            attention_mask=attention_mask 
        )
        
        # 投影到对比学习空间
        text_features = self.text_projection(sentence_features)
        
        return text_features, input_ids, attention_mask
    
    def contrastive_loss(self, encoded_eeg, eeg_mask, patch_lengths, text_features, text_data):
        """计算对比损失 - 考虑重复句子"""
        # 🔧 获取设备
        device = encoded_eeg.device
        
        # EEG特征处理（保持原有逻辑）
        eeg_features = []
        for i, length in enumerate(patch_lengths):
            if length > 0:
                valid_features = encoded_eeg[i, :length, :].mean(dim=0)
            else:
                valid_features = encoded_eeg[i, 0, :]
            eeg_features.append(valid_features)
        
        eeg_features = torch.stack(eeg_features)
        eeg_features = self.eeg_projection(eeg_features)
        
        # L2标准化
        eeg_features = F.normalize(eeg_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(eeg_features, text_features.T) / self.temperature
        batch_size = eeg_features.size(0)
        
        # ✅ 创建考虑重复句子的标签矩阵 - 在正确设备上
        positive_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
        
        # 标记所有正样本对（相同句子）
        for i in range(batch_size):
            for j in range(batch_size):
                if text_data[i] == text_data[j]:  # 相同句子
                    positive_mask[i, j] = True
        
        # ✅ 使用多标签对比损失
        losses = []
        for i in range(batch_size):
            # 对于每个EEG，找到所有正样本
            positive_indices = torch.where(positive_mask[i])[0]
            negative_indices = torch.where(~positive_mask[i])[0]
            
            if len(positive_indices) > 0 and len(negative_indices) > 0:
                # 正样本的logits
                pos_logits = similarity_matrix[i][positive_indices]
                # 负样本的logits  
                neg_logits = similarity_matrix[i][negative_indices]
                
                # InfoNCE损失：log(sum(exp(pos)) / (sum(exp(pos)) + sum(exp(neg))))
                pos_exp = torch.exp(pos_logits)
                neg_exp = torch.exp(neg_logits)
                
                loss_i = -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum()))
                losses.append(loss_i)
        
        if losses:
            contrastive_loss = torch.stack(losses).mean()
        else:
            # 退回到原始方法 - 确保labels在正确设备上
            labels = torch.arange(batch_size, device=device)
            contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        return contrastive_loss
    
    def forward(self, eeg_data, text_data=None, tokenizer=None):
        """
        前向传播 - 支持三种模式
        """
        if text_data is None or tokenizer is None:
            raise ValueError("对比学习模式需要提供text_data和tokenizer")
            
            # ✅ 获取序列特征而不是池化特征
        encoded_eeg, eeg_mask, patch_lengths = self.encode_eeg(eeg_data)
        text_features, input_ids, attention_mask = self.encode_text(text_data, tokenizer)
            
            # ✅ 在对比损失中进行池化
        loss = self.contrastive_loss(encoded_eeg, eeg_mask, patch_lengths, text_features, text_data)
            
        return {
            'loss': loss,
            'encoded_eeg': encoded_eeg,
            'eeg_mask': eeg_mask,
            'patch_lengths': patch_lengths,
            'text_features': text_features
        }

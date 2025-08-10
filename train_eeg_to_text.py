import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging
import json
from datetime import datetime
from pathlib import Path

from eeg_to_text_model_contrastive import create_contrastive_model 

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGTextDataset(Dataset):
    """EEG和文本配对数据集"""
    
    def __init__(self, csv_file, eeg_dir, tokenizer, max_text_length=32):
        """
        Args:
            csv_file: CSV文件路径，包含id和sentence列
            eeg_dir: EEG数据目录路径
            tokenizer: 文本tokenizer
            max_text_length: 最大文本长度
        """
        self.data = pd.read_csv(csv_file)
        self.eeg_dir = eeg_dir
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # ✅ 首先检查EEG目录下的实际文件
        logger.info(f"检查EEG目录: {eeg_dir}")
        eeg_files = [f for f in os.listdir(eeg_dir) if f.endswith('.npy')]
        logger.info(f"找到 {len(eeg_files)} 个EEG文件")
        if len(eeg_files) <= 10:
            logger.info(f"EEG文件示例: {eeg_files}")
        else:
            logger.info(f"前10个EEG文件: {eeg_files[:10]}")
        
        # ✅ 检查CSV数据
        logger.info(f"CSV数据总数: {len(self.data)}")
        logger.info(f"ID范围: {self.data['id'].min()} - {self.data['id'].max()}")
        unique_ids = self.data['id'].nunique()
        logger.info(f"唯一ID数量: {unique_ids}")
        
        # 检查是否有重复ID
        if unique_ids != len(self.data):
            logger.warning(f"发现重复ID! 总行数: {len(self.data)}, 唯一ID: {unique_ids}")
            duplicates = self.data[self.data.duplicated(['id'], keep=False)]
            logger.info(f"重复ID示例: {duplicates['id'].unique()[:10]}")
        
        # 过滤掉没有对应EEG文件的数据
        self.valid_data = []
        not_found_ids = []
        
        for idx, row in self.data.iterrows():
            # ✅ 尝试多种可能的文件名模式
            possible_patterns = [
                f"preprocessed_{row['id']}_raw_300_0328.npy",
                f"preprocessed_{row['id']}.npy",
                f"{row['id']}_raw_300_0328.npy",
                f"{row['id']}.npy",
            ]
            
            eeg_path = None
            for pattern in possible_patterns:
                test_path = os.path.join(eeg_dir, pattern)
                if os.path.exists(test_path):
                    eeg_path = test_path
                    break
            
            if eeg_path:
                self.valid_data.append({
                    'id': row['id'],
                    'sentence': row['sentence'],
                    'eeg_path': eeg_path
                })
            else:
                not_found_ids.append(row['id'])
        
        logger.info(f"找到 {len(self.valid_data)} 个有效的EEG-文本配对")
        logger.info(f"未找到EEG文件的ID数量: {len(not_found_ids)}")
        
        if len(not_found_ids) > 0:
            logger.warning(f"前10个未找到的ID: {not_found_ids[:10]}")
            # 检查这些ID对应的期望文件名
            logger.info("期望的文件名示例:")
            for i, missing_id in enumerate(not_found_ids[:3]):
                expected_file = f"preprocessed_{missing_id}_raw_300_0328.npy"
                logger.info(f"  ID {missing_id} → {expected_file}")
        
        if len(self.valid_data) == 0:
            logger.error("❌ 没有找到任何有效的EEG-文本配对!")
            logger.info("请检查：")
            logger.info("1. EEG文件命名格式是否正确")
            logger.info("2. CSV中的ID是否与EEG文件名匹配")
            logger.info("3. 文件路径是否正确")
            
        # ✅ 检查有效数据的分布
        if len(self.valid_data) > 0:
            valid_ids = [item['id'] for item in self.valid_data]
            logger.info(f"有效ID范围: {min(valid_ids)} - {max(valid_ids)}")
            logger.info(f"有效数据示例:")
            for i, item in enumerate(self.valid_data[:3]):
                logger.info(f"  ID: {item['id']}, 文本: {item['sentence'][:30]}...")
                logger.info(f"  EEG文件: {os.path.basename(item['eeg_path'])}")
    
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        """返回单个数据样本"""
        item = self.valid_data[idx]
        
        # 加载EEG数据
        try:
            eeg_data = np.load(item['eeg_path'])
            eeg_data = eeg_data.astype(np.float32)
            
            # ✅ 检查EEG数据形状
            if idx < 3:  # 只对前几个样本打印信息
                logger.info(f"样本 {idx}: EEG形状 {eeg_data.shape}, 文本: {item['sentence'][:30]}...")
            
            # 每个通道独立标准化
            for ch in range(eeg_data.shape[0]):
                mean = np.mean(eeg_data[ch])
                std = np.std(eeg_data[ch])
                if std > 0:
                    eeg_data[ch] = (eeg_data[ch] - mean) / std
        
        except Exception as e:
            logger.error(f"加载EEG文件失败: {item['eeg_path']}, 错误: {e}")
            eeg_data = np.zeros((66, 4000), dtype=np.float32)
        
        text = item['sentence']
        
        return {
            'eeg': torch.from_numpy(eeg_data),
            'text': text,
            'id': item['id']
        }

def collate_fn(batch):
    """批处理函数，处理不同长度的EEG数据"""
    eeg_list = [item['eeg'] for item in batch]
    text_list = [item['text'] for item in batch]
    ids = [item['id'] for item in batch]
    
    return {
        'eeg': eeg_list,
        'text': text_list,
        'ids': ids
    }

class TwoStageTrainer:
    """两阶段训练器：预训练编码器 → 冻结编码器微调解码器"""
    
    def __init__(self, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        
        # 训练历史记录
        self.stage1_losses = []
        self.stage2_losses = []
    
    def stage1_contrastive_training(self, train_loader, val_loader, epochs=30, 
                                   learning_rate=1e-4, save_path="stage1_encoder.pth"):
        """阶段1: 对比学习训练编码器"""
        print("\n" + "="*60)
        print("🔥 阶段1: 对比学习训练编码器")
        print("⚠️  注意: 此阶段只训练编码器，冻结BART解码器")
        print("="*60)
        
        # ✅ 使用模型的配置方法而不是手动配置
        print("🔧 配置阶段1参数状态...")
        
        # 阶段1配置：训练EEG编码器，冻结BART解码器
        self.model.configure_for_stage1()
        
        # ✅ 收集可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # ✅ 详细打印参数统计
        param_counts = self._analyze_trainable_parameters()
        
        total_trainable = sum(param_counts.values())
        print(f"  ✅ 总可训练参数: {total_trainable:,}")
        
        if total_trainable < 1000:
            logger.error("❌ 可训练参数过少，可能配置有误!")
            return float('inf'), save_path
        
        total_params_count = sum(p.numel() for p in self.model.parameters())
        print(f"  模型总参数: {total_params_count:,}")
        print(f"  阶段1训练比例: {total_trainable/total_params_count*100:.1f}%")
        
        if len(trainable_params) == 0:
            logger.error("❌ 没有找到可训练的参数!")
            return float('inf'), save_path
        
        # 创建优化器
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 训练循环...
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        self.stage1_losses = {'train': [], 'val': []}
        
        for epoch in range(epochs):
            print(f"\n--- Stage1 Epoch {epoch+1}/{epochs} ---")
            
            # 训练
            train_loss = self._train_contrastive_epoch(train_loader, optimizer)
            self.stage1_losses['train'].append(train_loss)
            print(f"训练对比损失: {train_loss:.4f}")
            
            # 验证
            val_loss = self._validate_contrastive_epoch(val_loader)
            self.stage1_losses['val'].append(val_loss)
            print(f"验证对比损失: {val_loss:.4f}")
            
            scheduler.step()
            
            # 早停和保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存编码器状态
                encoder_state = {
                    'patch_extractor': self.model.patch_extractor.state_dict(),
                    'eeg_encoder': self.model.eeg_encoder.state_dict(),
                    'eeg_projection': self.model.eeg_projection.state_dict(),
                    'text_encoder': self.model.text_encoder.state_dict(),
                    'text_projection': self.model.text_projection.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                }
                torch.save(encoder_state, save_path)
                print(f"💾 保存最佳编码器: {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹️ 早停触发")
                    break
        
        print(f"\n✅ 阶段1完成! 最佳验证损失: {best_val_loss:.4f}")
        return best_val_loss, save_path
    
    def stage2_decoder_finetuning(self, train_loader, val_loader, encoder_path, epochs=25, 
                                 learning_rate=2e-5, save_path="stage2_decoder_finetuned.pth"):
        """阶段2: 冻结编码器，微调BART解码器"""
        print("\n" + "="*60)
        print("🔒 阶段2: 冻结EEG编码器，微调BART解码器")
        print("📋 策略: 使用预训练的EEG特征，让BART学会生成对应文本")
        print("="*60)
        
        # 加载预训练的编码器
        encoder_state = torch.load(encoder_path, map_location='cpu')
        self.model.patch_extractor.load_state_dict(encoder_state['patch_extractor'])
        self.model.eeg_encoder.load_state_dict(encoder_state['eeg_encoder'])
        self.model.eeg_projection.load_state_dict(encoder_state['eeg_projection'])
        self.model.text_encoder.load_state_dict(encoder_state['text_encoder'])
        self.model.text_projection.load_state_dict(encoder_state['text_projection'])
        print(f"📥 加载预训练编码器: {encoder_path}")
        
        # ✅ 使用模型的配置方法
        print("🔧 配置阶段2参数状态...")
        self.model.configure_for_stage2(freeze_decoder=False, freeze_eeg_encoder=True)
        
        # ✅ 收集可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        print(f"📊 阶段2可训练参数数量: {len(trainable_params)}")
        
        if len(trainable_params) == 0:
            logger.error("❌ 没有可训练的参数!")
            return float('inf')
        
        # 创建优化器
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        
        # 使用适合微调的调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-7
        )
        
        # 统计参数
        trainable_params_count = sum(p.numel() for p in trainable_params)
        total_params_count = sum(p.numel() for p in self.model.parameters())
        
        print(f"📊 阶段2参数统计:")
        print(f"  总参数: {total_params_count:,}")
        print(f"  可训练参数: {trainable_params_count:,}")
        print(f"  训练比例: {trainable_params_count/total_params_count*100:.1f}%")
        
        # 训练循环...
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        self.stage2_losses = {'train': [], 'val': []}
        
        for epoch in range(epochs):
            print(f"\n--- Stage2 Decoder Finetuning Epoch {epoch+1}/{epochs} ---")
            
            # 训练
            train_loss = self._train_generation_epoch(train_loader, optimizer)
            self.stage2_losses['train'].append(train_loss)
            print(f"训练生成损失: {train_loss:.4f}")
            
            # 验证
            val_loss = self._validate_generation_epoch(val_loader)
            self.stage2_losses['val'].append(val_loss)
            print(f"验证生成损失: {val_loss:.4f}")
            
            # 调度器更新
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"当前学习率: {current_lr:.2e}")
            
            # 每3个epoch测试生成效果
            if (epoch + 1) % 3 == 0:
                print("\n📝 生成样本测试:")
                self._test_generation(val_loader, num_samples=3)
            
            # 早停和保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"💾 保存最佳解码器模型: {save_path}")
            else:
                patience_counter += 1
                print(f"📊 验证损失未改善 ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"⏹️ 早停触发 (微调完成)")
                    break
        
        print(f"\n✅ 阶段2解码器微调完成! 最佳验证损失: {best_val_loss:.4f}")
        return best_val_loss
    
    def _analyze_trainable_parameters(self):
        """分析可训练参数分布"""
        param_counts = {}
        
        # 分析各组件参数
        components = {
            'patch_extractor': self.model.patch_extractor,
            'eeg_encoder': self.model.eeg_encoder,
            'eeg_projection': self.model.eeg_projection,
            'text_encoder': self.model.text_encoder,
            'text_projection': self.model.text_projection,
            'bart_encoder': self.model.bart.model.encoder,
            'bart_decoder': self.model.bart.model.decoder,
            'bart_lm_head': self.model.bart.lm_head,
        }
        
        print(f"\n📊 详细参数分析:")
        for name, component in components.items():
            trainable = sum(p.numel() for p in component.parameters() if p.requires_grad)
            total = sum(p.numel() for p in component.parameters())
            status = "🔓" if trainable > 0 else "🔒"
            print(f"  {status} {name}: {trainable:,} / {total:,} 可训练")
            param_counts[name] = trainable
        
        return param_counts
    
    def _train_contrastive_epoch(self, train_loader, optimizer):
        """阶段1训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Stage1 Contrastive")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            try:
                outputs = self.model(
                    eeg_data=batch['eeg'],
                    text_data=batch['text'],
                    tokenizer=self.tokenizer,
                    mode="contrastive"
                )
                
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"训练批次出错: {e}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def _train_generation_epoch(self, train_loader, optimizer):
        """阶段2训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Stage2 Generation")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            try:
                outputs = self.model(
                    eeg_data=batch['eeg'],
                    text_data=batch['text'],
                    tokenizer=self.tokenizer,
                    mode="generation_training"
                )
                
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"训练批次出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def _validate_contrastive_epoch(self, val_loader):
        """阶段1验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Stage1 Validation"):
                try:
                    outputs = self.model(
                        eeg_data=batch['eeg'],
                        text_data=batch['text'],
                        tokenizer=self.tokenizer,
                        mode="contrastive"
                    )
                    
                    loss = outputs['loss']
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    continue
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def _validate_generation_epoch(self, val_loader):
        """阶段2验证"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Stage2 Validation"):
                try:
                    outputs = self.model(
                        eeg_data=batch['eeg'],
                        text_data=batch['text'],
                        tokenizer=self.tokenizer,
                        mode="generation_training"
                    )
                    
                    loss = outputs['loss']
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception as e:
                    logger.error(f"验证批次出错: {e}")
                    continue
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def _test_generation(self, val_loader, num_samples=3):
        """测试生成效果 - 增强调试信息"""
        self.model.eval()
        samples_generated = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if samples_generated >= num_samples:
                    break
                
                eeg_data = batch['eeg'][:min(num_samples - samples_generated, len(batch['eeg']))]
                original_texts = batch['text'][:len(eeg_data)]
                ids = batch['ids'][:len(eeg_data)]
                
                try:
                    # ✅ 增加调试信息
                    print(f"\n调试信息:")
                    print(f"  批次大小: {len(eeg_data)}")
                    print(f"  EEG数据类型: {type(eeg_data[0])}")
                    if hasattr(eeg_data[0], 'shape'):
                        print(f"  EEG形状: {eeg_data[0].shape}")
                    
                    generated_texts = self.model.generate_text(
                        eeg_data=eeg_data,
                        tokenizer=self.tokenizer,
                        max_length=32,
                        num_beams=4
                    )
                    
                    for i, (gen_text, orig_text, sample_id) in enumerate(zip(generated_texts, original_texts, ids)):
                        print(f"  样本 {samples_generated + i + 1} (ID: {sample_id}):")
                        print(f"    原文: {orig_text}")
                        print(f"    生成: {gen_text}")
                        
                        # ✅ 检查生成文本的特征
                        gen_tokens = gen_text.split()
                        if len(set(gen_tokens)) == 1:  # 所有token都相同
                            print(f"    ⚠️  警告: 生成文本只含单一token!")
                        if len(gen_tokens) < 3:
                            print(f"    ⚠️  警告: 生成文本过短!")
                        
                    samples_generated += len(generated_texts)
                    
                except Exception as e:
                    print(f"  生成失败: {e}")
                    import traceback
                    traceback.print_exc()
                    break
    
    def plot_training_history(self, save_dir):
        """绘制训练历史"""
        plt.figure(figsize=(15, 5))
        
        # 阶段1损失
        if self.stage1_losses['train']:
            plt.subplot(1, 3, 1)
            epochs1 = range(1, len(self.stage1_losses['train']) + 1)
            plt.plot(epochs1, self.stage1_losses['train'], 'b-', label='训练损失', linewidth=2)
            plt.plot(epochs1, self.stage1_losses['val'], 'r-', label='验证损失', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('对比损失')
            plt.title('阶段1: 对比学习训练')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 阶段2损失
        if self.stage2_losses['train']:
            plt.subplot(1, 3, 2)
            epochs2 = range(1, len(self.stage2_losses['train']) + 1)
            plt.plot(epochs2, self.stage2_losses['train'], 'b-', label='训练损失', linewidth=2)
            plt.plot(epochs2, self.stage2_losses['val'], 'r-', label='验证损失', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('生成损失')
            plt.title('阶段2: 生成训练')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # 对比两个阶段
        if self.stage1_losses['val'] and self.stage2_losses['val']:
            plt.subplot(1, 3, 3)
            stage1_min = min(self.stage1_losses['val'])
            stage2_min = min(self.stage2_losses['val'])
            plt.bar(['阶段1\n(对比损失)', '阶段2\n(生成损失)'], [stage1_min, stage2_min], 
                   color=['skyblue', 'lightcoral'], alpha=0.7)
            plt.ylabel('最佳验证损失')
            plt.title('两阶段对比')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = save_dir / 'two_stage_training.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"训练历史图保存到: {plot_path}")

def main():
    """主训练函数"""
    # 配置参数
    config = {
        'csv_file': 'additional_data/subject1_gxy_0328/gxy_0328.csv',
        'eeg_dir': 'additional_data/subject1_gxy_0328/',
        'batch_size': 4,
        'test_size': 0.2,
        'random_state': 42,
        
        # ✅ 修改：跳过阶段1，直接训练阶段2
        'skip_stage1': True,  # 跳过阶段1
        'stage1_model_path': 'two_stage_bart_encoder_20250710_131923/stage1_encoder.pth',  # ✅ 您的阶段1模型路径
        
        # 阶段1参数（如果需要）
        'stage1_epochs': 30,
        'stage1_lr': 1e-4,
        
        # 阶段2参数：冻结编码器，微调解码器
        'stage2_epochs': 25,
        'stage2_lr': 2e-5,  # 稍微小一点的学习率用于微调
        
        'save_dir': f'stage2_only_bart_decoder_{datetime.now().strftime("%Y%m%d_%H%M%S")}'  # ✅ 修改目录名
    }
    
    logger.info("=== EEG-to-Text 阶段2训练（BART解码器微调） ===")
    logger.info("🎯 训练策略:")
    logger.info("  🔄 跳过阶段1，直接加载预训练编码器")
    logger.info("  🔒 阶段2: 冻结EEG编码器，微调BART解码器")
    logger.info("  📝 文本生成器: BART解码器 (fnlp/bart-base-chinese)")
    
    logger.info("配置参数:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # ✅ 检查阶段1模型路径
    if config['skip_stage1']:
        stage1_path = config['stage1_model_path']
        if not os.path.exists(stage1_path):
            logger.error(f"❌ 阶段1模型文件不存在: {stage1_path}")
            logger.info("请确保路径正确，或设置 skip_stage1=False 重新训练阶段1")
            return
        else:
            logger.info(f"✅ 找到阶段1模型: {stage1_path}")
    
    # 检查数据文件
    if not os.path.exists(config['csv_file']):
        logger.error(f"CSV文件不存在: {config['csv_file']}")
        return
    
    if not os.path.exists(config['eeg_dir']):
        logger.error(f"EEG目录不存在: {config['eeg_dir']}")
        return
    
    # ✅ 检查EEG目录内容
    logger.info(f"\n🔍 详细检查EEG目录内容:")
    try:
        files = os.listdir(config['eeg_dir'])
        npy_files = [f for f in files if f.endswith('.npy')]
        logger.info(f"  总文件数: {len(files)}")
        logger.info(f"  .npy文件数: {len(npy_files)}")
        
        if len(npy_files) > 0:
            logger.info(f"  文件名示例: {npy_files[:5]}")
        else:
            logger.error("  ❌ 没有找到.npy文件!")
            return
    except Exception as e:
        logger.error(f"读取EEG目录失败: {e}")
        return
    
    # 创建保存目录
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # ✅ 初始化tokenizer
    logger.info("初始化tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('fnlp/bart-base-chinese')
        logger.info(f"✅ 成功加载tokenizer: {type(tokenizer).__name__}")
    except Exception as e:
        logger.warning(f"AutoTokenizer加载失败: {e}")
        try:
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('fnlp/bart-base-chinese')
            logger.info("✅ 成功加载BertTokenizer (fnlp/bart-base-chinese)")
        except Exception as e2:
            logger.warning(f"BertTokenizer加载失败: {e2}")
            from transformers import BertTokenizer
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            logger.info("✅ 使用备用BertTokenizer (bert-base-chinese)")
    
    # 创建数据集
    logger.info("创建数据集...")
    full_dataset = EEGTextDataset(
        csv_file=config['csv_file'],
        eeg_dir=config['eeg_dir'],
        tokenizer=tokenizer
    )
    
    if len(full_dataset) == 0:
        logger.error("❌ 数据集为空，无法训练!")
        return
    
    logger.info(f"总数据样本: {len(full_dataset)}")
    
    # 分割训练和验证集
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)), 
        test_size=config['test_size'], 
        random_state=config['random_state']
    )
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    logger.info(f"训练样本: {len(train_dataset)}, 验证样本: {len(val_dataset)}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # 创建模型
    logger.info("创建EEG-to-Text模型...")
    logger.info("  📝 文本编码器: BART编码器 (fnlp/bart-base-chinese)")
    logger.info("  📝 文本生成器: BART解码器 (fnlp/bart-base-chinese)")
    logger.info("  🔄 编码器和解码器使用同一个BART模型")
    
    model = create_contrastive_model(
        bart_model_name="fnlp/bart-base-chinese",
        patch_dim=256,
        feature_dim=768,
        target_channels=66,
        target_length=4000,
        temperature=0.07
    )
    
    # 创建训练器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    trainer = TwoStageTrainer(model, tokenizer, device)
    
    # ✅ 根据配置决定是否跳过阶段1
    if config['skip_stage1']:
        logger.info("\n" + "="*60)
        logger.info("🔄 跳过阶段1，直接进入阶段2训练")
        logger.info(f"📥 将使用预训练编码器: {config['stage1_model_path']}")
        logger.info("="*60)
        
        # 直接使用提供的阶段1模型路径
        stage1_encoder_path = config['stage1_model_path']
        stage1_loss = 0.0  # 占位值
        
    else:
        # 如果不跳过，执行阶段1训练
        logger.info("\n" + "="*60)
        logger.info("🔥 开始阶段1: 对比学习预训练EEG编码器")
        logger.info("📋 目标: 学习EEG信号与中文文本的对应关系")
        logger.info("🎯 使用BART编码器进行文本特征提取")
        logger.info("="*60)
        
        stage1_encoder_path = save_dir / 'stage1_encoder.pth'
        stage1_loss, _ = trainer.stage1_contrastive_training(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['stage1_epochs'],
            learning_rate=config['stage1_lr'],
            save_path=stage1_encoder_path
        )
    
    # ✅ 阶段2：冻结编码器，微调解码器
    logger.info("\n" + "="*60)
    logger.info("🔒 开始阶段2: 冻结EEG编码器，微调BART解码器")
    logger.info("📋 目标: 让BART解码器学会从预训练EEG特征生成中文文本")
    logger.info("🎯 冻结BART编码器和EEG编码器，只训练BART解码器")
    logger.info(f"📥 加载预训练编码器: {stage1_encoder_path}")
    logger.info("="*60)
    
    stage2_loss = trainer.stage2_decoder_finetuning(
        train_loader=train_loader,
        val_loader=val_loader,
        encoder_path=stage1_encoder_path,
        epochs=config['stage2_epochs'],
        learning_rate=config['stage2_lr'],
        save_path=save_dir / 'stage2_decoder_finetuned.pth'
    )
    
    # 绘制训练历史（如果有阶段1数据）
    if not config['skip_stage1']:
        trainer.plot_training_history(save_dir)
    else:
        # 只绘制阶段2的训练历史
        if trainer.stage2_losses['train']:
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            epochs2 = range(1, len(trainer.stage2_losses['train']) + 1)
            plt.plot(epochs2, trainer.stage2_losses['train'], 'b-', label='训练损失', linewidth=2)
            plt.plot(epochs2, trainer.stage2_losses['val'], 'r-', label='验证损失', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('生成损失')
            plt.title('阶段2: BART解码器微调')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            stage2_min = min(trainer.stage2_losses['val'])
            plt.bar(['阶段2\n(生成损失)'], [stage2_min], 
                   color=['lightcoral'], alpha=0.7)
            plt.ylabel('最佳验证损失')
            plt.title('训练结果')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = save_dir / 'stage2_training.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.show()
            logger.info(f"阶段2训练历史图保存到: {plot_path}")
    
    # 保存训练结果
    results = {
        'stage1_loss': stage1_loss,
        'stage2_loss': stage2_loss,
        'stage1_history': trainer.stage1_losses if not config['skip_stage1'] else {},
        'stage2_history': trainer.stage2_losses,
        'config': config,
        'training_strategy': 'stage2_only_bart_decoder_finetuning',
        'stage1_model_used': stage1_encoder_path,
        'models': {
            'text_encoder': 'BART编码器 (fnlp/bart-base-chinese)',
            'text_generator': 'BART解码器 (fnlp/bart-base-chinese)',
            'unified_model': 'fnlp/bart-base-chinese'
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = save_dir / 'training_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # ✅ 完成信息
    if config['skip_stage1']:
        logger.info(f"\n🎉 阶段2 BART解码器微调完成!")
        logger.info(f"📥 使用的阶段1编码器: {stage1_encoder_path}")
        logger.info(f"📊 阶段2 (BART解码器微调): {stage2_loss:.4f}")
    else:
        logger.info(f"\n🎉 两阶段BART编码器训练完成!")
        logger.info(f"📊 阶段1 (EEG编码器预训练): {stage1_loss:.4f}")
        logger.info(f"📊 阶段2 (BART解码器微调): {stage2_loss:.4f}")
    
    logger.info(f"📂 结果保存在: {save_dir}")
    logger.info(f"💾 最终模型: {save_dir / 'stage2_decoder_finetuned.pth'}")
    logger.info(f"🔄 使用统一的BART模型: fnlp/bart-base-chinese")
    
    # ✅ 测试最终模型
    logger.info("\n🧪 测试最终模型生成效果:")
    trainer._test_generation(val_loader, num_samples=5)

if __name__ == "__main__":
    main()

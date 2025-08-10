import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging
import json
from datetime import datetime
from pathlib import Path

from eeg_to_text_model_contrastive import ContrastiveEEGTextModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGTextDatasetFullTrain(Dataset):
    """EEG和文本配对数据集 - 全部用于训练"""
    
    def __init__(self, csv_file, eeg_dir, tokenizer, max_text_length=32):
        self.data = pd.read_csv(csv_file)
        self.eeg_dir = eeg_dir
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        
        # 过滤掉没有对应EEG文件的数据
        self.valid_data = []
        not_found_ids = []
        
        for idx, row in self.data.iterrows():
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
        
        logger.info(f"✅ 找到 {len(self.valid_data)} 个有效的EEG-文本配对用于预训练")
        if len(not_found_ids) > 0:
            logger.warning(f"⚠️ 未找到EEG文件的ID数量: {len(not_found_ids)}")
            logger.warning(f"   前10个未找到的ID: {not_found_ids[:10]}")
    
    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        item = self.valid_data[idx]
        
        # 加载EEG数据
        try:
            eeg_data = np.load(item['eeg_path'])
            eeg_data = eeg_data.astype(np.float32)
            
            # 标准化每个通道
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

def collate_fn_full_train(batch):
    """批处理函数 - 全训练集版本"""
    eeg_list = [item['eeg'] for item in batch]
    text_list = [item['text'] for item in batch]
    ids = [item['id'] for item in batch]
    
    return {
        'eeg': eeg_list,
        'text': text_list,
        'ids': ids
    }

class EncoderPretrainer:
    """EEG编码器预训练器"""
    
    def __init__(self, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = tokenizer
        self.device = device
        
        # 训练历史记录
        self.training_losses = []
        self.similarity_scores = []
        self.epoch_times = []
    
    def train_encoder_pretraining(self, train_loader, model, 
                                 epochs=100, learning_rate=1e-4, 
                                 save_path="encoder_pretrained.pth",
                                 eval_every_n_epochs=5):
        """预训练EEG编码器"""
        
        print("\n" + "="*80)
        print("🚀 EEG编码器预训练")
        print("🎯 目标: 通过对比学习预训练EEG编码器")
        print("📚 数据策略: 全部数据用于训练")
        print("🔧 训练组件: EEG编码器 + 文本编码器（冻结）")
        print("="*80)
        
        # 配置阶段1训练参数
        model.configure_for_stage1()
        
        # 分析可训练参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"\n📊 模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {total_trainable:,}")
        print(f"  冻结参数: {total_params - total_trainable:,}")
        print(f"  训练比例: {total_trainable/total_params*100:.1f}%")
        
        # 详细参数分析
        eeg_patch_params = sum(p.numel() for p in model.patch_extractor.parameters())
        eeg_transformer_params = sum(p.numel() for p in model.eeg_encoder.parameters())
        eeg_proj_params = sum(p.numel() for p in model.eeg_projection.parameters())
        text_proj_params = sum(p.numel() for p in model.text_projection.parameters())
        
        print(f"\n🔍 组件参数详情:")
        print(f"  🧠 CNN特征提取器: {eeg_patch_params:,}")
        print(f"  🔄 EEG Transformer: {eeg_transformer_params:,}")
        print(f"  📊 EEG投影层: {eeg_proj_params:,}")
        print(f"  📝 文本投影层: {text_proj_params:,}")
        
        # 优化器和学习率调度器
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-7
        )
        
        # 训练循环
        best_loss = float('inf')
        patience = 25
        patience_counter = 0
        
        for epoch in range(epochs):
            epoch_start_time = datetime.now()
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            
            # 训练
            train_loss = self._train_epoch(model, train_loader, optimizer, epoch+1)
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_end_time = datetime.now()
            epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()
            
            print(f"📈 训练损失: {train_loss:.4f}")
            print(f"⚡ 学习率: {current_lr:.2e}")
            print(f"⏰ 训练时间: {epoch_duration:.1f}秒")
            
            # 记录历史
            self.training_losses.append(train_loss)
            self.epoch_times.append(epoch_duration)
            
            # 定期评估
            if (epoch + 1) % eval_every_n_epochs == 0:
                print("\n🧪 对比学习效果评估:")
                avg_similarity = self._evaluate_similarity(model, train_loader, num_samples=20)
                self.similarity_scores.append(avg_similarity)
                print(f"   同句EEG-文本相似度: {avg_similarity:.4f}")
            
            # 早停和保存
            if train_loss < best_loss:
                best_loss = train_loss
                patience_counter = 0
                
                # 保存最佳模型
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'best_loss': best_loss,
                    'config': {
                        'stage': 1,
                        'training_mode': 'contrastive_pretraining',
                        'patch_dim': model.patch_dim,
                        'feature_dim': model.feature_dim,
                        'target_channels': model.target_channels,
                        'target_length': model.target_length,
                        'temperature': model.temperature
                    },
                    'training_history': {
                        'losses': self.training_losses,
                        'similarities': self.similarity_scores,
                        'epoch_times': self.epoch_times
                    }
                }
                
                # 确保save_path是Path对象
                if isinstance(save_path, str):
                    save_path = Path(save_path)
                
                torch.save(save_dict, save_path)
                print(f"💾 保存最佳模型: {save_path}")
                
            else:
                patience_counter += 1
                print(f"⏳ 早停计数: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"⏹️ 早停触发 - 连续{patience}个epoch无改善")
                    break
        
        print(f"\n✅ 预训练完成! 最佳损失: {best_loss:.4f}")
        print(f"📊 总训练轮数: {len(self.training_losses)}")
        print(f"⏰ 平均每轮时间: {np.mean(self.epoch_times):.1f}秒")
        
        return model, best_loss
    
    def _train_epoch(self, model, train_loader, optimizer, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"🔥 训练 Epoch {epoch}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            try:
                # 🔧 将EEG数据移动到GPU
                if self.device.type == 'cuda':
                    batch['eeg'] = [eeg.to(self.device) for eeg in batch['eeg']]
                
                # 对比学习前向传播
                outputs = model(
                    eeg_data=batch['eeg'],
                    text_data=batch['text'],
                    tokenizer=self.tokenizer
                )
                
                loss = outputs['loss']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss/num_batches:.4f}'
                })
                
            except Exception as e:
                logger.error(f"训练批次出错: {e}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def _evaluate_similarity(self, model, train_loader, num_samples=20):
        """评估同句子EEG-文本相似度"""
        model.eval()
        similarities = []
        samples_processed = 0
        
        with torch.no_grad():
            for batch in train_loader:
                if samples_processed >= num_samples:
                    break
                
                try:
                    # 🔧 将EEG数据移动到GPU
                    if self.device.type == 'cuda':
                        batch['eeg'] = [eeg.to(self.device) for eeg in batch['eeg']]
                    
                    # 前向传播获取特征
                    outputs = model(
                        eeg_data=batch['eeg'],
                        text_data=batch['text'],
                        tokenizer=self.tokenizer
                    )
                    
                    # 获取EEG和文本特征
                    encoded_eeg = outputs['encoded_eeg']
                    text_features = outputs['text_features']
                    patch_lengths = outputs['patch_lengths']
                    
                    # 计算EEG特征（池化）
                    eeg_features = []
                    for i, length in enumerate(patch_lengths):
                        if length > 0:
                            valid_features = encoded_eeg[i, :length, :].mean(dim=0)
                        else:
                            valid_features = encoded_eeg[i, 0, :]
                        eeg_features.append(valid_features)
                    
                    eeg_features = torch.stack(eeg_features)
                    eeg_features = model.eeg_projection(eeg_features)
                    
                    # L2标准化
                    eeg_features = nn.functional.normalize(eeg_features, p=2, dim=1)
                    text_features = nn.functional.normalize(text_features, p=2, dim=1)
                    
                    # 计算对角线相似度（同句子的EEG-文本相似度）
                    batch_similarities = torch.sum(eeg_features * text_features, dim=1)
                    similarities.extend(batch_similarities.cpu().numpy())
                    
                    samples_processed += len(batch['eeg'])
                    
                except Exception as e:
                    logger.error(f"相似度评估出错: {e}")
                    continue
        
        return np.mean(similarities) if similarities else 0.0
    
    def analyze_same_sentence_similarity(self, model, train_loader, save_dir):
        """分析同句子不同EEG的相似度"""
        logger.info("🔍 分析同句子不同EEG样本的相似度...")
        
        model.eval()
        sentence_groups = {}
        
        # 收集所有句子的EEG特征
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="收集特征"):
                try:
                    # 🔧 将EEG数据移动到GPU
                    if self.device.type == 'cuda':
                        batch['eeg'] = [eeg.to(self.device) for eeg in batch['eeg']]
                    
                    outputs = model(
                        eeg_data=batch['eeg'],
                        text_data=batch['text'],
                        tokenizer=self.tokenizer
                    )
                    
                    encoded_eeg = outputs['encoded_eeg']
                    patch_lengths = outputs['patch_lengths']
                    
                    # 计算EEG特征
                    for i, (text, eeg_id) in enumerate(zip(batch['text'], batch['ids'])):
                        length = patch_lengths[i]
                        if length > 0:
                            eeg_feature = encoded_eeg[i, :length, :].mean(dim=0)
                        else:
                            eeg_feature = encoded_eeg[i, 0, :]
                        
                        eeg_feature = model.eeg_projection(eeg_feature.unsqueeze(0)).squeeze(0)
                        eeg_feature = nn.functional.normalize(eeg_feature, p=2, dim=0)
                        
                        if text not in sentence_groups:
                            sentence_groups[text] = []
                        sentence_groups[text].append({
                            'id': eeg_id,
                            'feature': eeg_feature.cpu().numpy()
                        })
                
                except Exception as e:
                    logger.error(f"特征收集出错: {e}")
                    continue
        
        # 分析同句子相似度
        same_sentence_analysis = {}
        all_similarities = []
        
        for sentence, eeg_list in sentence_groups.items():
            if len(eeg_list) > 1:  # 只分析有多个EEG的句子
                similarities = []
                features = [item['feature'] for item in eeg_list]
                
                # 计算两两相似度
                for i in range(len(features)):
                    for j in range(i+1, len(features)):
                        sim = np.dot(features[i], features[j])
                        similarities.append(sim)
                
                if similarities:
                    same_sentence_analysis[sentence] = {
                        'num_eegs': len(eeg_list),
                        'avg_similarity': float(np.mean(similarities)),
                        'std_similarity': float(np.std(similarities)),
                        'min_similarity': float(np.min(similarities)),
                        'max_similarity': float(np.max(similarities)),
                        'all_similarities': [float(s) for s in similarities],
                        'eeg_ids': [item['id'] for item in eeg_list]
                    }
                    all_similarities.extend(similarities)
        
        # 整体统计
        if all_similarities:
            overall_stats = {
                'num_sentences_with_multiple_eegs': len(same_sentence_analysis),
                'total_sentence_pairs': len(all_similarities),
                'avg_same_sentence_similarity': float(np.mean(all_similarities)),
                'std_same_sentence_similarity': float(np.std(all_similarities)),
                'min_same_sentence_similarity': float(np.min(all_similarities)),
                'max_same_sentence_similarity': float(np.max(all_similarities))
            }
        else:
            overall_stats = {
                'num_sentences_with_multiple_eegs': 0,
                'total_sentence_pairs': 0,
                'avg_same_sentence_similarity': 0.0,
                'std_same_sentence_similarity': 0.0,
                'min_same_sentence_similarity': 0.0,
                'max_same_sentence_similarity': 0.0
            }
        
        # 保存分析结果
        analysis_results = {
            'per_sentence_analysis': same_sentence_analysis,
            'overall_statistics': overall_stats,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        analysis_path = save_dir / 'same_sentence_similarity_analysis.json'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        # 打印结果
        logger.info(f"\n📊 同句子相似度分析结果:")
        logger.info(f"  有多个EEG的句子数量: {overall_stats['num_sentences_with_multiple_eegs']}")
        logger.info(f"  总的句子对数量: {overall_stats['total_sentence_pairs']}")
        if overall_stats['total_sentence_pairs'] > 0:
            logger.info(f"  平均相似度: {overall_stats['avg_same_sentence_similarity']:.4f}")
            logger.info(f"  标准差: {overall_stats['std_same_sentence_similarity']:.4f}")
            logger.info(f"  最高相似度: {overall_stats['max_same_sentence_similarity']:.4f}")
            logger.info(f"  最低相似度: {overall_stats['min_same_sentence_similarity']:.4f}")
        
        logger.info(f"📁 详细分析保存到: {analysis_path}")
        
        return analysis_results

def main():
    """主预训练函数"""
    config = {
        'csv_file': 'additional_data/subject1_gxy_0328/gxy_0328.csv',
        'eeg_dir': 'additional_data/subject1_gxy_0328/',
        'batch_size': 32,  # 稍微大一点的batch size用于预训练
        
        # 预训练参数
        'epochs': 150,
        'learning_rate': 1e-4,
        'eval_every_n_epochs': 5,
        
        # 模型参数
        'patch_dim': 256,
        'feature_dim': 768,
        'target_channels': 66,
        'target_length': 4000,
        'temperature': 0.07,
        
        'save_dir': f'encoder_pretrain_full_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    logger.info("=== EEG编码器预训练（全数据集） ===")
    logger.info("🎯 训练策略:")
    logger.info("  📚 对比学习预训练EEG编码器")
    logger.info("  🔒 冻结文本编码器（BART编码器）")
    logger.info("  📊 全部数据用于训练")
    logger.info("  🎯 目标：学习EEG-文本对应关系")
    
    # 检查数据文件
    if not os.path.exists(config['csv_file']):
        logger.error(f"❌ CSV文件不存在: {config['csv_file']}")
        return
    
    if not os.path.exists(config['eeg_dir']):
        logger.error(f"❌ EEG目录不存在: {config['eeg_dir']}")
        return
    
    # 创建保存目录
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_path = save_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"📁 配置保存到: {config_path}")
    
    # 初始化tokenizer
    logger.info("🔧 初始化tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('fnlp/bart-base-chinese')
        logger.info(f"✅ 成功加载BART tokenizer")
    except Exception as e:
        logger.error(f"❌ tokenizer加载失败: {e}")
        return
    
    # 创建数据集
    logger.info("📚 创建全训练数据集...")
    full_dataset = EEGTextDatasetFullTrain(
        csv_file=config['csv_file'],
        eeg_dir=config['eeg_dir'],
        tokenizer=tokenizer
    )
    
    if len(full_dataset) == 0:
        logger.error("❌ 数据集为空，无法训练!")
        return
    
    logger.info(f"✅ 数据集创建完成，总样本数: {len(full_dataset)}")
    
    # 分析数据集
    logger.info("🔍 分析数据集...")
    sentences = [item['sentence'] for item in full_dataset.valid_data]
    unique_sentences = set(sentences)
    logger.info(f"  总句子数: {len(sentences)}")
    logger.info(f"  唯一句子数: {len(unique_sentences)}")
    logger.info(f"  平均每句EEG数: {len(sentences)/len(unique_sentences):.2f}")
    
    # 创建数据加载器
    train_loader = DataLoader(
        full_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn_full_train, 
        num_workers=2,
        drop_last=True  # 确保批次大小一致
    )
    
    logger.info(f"📊 数据加载器创建完成:")
    logger.info(f"  批次大小: {config['batch_size']}")
    logger.info(f"  总批次数: {len(train_loader)}")
    
    # 创建模型
    logger.info("🏗️ 创建对比学习模型...")
    model = ContrastiveEEGTextModel(
        bart_model_name="fnlp/bart-base-chinese",
        patch_dim=config['patch_dim'],
        feature_dim=config['feature_dim'],
        target_channels=config['target_channels'],
        target_length=config['target_length'],
        temperature=config['temperature']
    )
    
    # 移动到设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🖥️ 使用设备: {device}")
    
    # 检查GPU内存
    if device.type == 'cuda':
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"🎮 GPU内存: {gpu_memory:.1f} GB")
    
    model = model.to(device)
    
    # 创建预训练器
    pretrainer = EncoderPretrainer(tokenizer, device)
    
    # 开始预训练
    model_path = save_dir / 'stage1_contrastive_pretrained.pth'
    final_model, best_loss = pretrainer.train_encoder_pretraining(
        train_loader=train_loader,
        model=model,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        save_path=model_path,
        eval_every_n_epochs=config['eval_every_n_epochs']
    )
    
    # 分析同句子相似度
    logger.info("\n" + "="*80)
    logger.info("🔍 同句子EEG相似度分析")
    logger.info("="*80)
    
    # 加载最佳模型进行分析
    best_checkpoint = torch.load(model_path, map_location=device)
    final_model.load_state_dict(best_checkpoint['model_state_dict'])
    
    similarity_analysis = pretrainer.analyze_same_sentence_similarity(
        final_model, train_loader, save_dir
    )
    
    # 绘制训练历史
    if pretrainer.training_losses:
        plt.figure(figsize=(15, 10))
        
        # 训练损失
        plt.subplot(2, 3, 1)
        epochs = range(1, len(pretrainer.training_losses) + 1)
        plt.plot(epochs, pretrainer.training_losses, 'b-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('对比损失')
        plt.title('训练损失曲线')
        plt.grid(True, alpha=0.3)
        
        # 相似度变化
        plt.subplot(2, 3, 2)
        if pretrainer.similarity_scores:
            sim_epochs = range(config['eval_every_n_epochs'], 
                             len(pretrainer.similarity_scores) * config['eval_every_n_epochs'] + 1, 
                             config['eval_every_n_epochs'])
            plt.plot(sim_epochs, pretrainer.similarity_scores, 'g-', linewidth=2, marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('同句EEG-文本相似度')
            plt.title('相似度变化')
            plt.grid(True, alpha=0.3)
        
        # 训练时间
        plt.subplot(2, 3, 3)
        plt.plot(epochs, pretrainer.epoch_times, 'r-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('时间 (秒)')
        plt.title('每轮训练时间')
        plt.grid(True, alpha=0.3)
        
        # 最终相似度分布
        plt.subplot(2, 3, 4)
        if 'overall_statistics' in similarity_analysis:
            stats = similarity_analysis['overall_statistics']
            metrics = ['平均相似度', '最高相似度', '最低相似度']
            values = [
                stats['avg_same_sentence_similarity'],
                stats['max_same_sentence_similarity'],
                stats['min_same_sentence_similarity']
            ]
            plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral'])
            plt.ylabel('相似度')
            plt.title('同句子EEG相似度分析')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        # 训练摘要
        plt.subplot(2, 3, 5)
        summary_text = [
            f"训练轮数: {len(pretrainer.training_losses)}",
            f"最佳损失: {best_loss:.4f}",
            f"数据集大小: {len(full_dataset)}",
            f"唯一句子: {len(unique_sentences)}",
            f"批次大小: {config['batch_size']}",
            f"学习率: {config['learning_rate']:.0e}"
        ]
        
        plt.text(0.1, 0.8, "🎯 预训练总结", fontsize=14, fontweight='bold')
        for i, text in enumerate(summary_text):
            plt.text(0.1, 0.7 - i*0.1, text, fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        # 损失改善趋势
        plt.subplot(2, 3, 6)
        if len(pretrainer.training_losses) > 10:
            # 计算滑动平均
            window_size = min(10, len(pretrainer.training_losses) // 4)
            smoothed_losses = []
            for i in range(window_size, len(pretrainer.training_losses)):
                smoothed_losses.append(
                    np.mean(pretrainer.training_losses[i-window_size:i])
                )
            
            smooth_epochs = range(window_size+1, len(pretrainer.training_losses) + 1)
            plt.plot(epochs, pretrainer.training_losses, 'b-', alpha=0.3, label='原始损失')
            plt.plot(smooth_epochs, smoothed_losses, 'r-', linewidth=2, label=f'滑动平均({window_size})')
            plt.xlabel('Epoch')
            plt.ylabel('损失')
            plt.title('损失平滑趋势')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = save_dir / 'pretraining_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"📊 训练历史图保存到: {plot_path}")
    
    # 保存完整结果
    results = {
        'best_loss': best_loss,
        'training_epochs': len(pretrainer.training_losses),
        'training_history': {
            'losses': pretrainer.training_losses,
            'similarities': pretrainer.similarity_scores,
            'epoch_times': pretrainer.epoch_times
        },
        'dataset_info': {
            'total_samples': len(full_dataset),
            'unique_sentences': len(unique_sentences),
            'avg_eegs_per_sentence': len(sentences) / len(unique_sentences)
        },
        'similarity_analysis': similarity_analysis,
        'config': config,
        'training_strategy': 'contrastive_pretraining_full_dataset',
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = save_dir / 'pretraining_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 完成信息
    logger.info(f"\n🎉 EEG编码器预训练完成!")
    logger.info(f"📊 最佳对比损失: {best_loss:.4f}")
    logger.info(f"📊 训练轮数: {len(pretrainer.training_losses)}")
    logger.info(f"📊 数据集大小: {len(full_dataset)}")
    
    if pretrainer.similarity_scores:
        logger.info(f"📊 最终同句相似度: {pretrainer.similarity_scores[-1]:.4f}")
    
    if 'overall_statistics' in similarity_analysis:
        stats = similarity_analysis['overall_statistics']
        logger.info(f"📊 同句EEG平均相似度: {stats['avg_same_sentence_similarity']:.4f}")
    
    logger.info(f"📂 结果保存在: {save_dir}")
    logger.info(f"💾 预训练模型: {model_path}")
    logger.info(f"📊 相似度分析: {save_dir / 'same_sentence_similarity_analysis.json'}")
    logger.info(f"📈 训练历史: {results_path}")

if __name__ == "__main__":
    main()

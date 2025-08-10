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

# ✅ 添加BLEU分数计算相关导入
try:
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    
    # 下载必要的NLTK数据
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    NLTK_AVAILABLE = True
except ImportError:
    print("⚠️ NLTK未安装，将使用简单的字符级BLEU计算")
    NLTK_AVAILABLE = False

from eeg_to_text_model_contrastive import ContrastiveEEGTextModel

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ BLEU分数计算器
class BLEUCalculator:
    """BLEU分数计算器"""
    
    def __init__(self, use_smoothing=True):
        self.use_smoothing = use_smoothing
        if NLTK_AVAILABLE and use_smoothing:
            self.smoothing_function = SmoothingFunction().method1
        else:
            self.smoothing_function = None
    
    def _tokenize_chinese(self, text):
        """中文分词 - 字符级别或词级别"""
        if NLTK_AVAILABLE:
            try:
                # 尝试使用NLTK分词
                tokens = word_tokenize(text)
                return tokens
            except:
                # 如果失败，使用字符级分词
                return list(text.replace(' ', ''))
        else:
            # 简单的字符级分词（适合中文）
            return list(text.replace(' ', ''))
    
    def calculate_sentence_bleu(self, reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)):
        """计算单句BLEU分数"""
        if not reference or not candidate:
            return 0.0
        
        # 分词
        ref_tokens = self._tokenize_chinese(reference)
        cand_tokens = self._tokenize_chinese(candidate)
        
        if len(ref_tokens) == 0 or len(cand_tokens) == 0:
            return 0.0
        
        if NLTK_AVAILABLE:
            try:
                # 使用NLTK计算BLEU
                bleu_score = sentence_bleu(
                    [ref_tokens], 
                    cand_tokens, 
                    weights=weights,
                    smoothing_function=self.smoothing_function
                )
                return bleu_score
            except:
                # 如果NLTK失败，使用简单方法
                return self._simple_bleu(ref_tokens, cand_tokens)
        else:
            return self._simple_bleu(ref_tokens, cand_tokens)
    
    def _simple_bleu(self, ref_tokens, cand_tokens):
        """简单的BLEU计算（当NLTK不可用时）"""
        # 计算1-gram精确度
        ref_set = set(ref_tokens)
        cand_set = set(cand_tokens)
        
        if len(cand_set) == 0:
            return 0.0
        
        precision_1 = len(ref_set.intersection(cand_set)) / len(cand_set)
        
        # 简化的长度惩罚
        bp = min(1.0, len(cand_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0.0
        
        return bp * precision_1
    
    def calculate_corpus_bleu(self, references, candidates, weights=(0.25, 0.25, 0.25, 0.25)):
        """计算语料库级BLEU分数"""
        if len(references) != len(candidates):
            raise ValueError("参考文本和候选文本数量不匹配")
        
        if not references or not candidates:
            return 0.0
        
        # 分词所有文本
        ref_tokens_list = []
        cand_tokens_list = []
        
        for ref, cand in zip(references, candidates):
            if ref and cand:  # 跳过空文本
                ref_tokens = self._tokenize_chinese(ref)
                cand_tokens = self._tokenize_chinese(cand)
                
                if len(ref_tokens) > 0 and len(cand_tokens) > 0:
                    ref_tokens_list.append([ref_tokens])  # NLTK需要嵌套列表
                    cand_tokens_list.append(cand_tokens)
        
        if not ref_tokens_list or not cand_tokens_list:
            return 0.0
        
        if NLTK_AVAILABLE:
            try:
                bleu_score = corpus_bleu(
                    ref_tokens_list, 
                    cand_tokens_list, 
                    weights=weights,
                    smoothing_function=self.smoothing_function
                )
                return bleu_score
            except:
                # 如果NLTK失败，计算平均句子BLEU
                scores = []
                for ref_tokens, cand_tokens in zip(ref_tokens_list, cand_tokens_list):
                    score = self._simple_bleu(ref_tokens[0], cand_tokens)
                    scores.append(score)
                return np.mean(scores) if scores else 0.0
        else:
            # 计算平均句子BLEU
            scores = []
            for ref_tokens, cand_tokens in zip(ref_tokens_list, cand_tokens_list):
                score = self._simple_bleu(ref_tokens[0], cand_tokens)
                scores.append(score)
            return np.mean(scores) if scores else 0.0

class EEGTextDataset(Dataset):
    """EEG和文本配对数据集"""
    
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
        
        logger.info(f"找到 {len(self.valid_data)} 个有效的EEG-文本配对")
        if len(not_found_ids) > 0:
            logger.warning(f"未找到EEG文件的ID数量: {len(not_found_ids)}")
    
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

def collate_fn(batch):
    """批处理函数"""
    eeg_list = [item['eeg'] for item in batch]
    text_list = [item['text'] for item in batch]
    ids = [item['id'] for item in batch]
    
    return {
        'eeg': eeg_list,
        'text': text_list,
        'ids': ids
    }

class IntegratedEEGToTextModel(nn.Module):
    """整合分类预训练的EEG-to-Text模型"""
    
    def __init__(self, pretrained_classification_model, bart_model_name="fnlp/bart-base-chinese"):
        super(IntegratedEEGToTextModel, self).__init__()
        
        # 使用预训练的分类模型作为EEG编码器
        self.eeg_encoder = pretrained_classification_model
        
        # BART模型用于文本生成
        from transformers import BartForConditionalGeneration
        self.bart = BartForConditionalGeneration.from_pretrained(bart_model_name)
        
        # ✅ 投影层：从patch_dim投影到BART隐藏维度
        eeg_patch_dim = self.eeg_encoder.patch_dim  # 256
        bart_hidden_dim = self.bart.config.d_model   # 768
        
        self.eeg_to_bart_projection = nn.Linear(eeg_patch_dim, bart_hidden_dim)

    def configure_for_stage2(self, freeze_eeg_encoder=True):
        """配置阶段2训练参数"""
        if freeze_eeg_encoder:
            print("🔒 冻结EEG编码器（使用预训练特征）")
            # ✅ 直接冻结整个EEG编码器
            for param in self.eeg_encoder.parameters():
                param.requires_grad = False
        else:
            print("🔓 解冻EEG编码器（联合微调）")
            for param in self.eeg_encoder.parameters():
                param.requires_grad = True
        
        # 解冻BART解码器
        for param in self.bart.model.decoder.parameters():
            param.requires_grad = True
        for param in self.bart.lm_head.parameters():
            param.requires_grad = True
        
        # 解冻投影层
        for param in self.eeg_to_bart_projection.parameters():
            param.requires_grad = True

    def forward(self, eeg_data, text_data=None, tokenizer=None):
        """前向传播"""
        # ✅ 直接使用预训练编码器的encode_eeg方法
        encoded_eeg, eeg_mask, patch_lengths = self.eeg_encoder.encode_eeg(eeg_data)
        
        # ✅ 投影到BART空间：(batch_size, seq_len, patch_dim) → (batch_size, seq_len, bart_hidden_dim)
        bart_sequence = self.eeg_to_bart_projection(encoded_eeg)
        
        eeg_mask = eeg_mask.to(encoded_eeg.device)
        
        # 转换mask格式：EEG mask (True=无效) → BART attention mask (True=有效)
        encoder_attention_mask = ~eeg_mask
        
        if text_data is not None and tokenizer is not None:
            # 训练模式：计算生成损失
            encodings = tokenizer(
                text_data,
                padding='max_length',
                max_length=32,
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encodings['input_ids'].to(bart_sequence.device)
            attention_mask = encodings['attention_mask'].to(bart_sequence.device)
            
            # 准备解码器输入
            decoder_input_ids = input_ids[:, :-1]
            labels = input_ids[:, 1:].clone()
            labels[labels == tokenizer.pad_token_id] = -100
            
            # BART前向传播
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(
                last_hidden_state=bart_sequence,
                hidden_states=None,
                attentions=None
            )
            
            outputs = self.bart(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,  # EEG的attention mask
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=attention_mask[:, :-1],  # 文本的attention mask
                labels=labels,
                return_dict=True
            )
            
            return outputs
        else:
            # 推理模式
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(
                last_hidden_state=bart_sequence,
                hidden_states=None,
                attentions=None
            )
            return encoder_outputs, encoder_attention_mask

    def generate_text(self, eeg_data, tokenizer, max_length=32, num_beams=4, **kwargs):
        """生成文本"""
        with torch.no_grad():
            # ✅ 获取EEG特征并投影
            encoded_eeg, eeg_mask, patch_lengths = self.eeg_encoder.encode_eeg(eeg_data)
            bart_sequence = self.eeg_to_bart_projection(encoded_eeg)
            
            # 获取attention mask
            eeg_mask = eeg_mask.to(encoded_eeg.device)
            encoder_attention_mask = ~eeg_mask
            
            # 创建encoder_outputs
            from transformers.modeling_outputs import BaseModelOutput
            encoder_outputs = BaseModelOutput(
                last_hidden_state=bart_sequence,
                hidden_states=None,
                attentions=None
            )
            
            generated_ids = self.bart.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=encoder_attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.sep_token_id,
                **kwargs
            )
            
            # 解码生成的文本
            generated_texts = []
            for ids in generated_ids:
                text = tokenizer.decode(ids, skip_special_tokens=True)
                generated_texts.append(text)
            
            return generated_texts

# ✅ 修改为训练集/测试集二分割
def create_train_test_split(dataset, test_size=0.2, random_state=42):
    """创建训练/测试集分割"""
    logger.info("🔄 创建二分割数据集 (训练/测试)")
    
    total_size = len(dataset)
    indices = list(range(total_size))
    
    # 分割为训练集和测试集
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    logger.info(f"📊 数据集分割结果:")
    logger.info(f"  🎓 训练集: {len(train_indices)} 样本 ({len(train_indices)/total_size*100:.1f}%)")
    logger.info(f"  🧪 测试集: {len(test_indices)} 样本 ({len(test_indices)/total_size*100:.1f}%)")
    
    return train_indices, test_indices

def save_data_splits(train_indices, test_indices, save_dir):
    """保存数据分割索引"""
    splits_dir = save_dir / 'data_splits'
    splits_dir.mkdir(exist_ok=True)
    
    np.save(splits_dir / 'train_indices.npy', train_indices)
    np.save(splits_dir / 'test_indices.npy', test_indices)
    
    # 保存分割信息
    split_info = {
        'train_size': len(train_indices),
        'test_size': len(test_indices),
        'total_size': len(train_indices) + len(test_indices),
        'split_ratios': {
            'train': len(train_indices) / (len(train_indices) + len(test_indices)),
            'test': len(test_indices) / (len(train_indices) + len(test_indices))
        },
        'creation_time': datetime.now().isoformat()
    }
    
    with open(splits_dir / 'split_info.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"📁 数据分割索引保存到: {splits_dir}")

class IntegratedTrainer:
    """整合的训练器"""
    
    def __init__(self, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = tokenizer
        self.device = device
        
        # 训练历史记录
        self.training_losses = []
        
        # ✅ 添加BLEU计算器
        self.bleu_calculator = BLEUCalculator(use_smoothing=True)
    
    def train_with_test_evaluation(self, train_loader, test_loader, pretrained_model, 
                                  epochs=60, learning_rate=2e-5, 
                                  save_path="final_model.pth", freeze_eeg=True):
        """训练模型并在测试集上评估"""
        print("\n" + "="*80)
        print("🔥 EEG-to-Text模型训练")
        print("🎯 目标: 使用预训练的EEG编码器训练文本生成")
        print(f"🔒 EEG编码器: {'冻结' if freeze_eeg else '微调'}")
        print("🔓 BART解码器: 微调")
        print("="*80)
        
        # 创建整合模型
        integrated_model = IntegratedEEGToTextModel(
            pretrained_classification_model=pretrained_model,
            bart_model_name="fnlp/bart-base-chinese"
        ).to(self.device)
        
        # 配置训练参数
        integrated_model.configure_for_stage2(freeze_eeg_encoder=freeze_eeg)
        
        # ✅ 分析可训练参数
        trainable_params = [p for p in integrated_model.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in integrated_model.parameters())
        
        print(f"📊 模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {total_trainable:,}")
        print(f"  冻结参数: {total_params - total_trainable:,}")
        print(f"  训练比例: {total_trainable/total_params*100:.1f}%")
        
        # 优化器和学习率调度器
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, min_lr=1e-7
        )
        
        # 训练循环
        best_train_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            
            # 训练
            train_loss = self._train_epoch(integrated_model, train_loader, optimizer, epoch+1)
            
            scheduler.step(train_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"训练损失: {train_loss:.4f}")
            print(f"当前学习率: {current_lr:.2e}")
            
            # 记录历史
            self.training_losses.append(train_loss)
            
            # 每5个epoch测试生成效果
            if (epoch + 1) % 5 == 0:
                print("\n📝 生成样本测试:")
                self._test_generation(integrated_model, test_loader, num_samples=3)
            
            # 早停和保存
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience_counter = 0
                
                torch.save({
                    'model_state_dict': integrated_model.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                }, save_path)
                print(f"💾 保存最佳模型: {save_path}")
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"⏹️ 早停触发")
                    break
        
        print(f"\n✅ 训练完成! 最佳训练损失: {best_train_loss:.4f}")
        return integrated_model, best_train_loss
    
    def evaluate_on_test_set(self, model, test_loader, save_dir):
        """在测试集上评估模型（包含BLEU分数）"""
        logger.info("\n🧪 测试集评估开始...")
        
        model.eval()
        test_results = {
            'total_loss': 0.0,
            'num_batches': 0,
            'generated_samples': [],
            'metrics': {}
        }
        
        all_generated_texts = []
        all_original_texts = []
        all_ids = []
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="测试集评估")
            
            for batch in progress_bar:
                try:
                    # 计算损失
                    outputs = model(
                        eeg_data=batch['eeg'],
                        text_data=batch['text'],
                        tokenizer=self.tokenizer
                    )
                    
                    loss = outputs.loss
                    test_results['total_loss'] += loss.item()
                    test_results['num_batches'] += 1
                    
                    # 生成文本
                    generated_texts = model.generate_text(
                        eeg_data=batch['eeg'],
                        tokenizer=self.tokenizer,
                        max_length=32,
                        num_beams=4
                    )
                    
                    # 收集结果
                    for gen_text, orig_text, sample_id in zip(generated_texts, batch['text'], batch['ids']):
                        all_generated_texts.append(gen_text)
                        all_original_texts.append(orig_text)
                        all_ids.append(sample_id)
                        
                        test_results['generated_samples'].append({
                            'id': sample_id,
                            'original': orig_text,
                            'generated': gen_text
                        })
                    
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                except Exception as e:
                    logger.error(f"测试批次处理失败: {e}")
                    continue
        
        # 计算平均损失
        avg_test_loss = test_results['total_loss'] / test_results['num_batches'] if test_results['num_batches'] > 0 else float('inf')
        test_results['avg_loss'] = avg_test_loss
        
        # ✅ 计算各种评估指标
        logger.info("📊 计算评估指标...")
        
        # 1. 文本相似度（原有）
        from difflib import SequenceMatcher
        similarities = []
        for gen, orig in zip(all_generated_texts, all_original_texts):
            sim = SequenceMatcher(None, gen, orig).ratio()
            similarities.append(sim)
        
        # 2. ✅ BLEU分数计算
        logger.info("🔢 计算BLEU分数...")
        bleu_scores = {
            'bleu_1': [],
            'bleu_2': [],
            'bleu_3': [],
            'bleu_4': [],
            'corpus_bleu_1': 0.0,
            'corpus_bleu_2': 0.0,
            'corpus_bleu_3': 0.0,
            'corpus_bleu_4': 0.0
        }
        
        # 计算每个句子的BLEU分数
        for gen_text, orig_text in zip(all_generated_texts, all_original_texts):
            # BLEU-1到BLEU-4
            bleu_1 = self.bleu_calculator.calculate_sentence_bleu(orig_text, gen_text, weights=(1.0, 0, 0, 0))
            bleu_2 = self.bleu_calculator.calculate_sentence_bleu(orig_text, gen_text, weights=(0.5, 0.5, 0, 0))
            bleu_3 = self.bleu_calculator.calculate_sentence_bleu(orig_text, gen_text, weights=(0.33, 0.33, 0.33, 0))
            bleu_4 = self.bleu_calculator.calculate_sentence_bleu(orig_text, gen_text, weights=(0.25, 0.25, 0.25, 0.25))
            
            bleu_scores['bleu_1'].append(bleu_1)
            bleu_scores['bleu_2'].append(bleu_2)
            bleu_scores['bleu_3'].append(bleu_3)
            bleu_scores['bleu_4'].append(bleu_4)
        
        # 计算语料库级BLEU分数
        if all_generated_texts and all_original_texts:
            bleu_scores['corpus_bleu_1'] = self.bleu_calculator.calculate_corpus_bleu(
                all_original_texts, all_generated_texts, weights=(1.0, 0, 0, 0)
            )
            bleu_scores['corpus_bleu_2'] = self.bleu_calculator.calculate_corpus_bleu(
                all_original_texts, all_generated_texts, weights=(0.5, 0.5, 0, 0)
            )
            bleu_scores['corpus_bleu_3'] = self.bleu_calculator.calculate_corpus_bleu(
                all_original_texts, all_generated_texts, weights=(0.33, 0.33, 0.33, 0)
            )
            bleu_scores['corpus_bleu_4'] = self.bleu_calculator.calculate_corpus_bleu(
                all_original_texts, all_generated_texts, weights=(0.25, 0.25, 0.25, 0.25)
            )
        
        # 3. ✅ ROUGE-like分数
        def calculate_rouge_like_score(reference, candidate):
            """简单的ROUGE-L类似分数"""
            ref_tokens = set(self.bleu_calculator._tokenize_chinese(reference))
            cand_tokens = set(self.bleu_calculator._tokenize_chinese(candidate))
            
            if not ref_tokens or not cand_tokens:
                return 0.0
            
            intersection = ref_tokens.intersection(cand_tokens)
            if not intersection:
                return 0.0
            
            precision = len(intersection) / len(cand_tokens)
            recall = len(intersection) / len(ref_tokens)
            
            if precision + recall == 0:
                return 0.0
            
            f1 = 2 * precision * recall / (precision + recall)
            return f1
        
        rouge_like_scores = []
        for gen, orig in zip(all_generated_texts, all_original_texts):
            score = calculate_rouge_like_score(orig, gen)
            rouge_like_scores.append(score)
        
        # 整合所有指标
        if similarities:
            test_results['metrics'] = {
                # 基本文本相似度
                'avg_similarity': np.mean(similarities),
                'median_similarity': np.median(similarities),
                'std_similarity': np.std(similarities),
                'max_similarity': np.max(similarities),
                'min_similarity': np.min(similarities),
                
                # ✅ BLEU分数
                'bleu_scores': {
                    'avg_bleu_1': np.mean(bleu_scores['bleu_1']) if bleu_scores['bleu_1'] else 0.0,
                    'avg_bleu_2': np.mean(bleu_scores['bleu_2']) if bleu_scores['bleu_2'] else 0.0,
                    'avg_bleu_3': np.mean(bleu_scores['bleu_3']) if bleu_scores['bleu_3'] else 0.0,
                    'avg_bleu_4': np.mean(bleu_scores['bleu_4']) if bleu_scores['bleu_4'] else 0.0,
                    'corpus_bleu_1': bleu_scores['corpus_bleu_1'],
                    'corpus_bleu_2': bleu_scores['corpus_bleu_2'],
                    'corpus_bleu_3': bleu_scores['corpus_bleu_3'],
                    'corpus_bleu_4': bleu_scores['corpus_bleu_4'],
                    'std_bleu_1': np.std(bleu_scores['bleu_1']) if bleu_scores['bleu_1'] else 0.0,
                    'std_bleu_2': np.std(bleu_scores['bleu_2']) if bleu_scores['bleu_2'] else 0.0,
                    'std_bleu_3': np.std(bleu_scores['bleu_3']) if bleu_scores['bleu_3'] else 0.0,
                    'std_bleu_4': np.std(bleu_scores['bleu_4']) if bleu_scores['bleu_4'] else 0.0,
                },
                
                # ✅ ROUGE-like分数
                'rouge_like': {
                    'avg_rouge_like': np.mean(rouge_like_scores) if rouge_like_scores else 0.0,
                    'std_rouge_like': np.std(rouge_like_scores) if rouge_like_scores else 0.0,
                    'max_rouge_like': np.max(rouge_like_scores) if rouge_like_scores else 0.0,
                    'min_rouge_like': np.min(rouge_like_scores) if rouge_like_scores else 0.0,
                }
            }
        
        # 保存测试结果
        test_results_path = save_dir / 'test_results_with_bleu.json'
        with open(test_results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, ensure_ascii=False, indent=2)
        
        # ✅ 保存详细的评估结果（包含BLEU分数）
        detailed_results = []
        for i, (gen_text, orig_text, sample_id) in enumerate(zip(all_generated_texts, all_original_texts, all_ids)):
            detailed_results.append({
                'id': sample_id,
                'original_text': orig_text,
                'generated_text': gen_text,
                'similarity': similarities[i] if similarities else 0,
                'bleu_1': bleu_scores['bleu_1'][i] if i < len(bleu_scores['bleu_1']) else 0,
                'bleu_2': bleu_scores['bleu_2'][i] if i < len(bleu_scores['bleu_2']) else 0,
                'bleu_3': bleu_scores['bleu_3'][i] if i < len(bleu_scores['bleu_3']) else 0,
                'bleu_4': bleu_scores['bleu_4'][i] if i < len(bleu_scores['bleu_4']) else 0,
                'rouge_like': rouge_like_scores[i] if i < len(rouge_like_scores) else 0,
            })
        
        samples_df = pd.DataFrame(detailed_results)
        samples_df.to_csv(save_dir / 'test_samples_with_bleu.csv', index=False, encoding='utf-8')
        
        # ✅ 打印详细的测试结果
        logger.info(f"\n📊 测试集评估结果:")
        logger.info(f"  📝 平均生成损失: {avg_test_loss:.4f}")
        logger.info(f"  📊 测试样本数量: {len(all_generated_texts)}")
        
        if similarities:
            logger.info(f"\n  📈 文本相似度指标:")
            logger.info(f"    平均相似度: {test_results['metrics']['avg_similarity']:.4f}")
            logger.info(f"    中位数相似度: {test_results['metrics']['median_similarity']:.4f}")
            logger.info(f"    最高相似度: {test_results['metrics']['max_similarity']:.4f}")
            logger.info(f"    最低相似度: {test_results['metrics']['min_similarity']:.4f}")
            
            # ✅ BLEU分数
            logger.info(f"\n  🔢 BLEU分数:")
            bleu_metrics = test_results['metrics']['bleu_scores']
            logger.info(f"    平均 BLEU-1: {bleu_metrics['avg_bleu_1']:.4f} ± {bleu_metrics['std_bleu_1']:.4f}")
            logger.info(f"    平均 BLEU-2: {bleu_metrics['avg_bleu_2']:.4f} ± {bleu_metrics['std_bleu_2']:.4f}")
            logger.info(f"    平均 BLEU-3: {bleu_metrics['avg_bleu_3']:.4f} ± {bleu_metrics['std_bleu_3']:.4f}")
            logger.info(f"    平均 BLEU-4: {bleu_metrics['avg_bleu_4']:.4f} ± {bleu_metrics['std_bleu_4']:.4f}")
            
            logger.info(f"\n  📚 语料库级 BLEU分数:")
            logger.info(f"    语料库 BLEU-1: {bleu_metrics['corpus_bleu_1']:.4f}")
            logger.info(f"    语料库 BLEU-2: {bleu_metrics['corpus_bleu_2']:.4f}")
            logger.info(f"    语料库 BLEU-3: {bleu_metrics['corpus_bleu_3']:.4f}")
            logger.info(f"    语料库 BLEU-4: {bleu_metrics['corpus_bleu_4']:.4f}")
            
            # ✅ ROUGE-like分数
            rouge_metrics = test_results['metrics']['rouge_like']
            logger.info(f"\n  📋 ROUGE-like分数:")
            logger.info(f"    平均 ROUGE-like: {rouge_metrics['avg_rouge_like']:.4f} ± {rouge_metrics['std_rouge_like']:.4f}")
            logger.info(f"    最高 ROUGE-like: {rouge_metrics['max_rouge_like']:.4f}")
            logger.info(f"    最低 ROUGE-like: {rouge_metrics['min_rouge_like']:.4f}")
        
        # 显示一些生成样本
        logger.info(f"\n📝 测试集生成样本示例 (带BLEU分数):")
        for i, sample in enumerate(test_results['generated_samples'][:5]):
            detail = detailed_results[i]
            logger.info(f"  样本 {i+1} (ID: {sample['id']}):")
            logger.info(f"    原文: {sample['original']}")
            logger.info(f"    生成: {sample['generated']}")
            logger.info(f"    相似度: {detail['similarity']:.3f}")
            logger.info(f"    BLEU-1: {detail['bleu_1']:.3f}, BLEU-2: {detail['bleu_2']:.3f}, BLEU-3: {detail['bleu_3']:.3f}, BLEU-4: {detail['bleu_4']:.3f}")
            logger.info(f"    ROUGE-like: {detail['rouge_like']:.3f}")
        
        logger.info(f"\n📁 测试结果保存到: {test_results_path}")
        logger.info(f"📁 详细样本（含BLEU）保存到: {save_dir / 'test_samples_with_bleu.csv'}")
        
        return test_results
    
    def _train_epoch(self, model, train_loader, optimizer, epoch):
        """训练epoch"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            try:
                outputs = model(
                    eeg_data=batch['eeg'],
                    text_data=batch['text'],
                    tokenizer=self.tokenizer
                )
                
                loss = outputs.loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"训练批次出错: {e}")
                continue
        
        return total_loss / num_batches if num_batches > 0 else float('inf')
    
    def _test_generation(self, model, test_loader, num_samples=3):
        """测试生成效果"""
        model.eval()
        samples_generated = 0
        
        with torch.no_grad():
            for batch in test_loader:
                if samples_generated >= num_samples:
                    break
                
                eeg_data = batch['eeg'][:min(num_samples - samples_generated, len(batch['eeg']))]
                original_texts = batch['text'][:len(eeg_data)]
                ids = batch['ids'][:len(eeg_data)]
                
                try:
                    generated_texts = model.generate_text(
                        eeg_data=eeg_data,
                        tokenizer=self.tokenizer,
                        max_length=32,
                        num_beams=4
                    )
                    
                    for i, (gen_text, orig_text, sample_id) in enumerate(zip(generated_texts, original_texts, ids)):
                        print(f"  样本 {samples_generated + i + 1} (ID: {sample_id}):")
                        print(f"    原文: {orig_text}")
                        print(f"    生成: {gen_text}")
                        
                    samples_generated += len(generated_texts)
                    
                except Exception as e:
                    print(f"  生成失败: {e}")
                    break

def main():
    """主训练函数"""
    config = {
        'csv_file': 'additional_data/subject1_gxy_0328/gxy_0328.csv',
        'eeg_dir': 'additional_data/subject1_gxy_0328/',
        'batch_size': 16,
        'test_size': 0.2,      # ✅ 测试集比例
        'random_state': 42,
        
        # 训练参数
        'epochs': 60,
        'learning_rate': 2e-5,
        'freeze_eeg': True,  # 冻结EEG编码器
        
        # 预训练模型路径
        'pretrained_model_path': 'stage1_only_900samples_20250711_164810/stage1_contrastive_pretrained.pth',
        
        'save_dir': f'train_test_bleu_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    logger.info("=== EEG-to-Text训练（含BLEU评估） ===")
    logger.info("🎯 训练策略:")
    logger.info("  📚 使用300分类预训练的EEG编码器")
    logger.info("  🔒 冻结EEG编码器，微调BART解码器")
    logger.info("  📊 二分割: 训练集/测试集")
    logger.info("  🔢 包含BLEU分数评估")
    
    # 检查数据文件
    if not os.path.exists(config['csv_file']):
        logger.error(f"CSV文件不存在: {config['csv_file']}")
        return
    
    if not os.path.exists(config['eeg_dir']):
        logger.error(f"EEG目录不存在: {config['eeg_dir']}")
        return
    
    if not os.path.exists(config['pretrained_model_path']):
        logger.error(f"预训练模型不存在: {config['pretrained_model_path']}")
        return
    
    # 创建保存目录
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化tokenizer
    logger.info("初始化tokenizer...")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('fnlp/bart-base-chinese')
        logger.info(f"✅ 成功加载tokenizer: {type(tokenizer).__name__}")
    except:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        logger.info("✅ 使用BertTokenizer")
    
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
    
    # ✅ 创建训练/测试集分割
    train_indices, test_indices = create_train_test_split(
        full_dataset,
        test_size=config['test_size'],
        random_state=config['random_state']
    )
    
    # 保存数据分割索引
    save_data_splits(train_indices, test_indices, save_dir)
    
    # 创建数据子集
    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], 
        shuffle=False, collate_fn=collate_fn, num_workers=0
    )
    
    # 加载预训练模型
    logger.info(f"加载预训练的300分类模型: {config['pretrained_model_path']}")
    pretrained_model = ContrastiveEEGTextModel(
        patch_dim=256,
        feature_dim=768,
        target_channels=66,
        target_length=4000
    )
    
    checkpoint = torch.load(config['pretrained_model_path'], map_location='cpu', weights_only=False)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"✅ 成功加载预训练模型 (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # 创建训练器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    pretrained_model = pretrained_model.to(device)
    trainer = IntegratedTrainer(tokenizer, device)
    
    # 训练模型
    model_path = save_dir / 'final_model.pth'
    final_model, best_loss = trainer.train_with_test_evaluation(
        train_loader=train_loader,
        test_loader=test_loader,
        pretrained_model=pretrained_model,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        save_path=model_path,
        freeze_eeg=config['freeze_eeg']
    )
    
    # ✅ 在测试集上评估最终模型
    logger.info("\n" + "="*80)
    logger.info("🧪 测试集评估（包含BLEU分数）")
    logger.info("="*80)
    
    # 加载最佳模型
    best_checkpoint = torch.load(model_path, map_location=device)
    final_model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # 测试集评估
    test_results = trainer.evaluate_on_test_set(final_model, test_loader, save_dir)
    
    # ✅ 绘制包含BLEU分数的训练历史和结果
    if trainer.training_losses:
        plt.figure(figsize=(15, 10))
        
        # 训练损失曲线
        plt.subplot(2, 3, 1)
        epochs = range(1, len(trainer.training_losses) + 1)
        plt.plot(epochs, trainer.training_losses, 'b-', label='训练损失', linewidth=2)
        plt.axhline(y=test_results['avg_loss'], color='r', linestyle='--', label=f'测试损失 ({test_results["avg_loss"]:.4f})')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.title('训练损失 vs 测试损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # ✅ BLEU分数条形图
        plt.subplot(2, 3, 2)
        if 'bleu_scores' in test_results['metrics']:
            bleu_metrics = test_results['metrics']['bleu_scores']
            bleu_names = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
            bleu_values = [
                bleu_metrics['avg_bleu_1'],
                bleu_metrics['avg_bleu_2'],
                bleu_metrics['avg_bleu_3'],
                bleu_metrics['avg_bleu_4']
            ]
            bleu_stds = [
                bleu_metrics['std_bleu_1'],
                bleu_metrics['std_bleu_2'],
                bleu_metrics['std_bleu_3'],
                bleu_metrics['std_bleu_4']
            ]
            
            plt.bar(bleu_names, bleu_values, yerr=bleu_stds, capsize=5, 
                   color=['skyblue', 'lightgreen', 'lightcoral', 'gold'], alpha=0.8)
            plt.ylabel('BLEU分数')
            plt.title('测试集 BLEU分数')
            plt.grid(True, alpha=0.3)
        
        # ✅ 语料库级BLEU分数
        plt.subplot(2, 3, 3)
        if 'bleu_scores' in test_results['metrics']:
            corpus_bleu_names = ['Corpus\nBLEU-1', 'Corpus\nBLEU-2', 'Corpus\nBLEU-3', 'Corpus\nBLEU-4']
            corpus_bleu_values = [
                bleu_metrics['corpus_bleu_1'],
                bleu_metrics['corpus_bleu_2'],
                bleu_metrics['corpus_bleu_3'],
                bleu_metrics['corpus_bleu_4']
            ]
            
            plt.bar(corpus_bleu_names, corpus_bleu_values, 
                   color=['navy', 'darkgreen', 'darkred', 'orange'], alpha=0.8)
            plt.ylabel('语料库BLEU分数')
            plt.title('语料库级 BLEU分数')
            plt.grid(True, alpha=0.3)
        
        # ✅ 各种指标对比
        plt.subplot(2, 3, 4)
        if 'avg_similarity' in test_results['metrics'] and 'bleu_scores' in test_results['metrics']:
            metrics_names = ['文本\n相似度', 'BLEU-1', 'BLEU-2', 'BLEU-4', 'ROUGE\n-like']
            metrics_values = [
                test_results['metrics']['avg_similarity'],
                test_results['metrics']['bleu_scores']['avg_bleu_1'],
                test_results['metrics']['bleu_scores']['avg_bleu_2'],
                test_results['metrics']['bleu_scores']['avg_bleu_4'],
                test_results['metrics']['rouge_like']['avg_rouge_like']
            ]
            
            plt.bar(metrics_names, metrics_values, 
                   color=['purple', 'skyblue', 'lightgreen', 'gold', 'pink'], alpha=0.8)
            plt.ylabel('分数')
            plt.title('各评估指标对比')
            plt.grid(True, alpha=0.3)
        
        # ✅ 性能总结雷达图（简化版）
        plt.subplot(2, 3, 5)
        if 'bleu_scores' in test_results['metrics']:
            performance_summary = {
                '损失': 1 - min(1.0, test_results['avg_loss'] / 10),  # 归一化损失
                'BLEU-1': test_results['metrics']['bleu_scores']['avg_bleu_1'],
                'BLEU-4': test_results['metrics']['bleu_scores']['avg_bleu_4'],
                '相似度': test_results['metrics']['avg_similarity'],
                'ROUGE': test_results['metrics']['rouge_like']['avg_rouge_like']
            }
            
            metrics_names = list(performance_summary.keys())
            metrics_values = list(performance_summary.values())
            
            plt.bar(metrics_names, metrics_values, 
                   color='lightblue', alpha=0.8, edgecolor='navy', linewidth=1)
            plt.ylabel('性能分数')
            plt.title('综合性能总结')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            # 添加数值标签
            for i, v in enumerate(metrics_values):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # ✅ 训练进度和最终结果
        plt.subplot(2, 3, 6)
        final_metrics = [
            f"训练轮数: {len(trainer.training_losses)}",
            f"最佳损失: {best_loss:.4f}",
            f"测试损失: {test_results['avg_loss']:.4f}",
            f"BLEU-4: {test_results['metrics'].get('bleu_scores', {}).get('avg_bleu_4', 0):.3f}",
            f"相似度: {test_results['metrics'].get('avg_similarity', 0):.3f}"
        ]
        
        plt.text(0.1, 0.8, "🎯 训练结果总结", fontsize=14, fontweight='bold')
        for i, metric in enumerate(final_metrics):
            plt.text(0.1, 0.7 - i*0.1, metric, fontsize=10)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        plt.tight_layout()
        plot_path = save_dir / 'training_results_with_bleu.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"训练历史图（含BLEU）保存到: {plot_path}")
    
    # ✅ 保存包含BLEU分数的完整结果
    results = {
        'best_train_loss': best_loss,
        'test_results': test_results,
        'training_history': {
            'losses': trainer.training_losses
        },
        'data_splits': {
            'train_size': len(train_indices),
            'test_size': len(test_indices),
            'train_ratio': len(train_indices) / len(full_dataset),
            'test_ratio': len(test_indices) / len(full_dataset)
        },
        'config': config,
        'training_strategy': 'pretrained_eeg_encoder_with_bart_decoder',
        'pretrained_model_info': {
            'path': config['pretrained_model_path'],
            'epoch': checkpoint.get('epoch', 'unknown'),
            'val_accuracy': checkpoint.get('val_accuracy', 'unknown')
        },
        'timestamp': datetime.now().isoformat(),
        # ✅ 评估摘要
        'evaluation_summary': {
            'avg_loss': test_results['avg_loss'],
            'avg_similarity': test_results['metrics'].get('avg_similarity', 0),
            'avg_bleu_1': test_results['metrics'].get('bleu_scores', {}).get('avg_bleu_1', 0),
            'avg_bleu_2': test_results['metrics'].get('bleu_scores', {}).get('avg_bleu_2', 0),
            'avg_bleu_3': test_results['metrics'].get('bleu_scores', {}).get('avg_bleu_3', 0),
            'avg_bleu_4': test_results['metrics'].get('bleu_scores', {}).get('avg_bleu_4', 0),
            'corpus_bleu_4': test_results['metrics'].get('bleu_scores', {}).get('corpus_bleu_4', 0),
            'avg_rouge_like': test_results['metrics'].get('rouge_like', {}).get('avg_rouge_like', 0)
        }
    }
    
    results_path = save_dir / 'final_results_with_bleu.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # ✅ 完成信息（包含BLEU分数）
    logger.info(f"\n🎉 EEG-to-Text训练完成!")
    logger.info(f"📊 最佳训练损失: {best_loss:.4f}")
    logger.info(f"📊 测试集平均损失: {test_results['avg_loss']:.4f}")
    
    if 'bleu_scores' in test_results['metrics']:
        bleu_scores = test_results['metrics']['bleu_scores']
        logger.info(f"📊 测试集BLEU分数:")
        logger.info(f"   BLEU-1: {bleu_scores['avg_bleu_1']:.4f}")
        logger.info(f"   BLEU-2: {bleu_scores['avg_bleu_2']:.4f}")
        logger.info(f"   BLEU-3: {bleu_scores['avg_bleu_3']:.4f}")
        logger.info(f"   BLEU-4: {bleu_scores['avg_bleu_4']:.4f}")
        logger.info(f"   语料库BLEU-4: {bleu_scores['corpus_bleu_4']:.4f}")
    
    if 'avg_similarity' in test_results['metrics']:
        logger.info(f"📊 测试集平均相似度: {test_results['metrics']['avg_similarity']:.4f}")
    
    if 'rouge_like' in test_results['metrics']:
        logger.info(f"📊 测试集ROUGE-like: {test_results['metrics']['rouge_like']['avg_rouge_like']:.4f}")
    
    logger.info(f"📂 结果保存在: {save_dir}")
    logger.info(f"💾 最终模型: {model_path}")
    logger.info(f"📊 测试结果: {save_dir / 'test_results_with_bleu.json'}")
    logger.info(f"📝 生成样本: {save_dir / 'test_samples_with_bleu.csv'}")

if __name__ == "__main__":
    main()

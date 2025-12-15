import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class MovieGenreDataset(Dataset):
    """영화 줄거리 데이터셋 클래스"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 텍스트 토큰화
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertGenreClassifier(nn.Module):
    """BERT 기반 장르 분류 모델"""
    
    def __init__(self, n_classes, model_name='bert-base-multilingual-cased', dropout_rate=0.3):
        super(BertGenreClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 추가 레이어로 성능 향상
        self.pre_classifier = nn.Linear(self.bert.config.hidden_size, 512)
        self.classifier = nn.Linear(512, n_classes)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # 추가 레이어 통과
        pre_logits = self.pre_classifier(pooled_output)
        pre_logits = self.relu(pre_logits)
        pre_logits = self.dropout(pre_logits)
        
        return self.classifier(pre_logits)

class ImprovedMovieGenreClassifierTrainer:
    """개선된 영화 장르 분류 모델 훈련 클래스"""
    
    def __init__(self, model_name='bert-base-multilingual-cased', max_length=512):
        print("🚀 시스템 초기화 중...")
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.label_encoder = LabelEncoder()
        
        print(f"🚀 사용 디바이스: {self.device}")
        print(f"📱 모델: {model_name}")

    def load_train_data(self, train_file='improved_train_data.csv'):
        """훈련 데이터 로드"""
        print(f"📊 훈련 데이터 로드 중... ({train_file})")
        
        try:
            df = pd.read_csv(train_file, encoding='utf-8-sig')
            
            print(f"✅ 훈련 데이터 로드 완료: {len(df)}개 샘플")
            print(f"📋 컬럼: {list(df.columns)}")
            
            # 결측값 확인 및 처리
            print(f"\n🔍 결측값 확인:")
            missing_values = df[['줄거리', '장르']].isnull().sum()
            print(missing_values)
            
            # 결측값이 있는 행 제거
            df = df.dropna(subset=['줄거리', '장르'])
            print(f"📊 결측값 제거 후: {len(df)}개 샘플")
            
            # 다중 장르 처리 (첫 번째 장르만 사용)
            df['장르_단일'] = df['장르'].apply(lambda x: x.split(',')[0].strip())
            
            # 장르 분포 확인
            print(f"\n🎭 훈련 데이터 장르 분포:")
            genre_counts = df['장르_단일'].value_counts()
            print(genre_counts)
            
            return df
            
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {train_file}")
            print("💡 먼저 데이터 전처리를 실행하세요:")
            print("   python improved_processed_data.py")
            return None
    
    def load_evaluation_data(self, eval_file='improved_evaluation_data.csv'):
        """평가 데이터 로드"""
        print(f"\n📊 평가 데이터 로드 중... ({eval_file})")
        
        try:
            df = pd.read_csv(eval_file, encoding='utf-8-sig')
            
            print(f"✅ 평가 데이터 로드 완료: {len(df)}개 샘플")
            
            # 결측값 확인 및 처리
            df = df.dropna(subset=['줄거리', '장르'])
            
            # 다중 장르 처리 (첫 번째 장르만 사용)
            df['장르_단일'] = df['장르'].apply(lambda x: x.split(',')[0].strip())
            
            # 장르 분포 확인
            print(f"🎭 평가 데이터 장르 분포:")
            genre_counts = df['장르_단일'].value_counts()
            print(genre_counts)
            
            return df
            
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {eval_file}")
            return None
    
    def prepare_train_data(self, train_df):
        """훈련 데이터 준비"""
        print(f"\n📂 훈련 데이터 준비 중...")
        
        # 텍스트와 라벨 추출
        texts = train_df['줄거리'].tolist()
        labels = train_df['장르_단일'].tolist()
        
        # 라벨 인코딩
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        print(f"✅ 훈련 데이터: {len(texts)}개")
        print(f"🎭 총 장르 수: {len(self.label_encoder.classes_)}")
        
        # 장르 매핑 정보 출력
        print(f"\n🔢 장르 라벨 매핑:")
        for i, genre in enumerate(self.label_encoder.classes_):
            count = np.sum(np.array(encoded_labels) == i)
            print(f"  {i}: {genre} ({count}개)")
        
        return texts, encoded_labels
    
    def prepare_evaluation_data(self, eval_df):
        """평가 데이터 준비"""
        print(f"\n📂 평가 데이터 준비 중...")
        
        # 텍스트와 라벨 추출
        texts = eval_df['줄거리'].tolist()
        labels = eval_df['장르_단일'].tolist()
        
        # 훈련 데이터에서 학습한 라벨 인코더 사용
        try:
            encoded_labels = self.label_encoder.transform(labels)
        except ValueError as e:
            print(f"⚠️ 평가 데이터에 훈련 중 보지 못한 장르가 있습니다: {e}")
            # 알려진 장르만 필터링
            valid_indices = []
            valid_labels = []
            valid_texts = []
            
            for i, label in enumerate(labels):
                if label in self.label_encoder.classes_:
                    valid_indices.append(i)
                    valid_labels.append(label)
                    valid_texts.append(texts[i])
            
            encoded_labels = self.label_encoder.transform(valid_labels)
            texts = valid_texts
            print(f"🔄 알려진 장르만 사용: {len(texts)}개")
        
        print(f"✅ 평가 데이터: {len(texts)}개")
        
        return texts, encoded_labels
    
    def create_data_loaders(self, train_texts, train_labels, eval_texts, eval_labels, batch_size=16):
        """데이터 로더 생성"""
        print(f"\n🔄 데이터 로더 생성 중... (배치 크기: {batch_size})")
        
        # 데이터셋 생성
        train_dataset = MovieGenreDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        eval_dataset = MovieGenreDataset(
            eval_texts, eval_labels, self.tokenizer, self.max_length
        )
        
        # 데이터 로더 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✅ 훈련 배치 수: {len(train_loader)}")
        print(f"✅ 평가 배치 수: {len(eval_loader)}")
        
        return train_loader, eval_loader
    
    def initialize_model(self):
        """모델 초기화"""
        print(f"\n🤖 모델 초기화 중...")
        
        n_classes = len(self.label_encoder.classes_)
        self.model = BertGenreClassifier(n_classes, self.model_name)
        self.model.to(self.device)
        
        print(f"✅ 모델 초기화 완료 (클래스 수: {n_classes})")
        
        return self.model
    
    def compute_class_weights(self, train_labels):
        """클래스 가중치 계산"""
        classes = np.unique(train_labels)
        class_weights = compute_class_weight(
            'balanced',
            classes=classes,
            y=train_labels
        )
        
        # PyTorch tensor로 변환
        class_weights_tensor = torch.FloatTensor(class_weights).to(self.device)
        
        print(f"\n📊 클래스 가중치:")
        for i, weight in enumerate(class_weights):
            genre = self.label_encoder.classes_[i]
            print(f"  {genre}: {weight:.3f}")
        
        return class_weights_tensor
    
    def train_model(self, train_loader, eval_loader, train_labels, epochs=5, learning_rate=1e-5, weight_decay=0.01, warmup_ratio=0.1):
        """모델 훈련"""
        print(f"\n🏋️ 모델 훈련 시작...")
        print(f"📊 에포크: {epochs}, 학습률: {learning_rate}, 가중치 감소: {weight_decay}")
        
        # 클래스 가중치 계산
        class_weights = self.compute_class_weights(train_labels)
        
        # 옵티마이저와 스케줄러 설정
        optimizer = AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 가중치가 적용된 손실 함수
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 훈련 기록
        train_losses = []
        eval_accuracies = []
        
        for epoch in range(epochs):
            print(f"\n📚 에포크 {epoch + 1}/{epochs}")
            
            # 훈련 모드
            self.model.train()
            total_train_loss = 0
            
            train_pbar = tqdm(train_loader, desc=f"훈련 중")
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

                # ...existing code... (현재 파일 내용 그대로 유지하고 아래 내용을 추가)
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 평가
            eval_accuracy = self.evaluate_model(eval_loader)
            eval_accuracies.append(eval_accuracy)
            
            print(f"📊 에포크 {epoch + 1} 결과:")
            print(f"  - 평균 훈련 손실: {avg_train_loss:.4f}")
            print(f"  - 평가 정확도: {eval_accuracy:.4f}")
        
        # 훈련 결과 시각화
        self.plot_training_history(train_losses, eval_accuracies)
        
        return train_losses, eval_accuracies
    
    def evaluate_model(self, eval_loader):
        """모델 평가"""
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        accuracy = accuracy_score(actual_labels, predictions)
        return accuracy
    
    def detailed_evaluation(self, eval_loader):
        """상세 평가 및 분석"""
        print(f"\n📊 상세 평가 실행 중...")
        
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="평가 중"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        # 정확도 계산
        accuracy = accuracy_score(actual_labels, predictions)
        print(f"🎯 전체 정확도: {accuracy:.4f}")
        
        # 분류 리포트
        target_names = self.label_encoder.classes_
        report = classification_report(
            actual_labels, predictions, 
            target_names=target_names, 
            output_dict=True
        )
        
        print(f"\n📋 분류 리포트:")
        print(classification_report(actual_labels, predictions, target_names=target_names))
        
        # 혼동 행렬 시각화
        self.plot_confusion_matrix(actual_labels, predictions, target_names)
        
        return accuracy, report
    
    def plot_training_history(self, train_losses, eval_accuracies):
        """훈련 과정 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 훈련 손실
        ax1.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
        ax1.set_title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 평가 정확도
        ax2.plot(eval_accuracies, 'r-', linewidth=2, label='Evaluation Accuracy')
        ax2.set_title('Evaluation Accuracy Over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📈 훈련 과정 그래프가 'training_history.png'로 저장되었습니다.")
    
    def plot_confusion_matrix(self, actual_labels, predictions, target_names):
        """혼동 행렬 시각화"""
        cm = confusion_matrix(actual_labels, predictions)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names
        )
        plt.title('Movie Genre Classification Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 혼동 행렬이 'confusion_matrix.png'로 저장되었습니다.")
    
    def save_model(self, model_path='movie_genre_bert_model.pth', 
                   tokenizer_path='movie_genre_tokenizer', 
                   label_encoder_path='label_encoder.pkl'):
        """모델 저장"""
        print(f"\n💾 모델 저장 중...")
        
        # 모델 저장
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'n_classes': len(self.label_encoder.classes_),
                'model_name': self.model_name,
                'max_length': self.max_length
            }
        }, model_path)
        
        # 토크나이저 저장
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # 라벨 인코더 저장
        import pickle
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"✅ 모델 저장 완료:")
        print(f"  - 모델: {model_path}")
        print(f"  - 토크나이저: {tokenizer_path}")
        print(f"  - 라벨 인코더: {label_encoder_path}")
    
    def predict_text(self, text):
        """단일 텍스트 예측"""
        self.model.eval()
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,  # 문장 구분 ID는 사용하지 않음
            padding='max_length',
            truncation=True,
            return_attention_mask=True,  # 어텐션 마스크 포함
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)
        
        predicted_genre = self.label_encoder.inverse_transform([predicted.cpu().numpy()[0]])[0]
        confidence = probabilities[0][predicted].item()
        
        # 상위 3개 예측 결과 반환
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        top3_genres = self.label_encoder.inverse_transform(top3_indices.cpu().numpy())
        
        return predicted_genre, confidence, list(zip(top3_genres, top3_probs.cpu().numpy()))

def main():
    """메인 실행 함수"""
    print("🎬 영화 장르 분류 모델 훈련 (분리된 데이터셋)")
    print("=" * 60)
    
    # 설정
    CONFIG = {
        'model_name': 'bert-base-multilingual-cased',
        'max_length': 512,
        'batch_size': 32,  # 배치 크기 증가
        'epochs': 50,       # 에포크 증가
        'learning_rate': 2e-5,  # 학습률 증가
        'weight_decay': 0.01,
        'warmup_ratio': 0.15   # 워밍업 비율 증가
    }
    
    print(f"⚙️ 설정:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    try:
        # 트레이너 초기화
        trainer = ImprovedMovieGenreClassifierTrainer(
            model_name=CONFIG['model_name'],
            max_length=CONFIG['max_length']
        )
        
        # 훈련 데이터 로드
        train_df = trainer.load_train_data('improved_train_data.csv')
        if train_df is None:
            print("❌ 훈련 데이터 로드 실패.")
            return
        
        # 평가 데이터 로드
        eval_df = trainer.load_evaluation_data('improved_evaluation_data.csv')
        if eval_df is None:
            print("❌ 평가 데이터 로드 실패.")
            return
        
        # 데이터 준비
        train_texts, train_labels = trainer.prepare_train_data(train_df)
        eval_texts, eval_labels = trainer.prepare_evaluation_data(eval_df)
        
        # 데이터 로더 생성
        train_loader, eval_loader = trainer.create_data_loaders(
            train_texts, train_labels, eval_texts, eval_labels,
            batch_size=CONFIG['batch_size']
        )
        
        # 모델 초기화
        model = trainer.initialize_model()
        
        # 모델 훈련
        train_losses, eval_accuracies = trainer.train_model(
            train_loader, eval_loader, train_labels,
            epochs=CONFIG['epochs'],
            learning_rate=CONFIG['learning_rate'],
            weight_decay=CONFIG['weight_decay'],
            warmup_ratio=CONFIG['warmup_ratio']
        )
        
        # 상세 평가
        final_accuracy, classification_report = trainer.detailed_evaluation(eval_loader)
        
        # 모델 저장
        trainer.save_model()
        
        # 예측 예시
        print(f"\n🔮 예측 예시:")
        sample_texts = [
            "주인공이 악마와 싸우며 세상을 구하는 판타지 액션 이야기",
            "두 남녀가 운명적으로 만나 사랑에 빠지는 로맨틱한 멜로드라마",
            "미래에서 온 로봇이 인류를 위협하는 SF 액션 스릴러",
            "가족이 함께 모험을 떠나는 따뜻한 가족 영화",
            "무서운 귀신이 나타나는 공포 호러 영화"
        ]
        
        for text in sample_texts:
            predicted_genre, confidence, top3 = trainer.predict_text(text)
            print(f"\n  📝 '{text[:40]}...'")
            print(f"     🥇 1위: {predicted_genre} (신뢰도: {confidence:.3f})")
            print(f"     📊 상위 3개:")
            for i, (genre, prob) in enumerate(top3):
                print(f"        {i+1}. {genre}: {prob:.3f}")
        
        print(f"\n🎉 모델 훈련 완료!")
        print(f"📊 최종 정확도: {final_accuracy:.4f}")
        
        # 최종 요약
        print(f"\n📈 최종 요약:")
        print(f"  훈련 데이터: {len(train_texts)}개")
        print(f"  평가 데이터: {len(eval_texts)}개")
        print(f"  장르 수: {len(trainer.label_encoder.classes_)}개")
        print(f"  최고 평가 정확도: {max(eval_accuracies):.4f}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import pickle
import warnings
warnings.filterwarnings('ignore')

class BertGenreClassifier(nn.Module):
    """BERT 기반 장르 분류 모델 (로드용)"""
    
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

class MovieGenrePredictor:
    """훈련된 모델로 영화 장르 예측하는 클래스"""
    
    def __init__(self, model_path='movie_genre_bert_model.pth', 
                 tokenizer_path='movie_genre_tokenizer', 
                 label_encoder_path='label_encoder.pkl'):
        print("🎬 영화 장르 예측기 초기화 중...")
        print("=" * 60)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 사용 디바이스: {self.device}")
        
        # 모델 설정 로드
        print("📂 모델 로드 중...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model_config = checkpoint['model_config']
        
        print(f"📊 모델 정보:")
        print(f"  - 클래스 수: {self.model_config['n_classes']}")
        print(f"  - 모델명: {self.model_config['model_name']}")
        print(f"  - 최대 길이: {self.model_config['max_length']}")
        
        # 토크나이저 로드
        print("🔤 토크나이저 로드 중...")
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        # 라벨 인코더 로드
        print("🏷️ 라벨 인코더 로드 중...")
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"🎭 학습된 장르 목록:")
        for i, genre in enumerate(self.label_encoder.classes_):
            print(f"  {i}: {genre}")
        
        # 모델 초기화 및 가중치 로드
        print("🤖 모델 초기화 중...")
        self.model = BertGenreClassifier(
            n_classes=self.model_config['n_classes'],
            model_name=self.model_config['model_name']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ 예측기 초기화 완료!")
        print("=" * 60)
    
    def predict_single_text(self, text, show_probabilities=True):
        """단일 텍스트의 장르 예측"""
        # 텍스트 전처리 및 토큰화
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.model_config['max_length'],
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 예측 수행
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)
        
        # 결과 해석
        predicted_genre = self.label_encoder.inverse_transform([predicted.cpu().numpy()[0]])[0]
        confidence = probabilities[0][predicted].item()
        
        # 상위 3개 예측 결과
        top3_probs, top3_indices = torch.topk(probabilities[0], min(3, len(self.label_encoder.classes_)))
        top3_genres = self.label_encoder.inverse_transform(top3_indices.cpu().numpy())
        top3_results = list(zip(top3_genres, top3_probs.cpu().numpy()))
        
        result = {
            'text': text,
            'predicted_genre': predicted_genre,
            'confidence': confidence,
            'top3_predictions': top3_results
        }
        
        if show_probabilities:
            # 모든 장르별 확률
            all_probs = probabilities[0].cpu().numpy()
            all_genres = self.label_encoder.classes_
            all_predictions = list(zip(all_genres, all_probs))
            all_predictions.sort(key=lambda x: x[1], reverse=True)
            result['all_predictions'] = all_predictions
        
        return result
    
    def predict_multiple_texts(self, texts, show_details=True):
        """여러 텍스트의 장르 예측"""
        print(f"\n🎬 {len(texts)}개 텍스트 장르 예측 중...")
        print("=" * 60)
        
        results = []
        for i, text in enumerate(texts):
            print(f"\n📝 예측 {i+1}/{len(texts)}:")
            print(f"텍스트: {text[:60]}{'...' if len(text) > 60 else ''}")
            
            result = self.predict_single_text(text, show_probabilities=False)
            results.append(result)
            
            if show_details:
                print(f"🎯 예측 장르: {result['predicted_genre']}")
                print(f"📊 신뢰도: {result['confidence']:.3f}")
                print(f"🏆 상위 3개:")
                for j, (genre, prob) in enumerate(result['top3_predictions']):
                    print(f"   {j+1}. {genre}: {prob:.3f}")
        
        return results
    
    def predict_from_csv(self, csv_file, text_column='줄거리', output_file='predictions.csv'):
        """CSV 파일에서 텍스트를 읽어 예측하고 결과 저장"""
        print(f"\n📁 CSV 파일에서 예측 수행: {csv_file}")
        print("=" * 60)
        
        # CSV 파일 읽기
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            print(f"✅ 파일 로드 완료: {len(df)}개 행")
            print(f"📋 컬럼: {list(df.columns)}")
            
            if text_column not in df.columns:
                print(f"❌ '{text_column}' 컬럼을 찾을 수 없습니다.")
                return None
            
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {csv_file}")
            return None
        
        # 예측 수행
        texts = df[text_column].tolist()
        predictions = []
        confidences = []
        top2_genres = []
        top2_confidences = []
        
        print(f"\n🔄 예측 진행 중...")
        for i, text in enumerate(texts):
            if pd.isna(text):
                predictions.append("알 수 없음")
                confidences.append(0.0)
                top2_genres.append("알 수 없음")
                top2_confidences.append(0.0)
                continue
            
            result = self.predict_single_text(str(text), show_probabilities=False)
            predictions.append(result['predicted_genre'])
            confidences.append(result['confidence'])
            
            # 2위 예측 결과
            if len(result['top3_predictions']) >= 2:
                top2_genres.append(result['top3_predictions'][1][0])
                top2_confidences.append(result['top3_predictions'][1][1])
            else:
                top2_genres.append(result['predicted_genre'])
                top2_confidences.append(result['confidence'])
            
            if (i + 1) % 10 == 0:
                print(f"  진행률: {i+1}/{len(texts)} ({(i+1)/len(texts)*100:.1f}%)")
        
        # 결과를 데이터프레임에 추가
        df_result = df.copy()
        df_result['예측_장르'] = predictions
        df_result['신뢰도'] = confidences
        df_result['2위_장르'] = top2_genres
        df_result['2위_신뢰도'] = top2_confidences
        
        # 결과 저장
        df_result.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 예측 완료!")
        print(f"📁 결과 저장: {output_file}")
        
        # 예측 결과 요약
        print(f"\n📊 예측 결과 요약:")
        prediction_counts = pd.Series(predictions).value_counts()
        for genre, count in prediction_counts.items():
            print(f"  {genre}: {count}개 ({count/len(predictions)*100:.1f}%)")
        
        print(f"\n📈 평균 신뢰도: {np.mean([c for c in confidences if c > 0]):.3f}")
        
        return df_result
    
    def interactive_prediction(self):
        """대화형 예측 모드"""
        print("\n🎮 대화형 장르 예측 모드")
        print("=" * 60)
        print("영화 줄거리를 입력하면 장르를 예측해드립니다.")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print("=" * 60)
        
        while True:
            text = input("\n📝 영화 줄거리를 입력하세요: ").strip()
            
            if text.lower() in ['quit', 'exit', '종료', 'q']:
                print("👋 예측기를 종료합니다.")
                break
            
            if not text:
                print("⚠️ 텍스트를 입력해주세요.")
                continue
            
            print(f"\n🔄 예측 중...")
            result = self.predict_single_text(text)
            
            print(f"\n🎯 예측 결과:")
            print(f"  📝 입력 텍스트: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"  🏆 예측 장르: {result['predicted_genre']}")
            print(f"  📊 신뢰도: {result['confidence']:.3f}")
            
            print(f"\n🏅 상위 3개 예측:")
            for i, (genre, prob) in enumerate(result['top3_predictions']):
                emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                print(f"    {emoji} {genre}: {prob:.3f}")
            
            if 'all_predictions' in result:
                print(f"\n📈 전체 장르별 확률:")
                for genre, prob in result['all_predictions']:
                    bar_length = int(prob * 20)  # 최대 20글자 막대
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    print(f"    {genre:<12} {bar} {prob:.3f}")

def main():
    """메인 실행 함수"""
    print("🎬 영화 장르 예측 시스템")
    print("=" * 60)
    
    try:
        # 예측기 초기화
        predictor = MovieGenrePredictor()
        
        # 사용 예시 선택
        print("\n📋 사용 방법을 선택하세요:")
        print("1. 직접 입력한 텍스트 예측")
        print("2. 여러 샘플 텍스트 예측")
        print("3. CSV 파일에서 예측")
        print("4. 대화형 예측 모드")
        
        choice = input("\n선택 (1-4): ").strip()
        
        if choice == "1":
            # 단일 텍스트 예측
            sample_text = "주인공이 악마와 싸우며 세상을 구하는 판타지 액션 이야기입니다. 마법과 검술을 사용하여 강력한 적들과 맞서 싸우며, 동료들과 함께 모험을 떠납니다."
            
            print(f"\n📝 샘플 텍스트 예측:")
            result = predictor.predict_single_text(sample_text)
            
            print(f"\n🎯 예측 결과:")
            print(f"  텍스트: {result['text']}")
            print(f"  예측 장르: {result['predicted_genre']}")
            print(f"  신뢰도: {result['confidence']:.3f}")
            
            print(f"\n🏆 상위 3개:")
            for i, (genre, prob) in enumerate(result['top3_predictions']):
                print(f"    {i+1}. {genre}: {prob:.3f}")
        
        elif choice == "2":
            # 여러 텍스트 예측
            sample_texts = [
                "두 남녀가 운명적으로 만나 사랑에 빠지는 로맨틱한 이야기입니다.",
                "우주에서 외계인이 지구를 침공하고 인류가 저항하는 SF 영화입니다.",
                "가족이 함께 여행을 떠나며 벌어지는 따뜻하고 감동적인 이야기입니다.",
                "탐정이 연쇄살인 사건의 진실을 파헤치는 스릴러 영화입니다.",
                "코믹한 상황들이 연속으로 펼쳐지는 웃음이 가득한 영화입니다."
            ]
            
            results = predictor.predict_multiple_texts(sample_texts)
        
        elif choice == "3":
            # CSV 파일 예측
            csv_file = input("CSV 파일명을 입력하세요: ").strip()
            text_column = input("텍스트 컬럼명을 입력하세요 (기본값: 줄거리): ").strip()
            if not text_column:
                text_column = "줄거리"
            
            output_file = input("결과 파일명을 입력하세요 (기본값: predictions.csv): ").strip()
            if not output_file:
                output_file = "predictions.csv"
            
            predictor.predict_from_csv(csv_file, text_column, output_file)
        
        elif choice == "4":
            # 대화형 모드
            predictor.interactive_prediction()
        
        else:
            print("❌ 잘못된 선택입니다.")
        
    except FileNotFoundError as e:
        print(f"❌ 필요한 파일을 찾을 수 없습니다: {e}")
        print("💡 먼저 모델 훈련을 실행하세요: python improved_model.py")
    
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
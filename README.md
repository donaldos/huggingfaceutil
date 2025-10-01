---
license: apache-2.0
---
# MySpeechDataset

**MySpeechDataset**은 영어 학습 및 음성인식 연구를 위해 구축된 소규모 음성 데이터셋입니다.  
데이터는 여러 연령대, 성별, 화자가 발화한 오디오 파일과 대응되는 전사(transcript)로 구성되어 있습니다.

## 데이터 구성

- **총 화자 수**: 10명
- **총 발화 수**: 1,000문장
- **언어**: en (English)
- **포맷**: WAV (16kHz, mono, PCM16)

### 디렉토리 구조
dataset/
├── README.md
├── meta.json
└── Audio/{speaker_id}/{utterance_id}.wav

### 메타데이터(`meta.json`)
- `split`: 데이터셋 분할 (train/valid/test)
- `audio_path`: 오디오 파일 상대 경로
- `transcript`: 전사 텍스트
- `speaker_id`: 화자 식별자
- `gender`: 화자 성별 (M/F/Other)
- `age`: 화자 나이 (또는 구간)
- `lang`: 언어 코드 (ISO 639-1)

## 라이선스
MIT License

## 인용
@dataset{myspeechdataset2025,
author = {홍길동 et al.},
title = {MySpeechDataset: A small-scale speech corpus for ASR and education},
year = {2025},
url = {https://huggingface.co/datasets/username/myspeechdataset}
}
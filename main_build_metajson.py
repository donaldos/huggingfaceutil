from __future__ import annotations

import os
import json
from typing import Tuple, List

def walk_path(root_path: str, extension: str) -> Tuple[bool, str | List[str]]:
    """
    지정한 루트 디렉토리 아래에서 특정 확장자 파일을 재귀적으로 탐색합니다.

    Args:
        root_path (str): 검색 시작 루트 경로
        extension (str): 찾고자 하는 파일 확장자 (예: ".wav", ".txt")

    Returns:
        Tuple[bool, str | List[str]]:
            - bool: 성공 여부
            - str | List[str]:
                성공 시: 파일 전체 경로 리스트
                실패 시: 오류 메시지 문자열
    """
    try:
        if not os.path.isdir(root_path):
            return False, f"지정한 루트 경로가 디렉토리가 아닙니다: {root_path}"

        matches: List[str] = []
        for dirpath, _, filenames in os.walk(root_path):
            for fname in filenames:
                if fname.lower().endswith(extension.lower()):
                    full_path = os.path.join(dirpath, fname)
                    matches.append(full_path)

        if not matches:
            return False, f"'{extension}' 확장자를 가진 파일을 찾을 수 없습니다."

        return True, matches

    except Exception as e:
        return False, f"예상치 못한 오류 발생: {e.__class__.__name__}: {e}"
import random
from typing import List, Tuple

def split_dataset(llist: List[str], 
                  train_ratio: float = 0.8, 
                  valid_ratio: float = 0.1, 
                  test_ratio: float = 0.1,
                  shuffle: bool = True,
                  seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    리스트를 train/valid/test 비율로 분할합니다.

    Args:
        llist (List[str]): 분할할 전체 리스트
        train_ratio (float): train 비율 (기본값 0.8)
        valid_ratio (float): valid 비율 (기본값 0.1)
        test_ratio (float): test 비율 (기본값 0.1)
        shuffle (bool): 리스트를 섞을지 여부 (기본값 True)
        seed (int): 랜덤 시드 (재현성 확보용, 기본값 42)

    Returns:
        Tuple[List[str], List[str], List[str]]:
            - train 리스트
            - valid 리스트
            - test 리스트
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, \
        "비율의 합은 1.0 이어야 합니다."

    data = llist.copy()
    if shuffle:
        random.seed(seed)
        random.shuffle(data)

    n = len(data)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train = data[:n_train]
    valid = data[n_train:n_train+n_valid]
    test = data[n_train+n_valid:]

    return train, valid, test
    
from pathlib import Path
from typing import Tuple

def extract_audio_info(path: str, index_text: str) -> Tuple[str, str]:
    """
    주어진 파일 경로에서 특정 기준 디렉토리(`index_text`)를 기준으로
    speaker_id(기준 디렉토리 바로 아래 폴더명)와 audio_path(기준 디렉토리부터 끝까지)를 추출합니다.

    Args:
        path (str): 전체 파일 경로 (예: "/root/data/audio/A0001/A0001_001.wav")
        index_text (str): 기준이 되는 디렉토리명 (예: "audio")

    Returns:
        Tuple[str, str]:
            - speaker_id (str): 기준 디렉토리 바로 하위 폴더명 (예: "A0001")
            - audio_path (str): 기준 디렉토리부터 끝까지의 상대 경로 
                                (예: "audio/A0001/A0001_001.wav")

    Raises:
        ValueError: 경로에 `index_text`가 포함되지 않은 경우
        IndexError: `index_text` 뒤에 speaker_id가 존재하지 않는 경우

    Example:
        >>> extract_audio_info("/User/work/audio/A0001/A0001_001.wav", "audio")
        ("A0001", "audio/A0001/A0001_001.wav")
    """
    full_path = Path(path)
    parts = full_path.parts

    if index_text not in parts:
        raise ValueError(f"경로에 '{index_text}'가 포함되어 있지 않습니다: {path}")

    idx = parts.index(index_text)

    try:
        speaker_id = parts[idx+1]
    except IndexError:
        raise IndexError(f"'{index_text}' 뒤에 speaker_id 디렉토리가 없습니다: {path}")

    audio_path = "/".join(parts[idx:])
    return speaker_id, audio_path


import os, json
from typing import List, Dict

def build_meta_data(filelist: List[str], settype: str) -> List[Dict]:
    """
    오디오 파일 리스트와 대응하는 JSON 메타 파일을 읽어
    meta.jsonl에 들어갈 메타데이터(dict) 리스트를 생성합니다.

    Args:
        filelist (List[str]): 처리할 WAV 파일 경로 리스트
        settype (str): 데이터 분할 유형 (예: "train", "valid", "test")

    Returns:
        List[Dict]: 각 오디오 샘플에 대한 메타데이터 리스트.
            각 항목은 아래 필드를 포함합니다:
            - split (str): 데이터 분할(train/valid/test)
            - audio_path (str): audio 디렉토리부터의 상대 경로
            - transcript (str): 전사 텍스트
            - speaker_id (str): 화자 ID
            - gender (str): 화자 성별 (기본값 "M")
            - age (int): 화자 나이 (기본값 24)
            - lang (str): 언어 코드 (기본값 "en")

    Notes:
        - 각 WAV 파일과 같은 경로에 `.json` 파일이 있어야 합니다.
        - JSON은 최소한 "Transcription" 키를 포함해야 합니다.
        - `extract_audio_info()` 함수로 speaker_id와 audio_path를 추출합니다.
        - gender/age/lang은 현재 하드코딩되어 있으므로 필요시 수정하세요.
    """
    metas = []

    for wavefile in filelist:
        jsonfile = wavefile.replace('.wav', '.json')

        if not os.path.exists(jsonfile):
            # JSON 메타 파일이 없으면 스킵
            continue

        try:
            with open(jsonfile, 'r', encoding='utf-8') as f:
                jsondata = json.load(f)
                transcription_text = jsondata.get('Transcription', "")

            speaker_id, audio_path = extract_audio_info(wavefile, 'audio')

            entry = {
                'split': settype,
                'audio_path': audio_path,
                'transcript': transcription_text,
                'speaker_id': speaker_id,
                'gender': jsondata.get('Gender', 'M'),
                'age': jsondata.get('Age', 24),
                'lang': jsondata.get('Lang', 'en'),
            }

            metas.append(entry)

        except Exception as e:
            print(f"[WARN] {wavefile} 처리 실패: {e}")

    return metas

def save_jsonl(data: list[dict], output_file: str) -> None:
    """
    리스트 데이터를 JSONL 형식으로 저장합니다.

    Args:
        data (list[dict]): JSON으로 직렬화할 딕셔너리 리스트
        output_file (str): 저장할 파일 경로 (예: "meta.jsonl")

    Example:
        >>> save_jsonl([{"a": 1}, {"b": 2}], "out.jsonl")
        # out.jsonl 내용:
        # {"a": 1}
        # {"b": 2}
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
from itertools import chain

def main(audio_root_path: str) -> bool | list[dict]:
    """
    오디오 루트 경로를 입력받아 데이터셋 메타정보를 생성합니다.

    Args:
        audio_root_path (str): WAV 파일을 검색할 최상위 디렉토리 경로.
            예: "/User/mujung/workspace.testbed/huggingfacetest/audio"

    Returns:
        bool | list[dict]:
            - False: wav 파일을 찾지 못하거나 실패한 경우
            - list[dict]: train/valid/test 전체 샘플에 대한 메타데이터 리스트.
                각 dict는 build_meta_data() 결과와 동일한 구조:
                {
                  "split": "train"/"valid"/"test",
                  "audio_path": "audio/A0001/A0001_001.wav",
                  "transcript": "텍스트",
                  "speaker_id": "A0001",
                  "gender": "M",
                  "age": 24,
                  "lang": "en"
                }

    Workflow:
        1. walk_path() → 모든 wav 파일 경로 수집
        2. split_dataset() → train/valid/test 리스트 분할 (기본 80/10/10)
        3. build_meta_data() → 각각의 파일 리스트에서 메타정보 추출
        4. 최종적으로 chain()으로 합쳐서 반환
    """
    ok, wavefiles = walk_path(audio_root_path, 'wav')
    if not ok:
        # wavefiles에 에러 메시지가 담겨 있을 가능성 있음
        return False, wavefiles  

    # 1. train/valid/test 분할
    trainset, validset, testset = split_dataset(wavefiles)

    # 2. 각각 메타데이터 생성
    trains = build_meta_data(trainset, 'train')
    valids = build_meta_data(validset, 'valid')
    tests  = build_meta_data(testset,  'test')

    # 3. 최종 병합
    metainfos = list(chain(trains, valids, tests))

    save_jsonl(metainfos,'./datasets.donaldos/meta.jsonl')


if __name__=='__main__':
    AUDIO_ROOT_PATH = './datasets.donaldos/audio'
    main(AUDIO_ROOT_PATH)
from __future__ import annotations

import os
from typing import Tuple
from huggingface_hub import login, HfApi, upload_file, upload_folder
from huggingface_hub.utils import HfHubHTTPError
from pathlib import Path, PurePosixPath

def login_huggingface(
    token: str | None = None,
    *,
    add_to_git_credential: bool = False,
    validate: bool = True,
) -> Tuple[bool, str]:
    """
    Hugging Face Hub 로그인 유틸.

    Args:
        token (str | None): Hugging Face access token.
            - None일 경우 환경 변수 `HF_TOKEN`을 사용합니다.
        add_to_git_credential (bool): True일 경우 git credential helper에도 토큰을 저장합니다.
            - 보안상 공용 머신에서는 권장하지 않습니다.
        validate (bool): True일 경우 로그인 후 `whoami()`를 호출하여 토큰 유효성을 검증합니다.

    Returns:
        Tuple[bool, str]: (성공 여부, 메시지)

    Note:
        `add_to_git_credential=True` 옵션은 개인 개발 환경에서만 사용하세요.
    """
    try:
        token = token or os.getenv("HF_TOKEN")
        if not token:
            return False, "토큰이 없습니다. 인자로 넘기거나 환경변수 HF_TOKEN을 설정하세요."

        # 토큰 저장(캐시에 기록). 절대 토큰을 print/log에 직접 남기지 마세요.
        login(token=token, add_to_git_credential=add_to_git_credential)

        if validate:
            api = HfApi()
            info = api.whoami(token)
            user = info.get("name") or info.get("user") or "<unknown>"
            return True, f"Hugging Face에 '{user}' 계정으로 로그인했습니다."
        else:
            return True, "토큰 저장 완료(검증 생략)."
    except HfHubHTTPError as e:
        return False, f"Hugging Face API 오류: {e}"
    except Exception as e:  # <- 올바른 문법: 'as e'
        return False, f"예상치 못한 오류: {e.__class__.__name__}: {e}"

def create_repo_safe(
    repo_id: str,
    *,
    private: bool = True,
    exist_ok: bool = True,
    token: str | None = None,   # 로그인 해두었으면 None으로 둬도 됨
) -> Tuple[bool, str]:
    """
    Hugging Face 'dataset' 레포 생성. (이미 있으면 exist_ok=True면 통과)

    Args:
        repo_id (str): 생성하거나 접근할 Hugging Face Hub 레포 ID
            (예: "username/my-dataset").
        private (bool, optional): 레포를 비공개로 생성 여부. 
            기본값은 True.
        exist_ok (bool, optional): 이미 레포가 존재할 경우 에러 없이 통과할지 여부. 
            기본값은 True.
        token (str | None, optional): Hugging Face 인증 토큰. 
            None이면 huggingface-cli 로그인 세션 사용.

    Returns:
        Tuple[bool, str]:
            - bool: 성공 여부 (True = 성공, False = 실패)
            - str : 상태 메시지 (성공/실패 사유)

    Notes:
        - 내부적으로 huggingface_hub 라이브러리의 create_repo API 호출.
        - repo_type은 'dataset'으로 고정되어 있음.
        - exist_ok=True이면 이미 같은 repo_id가 존재해도 에러 발생 없이 성공 처리.
        - 인증 필요. huggingface-cli login 또는 token 인자로 인증 가능.
    """
    try:
        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=exist_ok,
            token=token,
        )
        return True, f"레포 준비 완료: {repo_id}"
    except HfHubHTTPError as e:
        # 상태코드에 따른 간단한 메시지 분기
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code == 401:
            return False, "인증 실패(401): 토큰이 없거나 유효하지 않습니다."
        if code == 403:
            return False, "권한 거부(403): 조직 레포면 쓰기 권한이 있는지 확인하세요."
        if code == 404:
            return False, "대상 없음(404): 조직/사용자 계정 이름을 확인하세요."
        if code == 409:
            return False, "충돌(409): 동일한 이름의 레포가 이미 존재합니다. exist_ok=True이면 통과됩니다."
        return False, f"HfHubHTTPError: {e}"
    except Exception as e:
        return False, f"예상치 못한 오류: {e.__class__.__name__}: {e}"
    
def upload_file_to_hf(
    local_path: str,
    *,
    repo_id: str,
    path_in_repo: str | None = None,   # None이면 파일명 그대로 사용
    repo_type: str = "dataset",
    token: str | None = None,
    commit_message: str | None = None,
) -> Tuple[bool, str]:
    """
    단일 파일을 Hugging Face Hub 레포에 업로드합니다.

    Args:
        local_path (str): 업로드할 로컬 파일 경로.
        repo_id (str, optional): 업로드할 Hugging Face Hub 레포 ID.
            
        path_in_repo (str | None, optional): 레포 내 저장될 경로.
            None이면 로컬 파일명을 그대로 사용.
        repo_type (str, optional): 레포 타입. 
            일반적으로 "dataset" 사용. 기본값은 "dataset".
        token (str | None, optional): Hugging Face 인증 토큰.
            None이면 huggingface-cli 로그인 세션 사용.
        commit_message (str | None, optional): 업로드 시 커밋 메시지.
            예: "add train-0001.parquet"

    Returns:
        Tuple[bool, str]:
            - bool: 업로드 성공 여부 (True = 성공, False = 실패)
            - str : 상태 메시지 (성공/실패 사유)

    Notes:
        - 내부적으로 huggingface_hub의 `upload_file` API를 호출.
        - 파일이 존재하지 않으면 즉시 실패 처리.
        - path_in_repo는 POSIX 경로 형태로 변환되어 업로드됨.
        - 토큰이 없거나 권한 부족, 레포 없음, 파일 크기 초과 등
          `HfHubHTTPError` 코드별로 상세 에러 메시지를 반환.
        - 대용량 파일(>5GB) 업로드 시에는 `upload_folder` 또는
          파일 샤딩을 고려해야 함.
    """
    try:
        # 1) 로컬 파일 존재 검증
        if not os.path.isfile(local_path):
            return False, f"로컬 파일을 찾을 수 없습니다: {local_path}"

        # 2) 레포 내 경로 결정 (로컬 전체 경로를 그대로 쓰지 않도록 주의)
        if path_in_repo is None:
            path_in_repo = PurePosixPath(Path(local_path).name).as_posix()
        else:
            path_in_repo = PurePosixPath(path_in_repo).as_posix()

        # 3) 업로드
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,                   # 이미 login() 했다면 None이어도 OK
            commit_message=commit_message, # 예: "add train-0001.parquet"
        )

        return True, f"업로드 성공 → {repo_id}:{path_in_repo}"

    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code == 401:
            return False, "인증 실패(401): 토큰이 없거나 유효하지 않습니다."
        if code == 403:
            return False, "권한 거부(403): 레포 쓰기 권한을 확인하세요."
        if code == 404:
            return False, "레포 없음(404): repo_id가 맞는지, 레포가 생성되어 있는지 확인하세요."
        if code == 413:
            return False, "파일이 너무 큽니다(413): 폴더 업로드(upload_folder) 또는 샤딩을 고려하세요."
        return False, f"HfHubHTTPError: {e}"
    except FileNotFoundError:
        return False, f"로컬 파일을 찾을 수 없습니다: {local_path}"
    except Exception as e:
        return False, f"예상치 못한 오류: {e.__class__.__name__}: {e}"




def upload_folder_to_hf(
    local_dir: str,
    *,
    repo_id: str,
    path_in_repo: str | None = None,   # None이면 폴더명 그대로
    repo_type: str = "dataset",
    token: str | None = None,
    commit_message: str | None = None,
    ignore_patterns: list[str] | None = None,  # 업로드 제외할 패턴 (예: ["*.tmp", "*.log"])
) -> Tuple[bool, str]:
    """
    로컬 폴더 전체를 Hugging Face Hub 레포에 업로드합니다.

    Args:
        local_dir (str): 업로드할 로컬 폴더 경로.
        repo_id (str): 업로드 대상 Hugging Face Hub 레포 ID.
            (예: "username/my-dataset")
        path_in_repo (str | None, optional): 레포 내 저장될 경로.
            None이면 폴더명을 그대로 사용.
        repo_type (str, optional): 레포 타입. 보통 "dataset". 기본값은 "dataset".
        token (str | None, optional): Hugging Face 인증 토큰.
            None이면 huggingface-cli 로그인 세션 사용.
        commit_message (str | None, optional): 업로드 시 커밋 메시지.
        ignore_patterns (list[str] | None, optional): 업로드 제외할 glob 패턴.

    Returns:
        Tuple[bool, str]:
            - bool: 업로드 성공 여부
            - str : 상태 메시지 (성공/실패 이유)

    Notes:
        - 내부적으로 huggingface_hub의 `upload_folder` API 호출.
        - path_in_repo는 POSIX 경로 형태로 변환됨.
        - ignore_patterns를 사용해 불필요한 파일을 제외 가능.
        - 대용량 파일(>5GB)이 폴더에 포함되어 있으면 업로드 실패 가능.
    """
    try:
        if not os.path.isdir(local_dir):
            return False, f"로컬 폴더를 찾을 수 없습니다: {local_dir}"

        # 레포 내 경로 변환
        if path_in_repo is None:
            path_in_repo = PurePosixPath(Path(local_dir).name).as_posix()
        else:
            path_in_repo = PurePosixPath(path_in_repo).as_posix()

        upload_folder(
            repo_id=repo_id,
            folder_path=local_dir,
            path_in_repo=path_in_repo,
            repo_type=repo_type,
            token=token,
            commit_message=commit_message,
            ignore_patterns=ignore_patterns,
        )
        return True, f"폴더 업로드 성공 → {repo_id}:{path_in_repo}"

    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code == 401:
            return False, "인증 실패(401): 토큰이 없거나 유효하지 않습니다."
        if code == 403:
            return False, "권한 거부(403): 레포 쓰기 권한을 확인하세요."
        if code == 404:
            return False, "레포 없음(404): repo_id가 맞는지, 레포가 생성되어 있는지 확인하세요."
        if code == 413:
            return False, "파일이 너무 큽니다(413): 샤딩 후 업로드를 고려하세요."
        return False, f"HfHubHTTPError: {e}"
    except Exception as e:
        return False, f"예상치 못한 오류: {e.__class__.__name__}: {e}"
import os
from pathlib import Path, PurePosixPath
from typing import Tuple, List, Optional
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

def upload_large_folder_to_hf(
    local_dir: str,
    *,
    repo_id: str,
    path_in_repo: Optional[str] = None,   # None이면 폴더명 그대로
    repo_type: str = "dataset",
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,  # 예: ["*.tmp", "*.log"]
    # 대용량 업로드 제어 옵션
    multi_commits: bool = True,     # 여러 커밋으로 쪼개서 업로드
    files_per_commit: int = 1000,   # 커밋당 포함할 파일 수
    max_workers: int = 8,           # 동시 업로드 스레드 수
) -> Tuple[bool, str]:
    """
    로컬 폴더 전체를 Hugging Face Hub 레포에 업로드합니다.
    대용량 폴더는 HfApi.upload_large_folder를 사용해 다중 커밋으로 안전하게 업로드합니다.

    Args:
        local_dir (str): 업로드할 로컬 폴더 경로.
        repo_id (str): 업로드 대상 Hugging Face Hub 레포 ID. (예: "username/my-dataset")
        path_in_repo (str | None): 레포 내 저장 경로. None이면 폴더명을 그대로 사용.
        repo_type (str): 레포 타입. 보통 "dataset".
        token (str | None): HF 인증 토큰. None이면 기존 로그인 세션 사용.
        commit_message (str | None): 기본 커밋 메시지.
        ignore_patterns (list[str] | None): 업로드에서 제외할 glob 패턴.
        multi_commits (bool): 다중 커밋 사용 여부(대용량 권장).
        files_per_commit (int): 커밋당 파일 수(너무 작게 잡으면 커밋이 너무 쪼개짐).
        max_workers (int): 병렬 업로드 워커 수.

    Returns:
        (bool, str): (성공 여부, 메시지)

    Notes:
        - 내부적으로 `HfApi.upload_large_folder`를 사용합니다.
        - `ignore_patterns`로 불필요한/임시파일을 제외하면 실패 가능성이 줄고 속도도 개선됩니다.
        - `AttributeError`가 나면 huggingface_hub 버전을 업데이트 하세요:
          `pip install -U huggingface_hub`
    """
    try:
        if not os.path.isdir(local_dir):
            return False, f"로컬 폴더를 찾을 수 없습니다: {local_dir}"

        # 레포 내 경로 정규화
        if path_in_repo is None:
            path_in_repo = PurePosixPath(Path(local_dir).name).as_posix()
        else:
            path_in_repo = PurePosixPath(path_in_repo).as_posix()

        api = HfApi(token=token)

        # upload_large_folder 사용 (대용량에 안전)
        if not hasattr(api, "upload_large_folder"):
            return False, (
                "현재 huggingface_hub 버전에서 'upload_large_folder'를 사용할 수 없습니다. "
                "패키지를 업데이트 해주세요: pip install -U huggingface_hub"
            )

        api.upload_large_folder(
            repo_id=repo_id,
            folder_path=local_dir,
            #path_in_repo=path_in_repo,
            repo_type=repo_type,
            #commit_message=commit_message or f"Upload {Path(local_dir).name}",
            ignore_patterns=ignore_patterns,
            #multi_commits=multi_commits,
            #files_per_commit=files_per_commit,
            num_workers=max_workers,
            # 필요 시:
            # create_pr=False,
            # delete_patterns=None,
            # revision=None,
        )

        return True, f"폴더 업로드 성공 → {repo_id}:{path_in_repo}"

    except HfHubHTTPError as e:
        code = getattr(getattr(e, "response", None), "status_code", None)
        if code == 401:
            return False, "인증 실패(401): 토큰이 없거나 유효하지 않습니다."
        if code == 403:
            return False, "권한 거부(403): 레포 쓰기 권한을 확인하세요."
        if code == 404:
            return False, "레포 없음(404): repo_id 또는 레포 생성 여부를 확인하세요."
        if code == 413:
            return False, "요청 페이로드가 너무 큽니다(413): ignore_patterns/분할 업로드를 조정하세요."
        return False, f"HfHubHTTPError: {e}"
    except Exception as e:
        return False, f"예상치 못한 오류: {e.__class__.__name__}: {e}"
    
if __name__=='__main__':
    REPO_ID = 'donaldos-kim/englishcorpus'
    token = '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'
    ok, msg = login_huggingface(token=token,add_to_git_credential=False)  # CI/서버는 보통 False
    if not ok:
        print(msg)
        raise SystemExit(1)
    
    ok, msg = create_repo_safe(REPO_ID)
    if not ok:
        print(msg)
        raise SystemExit(1)
    
    ok, msg = upload_file_to_hf(local_path='./datasets.donaldos/README.md',
                                repo_id=REPO_ID,
                                path_in_repo="README.md",
                                repo_type='dataset',
                                commit_message='add README.md')
    if not ok:
        print('3')
        raise SystemExit(1)
    
    ok, msg = upload_file_to_hf(local_path='./datasets.donaldos/meta.jsonl',
                                repo_id=REPO_ID,
                                path_in_repo="meta.jsonl",
                                repo_type='dataset',
                                commit_message='add meta.jsonl')
    if not ok:
        print(msg)
        raise SystemExit(1)
    
    '''
    ok, msg = upload_folder_to_hf(local_dir='./datasets.donaldos/audio',
                                  repo_id=REPO_ID,
                                  path_in_repo='./audio',
                                  commit_message='add audio folder')    
    if not ok:
        print(msg)
    '''
    ok, msg = upload_large_folder_to_hf(local_dir='./datasets.donaldos/audio',
                                        repo_id=REPO_ID,
                                        path_in_repo='audio',
                                        repo_type='dataset',
                                        commit_message='add large data',
                                        ignore_patterns=['*.json','.DS_Store'],
                                        files_per_commit=1000,
                                        max_workers=4)
    
    
    if not ok:
        print(msg)
    